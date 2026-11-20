import random
import os
from itertools import combinations
from collections import Counter
# Card constants
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

# Hand rankings (higher = better)
HAND_RANKINGS = {
    'High Card': 0,
    'Pair': 1,
    'Two Pair': 2,
    'Three of a Kind': 3,
    'Straight': 4,
    'Flush': 5,
    'Full House': 6,
    'Four of a Kind': 7,
    'Straight Flush': 8,
    'Royal Flush': 9
}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = RANK_VALUES[rank]

    def __repr__(self):
        return f"{self.rank}{SUITS[self.suit]}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

def create_deck():
    return [Card(r, s) for r in RANKS for s in SUITS]

def log_game(text):
    with open("poker_history.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

class HandEvaluator:
    """Evaluates poker hands and determines winners."""

    @staticmethod
    def evaluate_hand(cards):
        """
        Evaluate a 5-card hand and return (ranking, tiebreaker_values).
        """
        ranks = sorted([c.value for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = HandEvaluator._check_straight(ranks)

        # Check for each hand type from best to worst
        if is_straight and is_flush:
            if straight_high == 14:  # Ace high straight flush
                return (HAND_RANKINGS['Royal Flush'], [14])
            return (HAND_RANKINGS['Straight Flush'], [straight_high])

        counts = sorted(rank_counts.values(), reverse=True)
        if counts == [4, 1]:
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return (HAND_RANKINGS['Four of a Kind'], [quad_rank, kicker])

        if counts == [3, 2]:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return (HAND_RANKINGS['Full House'], [trip_rank, pair_rank])

        if is_flush:
            return (HAND_RANKINGS['Flush'], ranks)

        if is_straight:
            return (HAND_RANKINGS['Straight'], [straight_high])

        if counts == [3, 1, 1]:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return (HAND_RANKINGS['Three of a Kind'], [trip_rank] + kickers)

        if counts == [2, 2, 1]:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return (HAND_RANKINGS['Two Pair'], pairs + [kicker])

        if counts == [2, 1, 1, 1]:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return (HAND_RANKINGS['Pair'], [pair_rank] + kickers)

        return (HAND_RANKINGS['High Card'], ranks)

    @staticmethod
    def _check_straight(ranks):
        """Check if ranks form a straight. Returns (is_straight, high_card)."""
        unique_ranks = sorted(set(ranks), reverse=True)

        # Check for wheel (A-2-3-4-5)
        if unique_ranks == [12, 3, 2, 1, 0]:  # A, 5, 4, 3, 2
            return True, 5  # 5-high straight

        if len(unique_ranks) == 5:
            if unique_ranks[0] - unique_ranks[4] == 4:
                return True, unique_ranks[0] + 2  # Convert to card rank (2 = index 0)
        return False, 0

    @staticmethod
    def get_best_hand(hole_cards, board):
        """Find the best 5-card hand from 7 cards."""
        all_cards = hole_cards + board
        best_hand = None
        best_eval = (-1, [])
        for combo in combinations(all_cards, 5):
            eval_result = HandEvaluator.evaluate_hand(list(combo))
            if eval_result > best_eval:
                best_eval = eval_result
                best_hand = list(combo)
        return best_hand, best_eval

    @staticmethod
    def get_hand_name(eval_result):
        """Get the name of a hand from its evaluation."""
        ranking = eval_result[0]
        for name, value in HAND_RANKINGS.items():
            if value == ranking:
                return name
        return "Unknown"

class DarkShark:
    """The poker bot opponent."""
    def __init__(self, stack=500):
        self.stack = stack
        self.name = "DarkShark"

    def get_action(self, game_state):
        """
        Decide action based on game state.
        game_state: dict with keys like 'current_bet', 'pot', 'street', 'my_cards', 'board', etc.
        """
        current_bet = game_state.get('current_bet', 0)
        my_bet = game_state.get('my_bet', 0)
        to_call = current_bet - my_bet
        pot = game_state.get('pot', 0)
        street = game_state.get('street', 'preflop')
        my_cards = game_state.get('my_cards', [])
        board = game_state.get('board', [])

        # Calculate a simple hand strength score
        strength = self._evaluate_strength(my_cards, board, street)

        # Decision logic based on strength and randomness
        decision = random.random()
        aggression = random.random()

        if to_call == 0:
            # Check or bet opportunity
            if strength > 0.7 or (strength > 0.4 and aggression > 0.6):
                bet_size = self._calculate_bet_size(pot, street, strength)
                if bet_size <= self.stack:
                    return ('bet', bet_size)
            return ('check', 0)
        else:
            # Facing a bet
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
            if strength > 0.8:  # Strong hand - raise sometimes
                if aggression > 0.5 and self.stack > current_bet * 2:
                    raise_to = min(current_bet * 2 + random.randint(0, pot // 2), self.stack + my_bet)
                    return ('raise', raise_to)
                return ('call', to_call)
            elif strength > pot_odds + 0.1:  # Decent hand - call
                if to_call <= self.stack:
                    return ('call', to_call)
            elif strength > pot_odds and decision > 0.3:  # Marginal - sometimes call
                if to_call <= self.stack:
                    return ('call', to_call)
            # Occasional bluff raise
            if decision > 0.92 and self.stack > current_bet * 2:
                raise_to = min(current_bet * 2, self.stack + my_bet)
                return ('raise', raise_to)
            return ('fold', 0)

    def _evaluate_strength(self, hole_cards, board, street):
        """Simple hand strength estimation."""
        if not hole_cards:
            return 0.5

        if street == 'preflop':
            return self._preflop_strength(hole_cards)
        else:
            return self._postflop_strength(hole_cards, board)

    def _preflop_strength(self, hole_cards):
        """Estimate preflop hand strength."""
        c1, c2 = hole_cards
        high = max(c1.value, c2.value)
        low = min(c1.value, c2.value)
        is_pair = c1.value == c2.value
        is_suited = c1.suit == c2.suit
        gap = high - low

        strength = 0.2
        # Pair bonus
        if is_pair:
            strength += 0.3 + (high / 12) * 0.3
        # High card bonus
        strength += (high / 12) * 0.2 + (low / 12) * 0.1
        # Suited bonus
        if is_suited:
            strength += 0.08
        # Connectedness
        if gap <= 2:
            strength += 0.05

        return min(strength, 1.0)

    def _postflop_strength(self, hole_cards, board):
        """Estimate postflop hand strength using actual hand evaluation."""
        if len(board) < 3:
            return self._preflop_strength(hole_cards)

        _, eval_result = HandEvaluator.get_best_hand(hole_cards, board)
        ranking = eval_result[0]

        # Map hand rankings to approximate strength
        base_strengths = {
            0: 0.15,  # High Card
            1: 0.35,  # Pair
            2: 0.55,  # Two Pair
            3: 0.65,  # Three of a Kind
            4: 0.75,  # Straight
            5: 0.80,  # Flush
            6: 0.88,  # Full House
            7: 0.95,  # Four of a Kind
            8: 0.98,  # Straight Flush
            9: 1.00,  # Royal Flush
        }
        return base_strengths.get(ranking, 0.3)

    def _calculate_bet_size(self, pot, street, strength):
        """Calculate appropriate bet size."""
        if street == 'preflop':
            base = 15
        else:
            base = pot * 0.5

        # Vary bet size based on strength (polarized somewhat)
        if strength > 0.7:
            multiplier = random.uniform(0.6, 1.0)
        else:
            multiplier = random.uniform(0.4, 0.7)

        bet = int(base * multiplier)
        return max(bet, 5)  # Minimum bet

class PokerGame:
    """Main game controller for heads-up Texas Hold'em."""
    def __init__(self, player_stack=500, bot_stack=500, small_blind=3, big_blind=5):
        self.player_stack = player_stack
        self.bot = DarkShark(bot_stack)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.is_player_button = True  # Button = dealer = small blind in heads-up
        self.hand_number = 0
        self.hand_log = []

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_state(self, street, player_hand, board, pot, player_bet=0, bot_bet=0):
        """Display current game state."""
        print("\n" + "=" * 50)
        print(f"Hand #{self.hand_number} | Street: {street.upper()}")
        print(f"Stacks - You: {self.player_stack} | {self.bot.name}: {self.bot.stack}")
        print("=" * 50)

        if board:
            board_str = ' '.join(str(c) for c in board)
            print(f"\nBoard: [ {board_str} ]")
        else:
            print("\nBoard: [ ]")

        print(f"\nYour hand: {player_hand[0]} {player_hand[1]}")

        print(f"\nPot: {pot}")
        if player_bet > 0 or bot_bet > 0:
            print(f"Your bet this round: {player_bet} | {self.bot.name}'s bet: {bot_bet}")

    def get_player_action(self, current_bet, player_bet, pot):
        """Get and validate player action."""
        to_call = current_bet - player_bet
        min_raise = current_bet + self.big_blind

        while True:
            if to_call == 0:
                prompt = "Action (check, bet <amount>, all-in, quit): "
            else:
                prompt = f"Action (fold, call {to_call}, raise <amount>, all-in, quit): "

            action_input = input(prompt).lower().strip()

            if action_input == 'quit':
                return ('quit', 0)
            if action_input == 'check' and to_call == 0:
                return ('check', 0)
            if action_input == 'fold':
                return ('fold', 0)
            if action_input == 'call' and to_call > 0:
                call_amount = min(to_call, self.player_stack)
                return ('call', call_amount)
            if action_input == 'all-in' or action_input == 'allin':
                if self.player_stack <= to_call:
                    return ('call', self.player_stack)
                else:
                    return ('raise', self.player_stack + player_bet)
            if action_input.startswith('bet ') and to_call == 0:
                try:
                    amount = int(action_input.split()[1])
                    if amount < self.big_blind:
                        print(f"Minimum bet is {self.big_blind}.")
                        continue
                    if amount > self.player_stack:
                        print(f"You only have {self.player_stack}.")
                        continue
                    return ('bet', amount)
                except (ValueError, IndexError):
                    print("Invalid bet format. Use: bet <amount>")
                    continue
            if action_input.startswith('raise '):
                try:
                    amount = int(action_input.split()[1])
                    if amount < min_raise:
                        print(f"Minimum raise is to {min_raise}.")
                        continue
                    raise_cost = amount - player_bet
                    if raise_cost > self.player_stack:
                        print(f"You only have {self.player_stack}. Use 'all-in' to go all in.")
                        continue
                    return ('raise', amount)
                except (ValueError, IndexError):
                    print("Invalid raise format. Use: raise <total amount>")
                    continue

            print("Invalid action. Try again.")

    def betting_round(self, street, player_hand, bot_hand, board, pot, is_player_first):
        """
        Execute a betting round.
        Returns: (new_pot, player_folded, bot_folded, quit_game)
        """
        player_bet = 0
        bot_bet = 0
        current_bet = 0
        last_raiser = None
        actions_taken = 0

        # Determine action order
        actors = ['player', 'bot'] if is_player_first else ['bot', 'player']

        while True:
            for actor in actors:
                # Check if betting is complete
                if actions_taken >= 2 and player_bet == bot_bet:
                    return (pot, False, False, False)

                # Check for all-in situations
                if self.player_stack == 0 or self.bot.stack == 0:
                    return (pot, False, False, False)

                if actor == 'player':
                    self.display_state(street, player_hand, board, pot, player_bet, bot_bet)
                    action, amount = self.get_player_action(current_bet, player_bet, pot)
                    self.hand_log.append(f"[{street}] Player: {action} {amount if amount else ''}")

                    if action == 'quit':
                        return (pot, False, False, True)
                    elif action == 'fold':
                        print("You fold.")
                        return (pot, True, False, False)
                    elif action == 'check':
                        print("You check.")
                        actions_taken += 1
                    elif action == 'call':
                        self.player_stack -= amount
                        player_bet += amount
                        pot += amount
                        print(f"You call {amount}. Pot: {pot}")
                        actions_taken += 1
                    elif action in ('bet', 'raise'):
                        bet_amount = amount - player_bet
                        self.player_stack -= bet_amount
                        player_bet = amount
                        pot += bet_amount
                        current_bet = amount
                        last_raiser = 'player'
                        print(f"You {'bet' if action == 'bet' else 'raise to'} {amount}. Pot: {pot}")
                        actions_taken = 1  # Reset to allow response
                else:
                    # Bot's turn
                    game_state = {
                        'current_bet': current_bet,
                        'my_bet': bot_bet,
                        'pot': pot,
                        'street': street,
                        'my_cards': bot_hand,
                        'board': board,
                    }
                    action, amount = self.bot.get_action(game_state)
                    self.hand_log.append(f"[{street}] {self.bot.name}: {action} {amount if amount else ''}")

                    if action == 'fold':
                        print(f"{self.bot.name} folds.")
                        return (pot, False, True, False)
                    elif action == 'check':
                        print(f"{self.bot.name} checks.")
                        actions_taken += 1
                    elif action == 'call':
                        call_amount = min(amount, self.bot.stack)
                        self.bot.stack -= call_amount
                        bot_bet += call_amount
                        pot += call_amount
                        print(f"{self.bot.name} calls {call_amount}. Pot: {pot}")
                        actions_taken += 1
                    elif action in ('bet', 'raise'):
                        bet_amount = min(amount - bot_bet, self.bot.stack)
                        actual_total = bot_bet + bet_amount
                        self.bot.stack -= bet_amount
                        bot_bet = actual_total
                        pot += bet_amount
                        current_bet = actual_total
                        last_raiser = 'bot'
                        print(f"{self.bot.name} {'bets' if action == 'bet' else 'raises to'} {actual_total}. Pot: {pot}")
                        actions_taken = 1  # Reset to allow response

            # Check end condition after both have acted
            if player_bet == bot_bet and actions_taken >= 2:
                return (pot, False, False, False)

    def preflop_betting(self, player_hand, bot_hand, pot):
        """Handle preflop betting with blinds already posted."""
        player_bet = self.small_blind if self.is_player_button else self.big_blind
        bot_bet = self.big_blind if self.is_player_button else self.small_blind
        current_bet = self.big_blind

        # In heads-up, button (SB) acts first preflop
        is_player_first = self.is_player_button
        actors = ['player', 'bot'] if is_player_first else ['bot', 'player']
        actions_taken = 0
        last_aggressor = None

        while True:
            for actor in actors:
                # End conditions
                if actions_taken >= 2 and player_bet == bot_bet:
                    return (pot, False, False, False)
                if self.player_stack == 0 or self.bot.stack == 0:
                    return (pot, False, False, False)

                if actor == 'player':
                    self.display_state('preflop', player_hand, [], pot, player_bet, bot_bet)
                    position = "Button/SB" if self.is_player_button else "Big Blind"
                    print(f"Your position: {position}")
                    action, amount = self.get_player_action(current_bet, player_bet, pot)
                    self.hand_log.append(f"[preflop] Player: {action} {amount if amount else ''}")

                    if action == 'quit':
                        return (pot, False, False, True)
                    elif action == 'fold':
                        print("You fold.")
                        return (pot, True, False, False)
                    elif action == 'check':
                        print("You check.")
                        actions_taken += 1
                    elif action == 'call':
                        to_call = min(current_bet - player_bet, self.player_stack)
                        self.player_stack -= to_call
                        player_bet += to_call
                        pot += to_call
                        print(f"You call {to_call}. Pot: {pot}")
                        actions_taken += 1
                    elif action in ('bet', 'raise'):
                        raise_to = amount
                        cost = raise_to - player_bet
                        self.player_stack -= cost
                        player_bet = raise_to
                        pot += cost
                        current_bet = raise_to
                        last_aggressor = 'player'
                        print(f"You raise to {raise_to}. Pot: {pot}")
                        actions_taken = 1
                else:
                    # Bot's turn
                    game_state = {
                        'current_bet': current_bet,
                        'my_bet': bot_bet,
                        'pot': pot,
                        'street': 'preflop',
                        'my_cards': bot_hand,
                        'board': [],
                    }
                    action, amount = self.bot.get_action(game_state)
                    self.hand_log.append(f"[preflop] {self.bot.name}: {action} {amount if amount else ''}")

                    if action == 'fold':
                        print(f"{self.bot.name} folds.")
                        return (pot, False, True, False)
                    elif action == 'check':
                        print(f"{self.bot.name} checks.")
                        actions_taken += 1
                    elif action == 'call':
                        to_call = min(current_bet - bot_bet, self.bot.stack)
                        self.bot.stack -= to_call
                        bot_bet += to_call
                        pot += to_call
                        print(f"{self.bot.name} calls {to_call}. Pot: {pot}")
                        actions_taken += 1
                    elif action in ('bet', 'raise'):
                        raise_to = min(amount, self.bot.stack + bot_bet)
                        cost = raise_to - bot_bet
                        self.bot.stack -= cost
                        bot_bet = raise_to
                        pot += cost
                        current_bet = raise_to
                        last_aggressor = 'bot'
                        print(f"{self.bot.name} raises to {raise_to}. Pot: {pot}")
                        actions_taken = 1

            if player_bet == bot_bet and actions_taken >= 2:
                return (pot, False, False, False)

    def showdown(self, player_hand, bot_hand, board, pot):
        """Determine winner at showdown and distribute pot."""
        print("\n" + "=" * 50)
        print("SHOWDOWN!")
        print("=" * 50)

        print(f"\nYour hand: {player_hand[0]} {player_hand[1]}")
        print(f"{self.bot.name}'s hand: {bot_hand[0]} {bot_hand[1]}")
        print(f"Board: {' '.join(str(c) for c in board)}")

        player_best, player_eval = HandEvaluator.get_best_hand(player_hand, board)
        bot_best, bot_eval = HandEvaluator.get_best_hand(bot_hand, board)

        player_hand_name = HandEvaluator.get_hand_name(player_eval)
        bot_hand_name = HandEvaluator.get_hand_name(bot_eval)

        print(f"\nYour best hand: {player_hand_name}")
        print(f"{self.bot.name}'s best hand: {bot_hand_name}")

        if player_eval > bot_eval:
            print(f"\n🎉 You win {pot} chips with {player_hand_name}!")
            self.player_stack += pot
            self.hand_log.append(f"Player wins {pot} with {player_hand_name}")
        elif bot_eval > player_eval:
            print(f"\n😔 {self.bot.name} wins {pot} chips with {bot_hand_name}.")
            self.bot.stack += pot
            self.hand_log.append(f"{self.bot.name} wins {pot} with {bot_hand_name}")
        else:
            # Split pot
            split = pot // 2
            print(f"\n🤝 Split pot! Both have {player_hand_name}. Each gets {split} chips.")
            self.player_stack += split
            self.bot.stack += pot - split
            self.hand_log.append(f"Split pot: {split} each")

    def play_hand(self):
        """Play a single hand of poker."""
        self.hand_number += 1
        self.hand_log = [f"\n{'='*40}\nHand #{self.hand_number}\n{'='*40}"]

        # Initialize deck and deal
        deck = create_deck()
        random.shuffle(deck)
        player_hand = [deck.pop(), deck.pop()]
        bot_hand = [deck.pop(), deck.pop()]

        self.hand_log.append(f"Player hand: {player_hand}")
        self.hand_log.append(f"Player position: {'Button/SB' if self.is_player_button else 'BB'}")

        # Post blinds
        if self.is_player_button:
            player_blind = self.small_blind
            bot_blind = self.big_blind
        else:
            player_blind = self.big_blind
            bot_blind = self.small_blind

        self.player_stack -= player_blind
        self.bot.stack -= bot_blind
        pot = player_blind + bot_blind
        self.hand_log.append(f"Blinds posted - Player: {player_blind}, Bot: {bot_blind}")

        self.clear_screen()
        print(f"\n🃏 Hand #{self.hand_number} | Blinds: {self.small_blind}/{self.big_blind}")
        print(f"Your position: {'Button (Small Blind)' if self.is_player_button else 'Big Blind'}")

        # Preflop
        pot, player_folded, bot_folded, quit_game = self.preflop_betting(player_hand, bot_hand, pot)
        if quit_game:
            return False
        if player_folded:
            self.bot.stack += pot
            self.hand_log.append(f"Player folds preflop. {self.bot.name} wins {pot}")
            input("\nPress Enter for next hand...")
            return True
        if bot_folded:
            self.player_stack += pot
            self.hand_log.append(f"{self.bot.name} folds preflop. Player wins {pot}")
            input("\nPress Enter for next hand...")
            return True

        # Flop
        board = [deck.pop(), deck.pop(), deck.pop()]
        print(f"\n--- FLOP: {board[0]} {board[1]} {board[2]} ---")
        self.hand_log.append(f"Flop: {board}")
        # Post-flop, BB acts first (opposite of preflop)
        is_player_first = not self.is_player_button
        pot, player_folded, bot_folded, quit_game = self.betting_round(
            'flop', player_hand, bot_hand, board, pot, is_player_first
        )
        if quit_game:
            return False
        if player_folded:
            self.bot.stack += pot
            self.hand_log.append(f"Player folds on flop. {self.bot.name} wins {pot}")
            input("\nPress Enter for next hand...")
            return True
        if bot_folded:
            self.player_stack += pot
            self.hand_log.append(f"{self.bot.name} folds on flop. Player wins {pot}")
            input("\nPress Enter for next hand...")
            return True

        # Turn
        board.append(deck.pop())
        print(f"\n--- TURN: {' '.join(str(c) for c in board)} ---")
        self.hand_log.append(f"Turn: {board[-1]}")
        pot, player_folded, bot_folded, quit_game = self.betting_round(
            'turn', player_hand, bot_hand, board, pot, is_player_first
        )
        if quit_game:
            return False
        if player_folded:
            self.bot.stack += pot
            self.hand_log.append(f"Player folds on turn. {self.bot.name} wins {pot}")
            input("\nPress Enter for next hand...")
            return True
        if bot_folded:
            self.player_stack += pot
            self.hand_log.append(f"{self.bot.name} folds on turn. Player wins {pot}")
            input("\nPress Enter for next hand...")
            return True

        # River
        board.append(deck.pop())
        print(f"\n--- RIVER: {' '.join(str(c) for c in board)} ---")
        self.hand_log.append(f"River: {board[-1]}")
        pot, player_folded, bot_folded, quit_game = self.betting_round(
            'river', player_hand, bot_hand, board, pot, is_player_first
        )
        if quit_game:
            return False
        if player_folded:
            self.bot.stack += pot
            self.hand_log.append(f"Player folds on river. {self.bot.name} wins {pot}")
            input("\nPress Enter for next hand...")
            return True
        if bot_folded:
            self.player_stack += pot
            self.hand_log.append(f"{self.bot.name} folds on river. Player wins {pot}")
            input("\nPress Enter for next hand...")
            return True

        # Showdown
        self.showdown(player_hand, bot_hand, board, pot)
        input("\nPress Enter for next hand...")
        return True

    def run(self):
        """Main game loop."""
        print("\n" + "=" * 60)
        print(" Welcome to Texas Hold'em Heads-Up vs DarkShark!")
        print("=" * 60)
        print(f"\nStarting stacks: You: {self.player_stack} | DarkShark: {self.bot.stack}")
        print(f"Blinds: {self.small_blind}/{self.big_blind}")
        print("\nCommands: check, fold, call, bet <amount>, raise <amount>, all-in, quit")
        input("\nPress Enter to start...")

        while self.player_stack > 0 and self.bot.stack > 0:
            self.clear_screen()
            continue_game = self.play_hand()
            if not continue_game:
                break

            # Log hand to file
            log_game("\n".join(self.hand_log))

            # Rotate button
            self.is_player_button = not self.is_player_button

            # Check for bust
            if self.player_stack <= 0:
                print("\n💀 You're busted! DarkShark wins the match!")
                break
            elif self.bot.stack <= 0:
                print("\n🏆 DarkShark is busted! You win the match!")
                break

        print("\n" + "=" * 60)
        print("GAME OVER")
        print(f"Final stacks - You: {self.player_stack} | DarkShark: {self.bot.stack}")
        print("=" * 60)
        print("\nThanks for playing! Hand history saved to poker_history.txt")

if __name__ == "__main__":
    game = PokerGame(player_stack=500, bot_stack=500, small_blind=3, big_blind=5)
    game.run()
