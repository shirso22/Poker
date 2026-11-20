#pragma once
// No-limit hold'em hand engine, 2..10 players.
//
// Rules implemented:
//  - Antes (dead), small/big blind; heads-up: button posts SB, acts first preflop, last postflop.
//  - Short blinds/antes post all-in for what the player has; others still owe the full BB.
//  - Min bet = BB. Min raise = previous full raise increment (raise-to semantics).
//  - Short all-in raises do not reopen betting for players who already acted, but
//    several short all-ins that together add up to a full raise do reopen it (TDA rule).
//  - BB option preflop; check-around; everyone all-in -> board runs out automatically.
//  - Side pots built from total contributions; uncalled excess comes back automatically
//    as a pot only its owner is eligible for. Odd chips go to winners in seat order
//    starting left of the button.
//  - Folding when check is available is not allowed (keeps the action space clean).
//
// Hands are dealt from a caller-supplied deck order (deterministic, test-friendly):
//   player i holds deck[2i], deck[2i+1]; board = deck[2n .. 2n+4]. Burn cards are irrelevant.

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "poker/eval.hpp"

namespace poker {

constexpr int kMaxPlayers = 10;
using Chips = int64_t;
using Deck = std::array<Card, 52>;

enum class Street : uint8_t { Preflop = 0, Flop, Turn, River, Done };
enum class ActionType : uint8_t { Fold, Check, Call, Raise };  // Raise = bet or raise; amount = raise-to

struct Action {
    ActionType type = ActionType::Check;
    Chips amount = 0;  // only for Raise: total street commitment after the action

    static Action fold() { return {ActionType::Fold, 0}; }
    static Action check() { return {ActionType::Check, 0}; }
    static Action call() { return {ActionType::Call, 0}; }
    static Action raise_to(Chips to) { return {ActionType::Raise, to}; }
};

struct LegalActions {
    bool can_fold = false;
    bool can_check = false;
    bool can_call = false;
    Chips call_cost = 0;       // chips added by calling (capped at stack)
    bool can_raise = false;
    Chips min_raise_to = 0;    // == max_raise_to when only an all-in short raise is possible
    Chips max_raise_to = 0;    // all-in
};

struct TableConfig {
    int num_players = 2;
    Chips small_blind = 1;
    Chips big_blind = 2;
    Chips ante = 0;
};

struct ActionRecord {
    int player;
    Street street;
    ActionType type;
    Chips amount_to;  // street commitment after the action
    Chips added;      // chips moved from stack to pot
    bool all_in;
};

enum class ApplyError { Ok, HandOver, IllegalAction, BadAmount };

class Hand {
public:
    Hand(const TableConfig& cfg, std::span<const Chips> stacks, int button, const Deck& deck)
        : cfg_(cfg), n_(cfg.num_players), button_(button), deck_(deck) {
        for (int i = 0; i < n_; ++i) {
            initial_[i] = stacks[i];
            stack_[i] = stacks[i];
        }
        post_forced_bets();
        begin_betting(/*preflop=*/true);
    }

    // ---- queries ----
    int num_players() const { return n_; }
    int button() const { return button_; }
    Street street() const { return street_; }
    bool is_over() const { return street_ == Street::Done; }
    int to_act() const { return to_act_; }
    Chips current_bet() const { return current_bet_; }
    Chips last_full_raise() const { return last_full_raise_; }
    Chips stack(int i) const { return stack_[i]; }
    Chips street_commit(int i) const { return street_commit_[i]; }
    Chips total_commit(int i) const { return total_commit_[i]; }
    bool folded(int i) const { return folded_[i]; }
    bool all_in(int i) const { return !folded_[i] && stack_[i] == 0; }
    Chips pot() const {
        Chips p = 0;
        for (int i = 0; i < n_; ++i) p += total_commit_[i];
        return p;
    }
    std::array<Card, 2> hole(int i) const { return {deck_[2 * i], deck_[2 * i + 1]}; }
    int board_size() const { return board_size_; }
    Card board(int k) const { return deck_[2 * n_ + k]; }
    const std::vector<ActionRecord>& history() const { return history_; }

    int small_blind_seat() const { return n_ == 2 ? button_ : next_seat(button_); }
    int big_blind_seat() const { return next_seat(small_blind_seat()); }

    // Results (valid once is_over()).
    Chips won(int i) const { return won_[i]; }
    Chips final_stack(int i) const { return stack_[i]; }
    Chips net(int i) const { return stack_[i] - initial_[i]; }
    bool went_to_showdown() const { return showdown_; }

    LegalActions legal_actions() const {
        LegalActions la;
        if (is_over()) return la;
        const int i = to_act_;
        const Chips to_call = current_bet_ - street_commit_[i];
        la.can_check = to_call == 0;
        la.can_fold = to_call > 0;
        la.can_call = to_call > 0;
        la.call_cost = to_call < stack_[i] ? to_call : stack_[i];
        const Chips max_to = street_commit_[i] + stack_[i];
        if (max_to > current_bet_ && raise_reopened(i)) {
            la.can_raise = true;
            la.max_raise_to = max_to;
            Chips min_to = current_bet_ + last_full_raise_;
            la.min_raise_to = max_to < min_to ? max_to : min_to;
        }
        return la;
    }

    ApplyError apply(Action a) {
        if (is_over()) return ApplyError::HandOver;
        const LegalActions la = legal_actions();
        const int i = to_act_;
        Chips added = 0;
        switch (a.type) {
            case ActionType::Fold:
                if (!la.can_fold) return ApplyError::IllegalAction;
                folded_[i] = true;
                break;
            case ActionType::Check:
                if (!la.can_check) return ApplyError::IllegalAction;
                break;
            case ActionType::Call:
                if (!la.can_call) return ApplyError::IllegalAction;
                added = la.call_cost;
                break;
            case ActionType::Raise: {
                if (!la.can_raise) return ApplyError::IllegalAction;
                if (a.amount < la.min_raise_to || a.amount > la.max_raise_to) return ApplyError::BadAmount;
                added = a.amount - street_commit_[i];
                const Chips increment = a.amount - current_bet_;
                if (increment >= last_full_raise_) last_full_raise_ = increment;  // full raise
                current_bet_ = a.amount;
                break;
            }
        }
        move_chips(i, added);
        has_acted_[i] = true;
        bet_seen_[i] = current_bet_;
        history_.push_back({i, street_, a.type, street_commit_[i], added, stack_[i] == 0 && !folded_[i]});
        advance();
        return ApplyError::Ok;
    }

private:
    int next_seat(int i) const { return (i + 1) % n_; }
    bool can_act(int i) const { return !folded_[i] && stack_[i] > 0; }
    bool needs_to_act(int i) const {
        return can_act(i) && (!has_acted_[i] || street_commit_[i] < current_bet_);
    }
    // A player who already acted may raise again only if the bet has grown by at
    // least one full raise since their last action.
    bool raise_reopened(int i) const {
        return !has_acted_[i] || current_bet_ - bet_seen_[i] >= last_full_raise_;
    }
    int count_live() const {
        int c = 0;
        for (int i = 0; i < n_; ++i) c += !folded_[i];
        return c;
    }
    int count_can_act() const {
        int c = 0;
        for (int i = 0; i < n_; ++i) c += can_act(i);
        return c;
    }

    void move_chips(int i, Chips amt) {
        stack_[i] -= amt;
        street_commit_[i] += amt;
        total_commit_[i] += amt;
    }

    void post_forced_bets() {
        if (cfg_.ante > 0)
            for (int i = 0; i < n_; ++i) {
                Chips a = cfg_.ante < stack_[i] ? cfg_.ante : stack_[i];
                stack_[i] -= a;
                total_commit_[i] += a;  // dead money: not part of the street bet
            }
        auto post = [&](int seat, Chips amt) { move_chips(seat, amt < stack_[seat] ? amt : stack_[seat]); };
        post(small_blind_seat(), cfg_.small_blind);
        post(big_blind_seat(), cfg_.big_blind);
    }

    void begin_betting(bool preflop) {
        for (int i = 0; i < n_; ++i) {
            has_acted_[i] = false;
            bet_seen_[i] = 0;
            if (!preflop) street_commit_[i] = 0;
        }
        current_bet_ = preflop ? cfg_.big_blind : 0;
        last_full_raise_ = cfg_.big_blind;
        to_act_ = preflop ? big_blind_seat() : button_;  // search starts after this seat
        advance();
    }

    // Moves to the next player needing action, or finishes the street/hand.
    void advance() {
        if (count_live() == 1) return finish();
        for (int k = 1; k <= n_; ++k) {
            int s = (to_act_ + k) % n_;
            if (needs_to_act(s)) {
                to_act_ = s;
                return;
            }
        }
        // Betting round complete.
        if (street_ == Street::River || count_can_act() <= 1) {
            board_size_ = 5;  // run out remaining cards
            return finish();
        }
        street_ = Street(int(street_) + 1);
        board_size_ = street_ == Street::Flop ? 3 : board_size_ + 1;
        begin_betting(false);
    }

    void finish() {
        street_ = Street::Done;
        to_act_ = -1;
        if (count_live() == 1) {
            for (int i = 0; i < n_; ++i)
                if (!folded_[i]) award(i, pot());
            return;
        }
        showdown_ = true;
        HandMask board = 0;
        for (int k = 0; k < 5; ++k) board |= card_bit(this->board(k));
        uint32_t strength[kMaxPlayers] = {};
        for (int i = 0; i < n_; ++i)
            if (!folded_[i]) strength[i] = evaluate(board | card_bit(deck_[2 * i]) | card_bit(deck_[2 * i + 1]));

        // Pot layers at each distinct contribution level of a live player.
        Chips prev = 0;
        while (true) {
            Chips level = -1;
            for (int i = 0; i < n_; ++i)
                if (!folded_[i] && total_commit_[i] > prev && (level < 0 || total_commit_[i] < level))
                    level = total_commit_[i];
            if (level < 0) break;
            Chips amount = 0;
            for (int i = 0; i < n_; ++i) amount += clamp(total_commit_[i], prev, level);
            // Dead money above the top live level (shouldn't happen, but never lose chips).
            bool top = true;
            for (int i = 0; i < n_; ++i)
                if (!folded_[i] && total_commit_[i] > level) top = false;
            if (top)
                for (int i = 0; i < n_; ++i)
                    if (total_commit_[i] > level) amount += total_commit_[i] - level;
            distribute(amount, level, strength);
            prev = level;
        }
    }

    static Chips clamp(Chips c, Chips lo, Chips hi) {
        if (c <= lo) return 0;
        return (c < hi ? c : hi) - lo;
    }

    void distribute(Chips amount, Chips level, const uint32_t* strength) {
        uint32_t best = 0;
        for (int i = 0; i < n_; ++i)
            if (!folded_[i] && total_commit_[i] >= level && strength[i] > best) best = strength[i];
        int winners[kMaxPlayers];
        int w = 0;
        for (int k = 1; k <= n_; ++k) {  // seat order left of button, for odd chips
            int i = (button_ + k) % n_;
            if (!folded_[i] && total_commit_[i] >= level && strength[i] == best) winners[w++] = i;
        }
        Chips share = amount / w, odd = amount % w;
        for (int k = 0; k < w; ++k) award(winners[k], share + (k < odd ? 1 : 0));
    }

    void award(int i, Chips amt) {
        stack_[i] += amt;
        won_[i] += amt;
    }

    TableConfig cfg_;
    int n_;
    int button_;
    Deck deck_;

    Chips initial_[kMaxPlayers] = {};
    Chips stack_[kMaxPlayers] = {};
    Chips street_commit_[kMaxPlayers] = {};
    Chips total_commit_[kMaxPlayers] = {};
    Chips won_[kMaxPlayers] = {};
    Chips bet_seen_[kMaxPlayers] = {};
    bool folded_[kMaxPlayers] = {};
    bool has_acted_[kMaxPlayers] = {};

    Street street_ = Street::Preflop;
    int board_size_ = 0;
    int to_act_ = 0;
    Chips current_bet_ = 0;
    Chips last_full_raise_ = 0;
    bool showdown_ = false;
    std::vector<ActionRecord> history_;
};

// Builds a deck with the given hole cards (2 per player, in seat order) and board
// prefix placed where Hand expects them; remaining cards fill the rest in order.
inline Deck make_deck(int num_players, std::span<const Card> holes, std::span<const Card> board_cards) {
    Deck d{};
    bool used[52] = {};
    int pos = 0;
    for (Card c : holes) { d[pos++] = c; used[c] = true; }
    pos = 2 * num_players;
    for (Card c : board_cards) { d[pos++] = c; used[c] = true; }
    // Fill unspecified hole slots, then the rest of the deck.
    int next = 0;
    auto take = [&]() { while (used[next]) ++next; used[next] = true; return Card(next); };
    for (int k = int(holes.size()); k < 2 * num_players; ++k) d[k] = take();
    for (int k = pos; k < 52; ++k) d[k] = take();
    return d;
}

inline const char* to_string(Street s) {
    static const char* names[] = {"preflop", "flop", "turn", "river", "done"};
    return names[int(s)];
}

}  // namespace poker
