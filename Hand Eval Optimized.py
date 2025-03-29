class Card:

    def __init__(self, rank, suit):
        self.rank = rank  # Numerical value such as 2,3,4...,King,Ace
        self.suit = suit  # Suit such as Spade(s), Heart(h), Club(c), Diamond(d)


from collections import Counter

def Hand_Score(hand):
    hand = sorted(hand, key=lambda card: card.rank) #Sort the hand in ascending order of rank
    kicker = sum(card.rank * 10**i for i, card in enumerate(hand)) / 1e6 #Floating point tiebreaker
    
    ranks = [card.rank for card in hand] #Array storing just the ranks in order
    rank_counts = Counter(ranks) #Dictionary with each rank and number of times it appears in the hand
    unique_ranks = sorted(rank_counts.keys()) #Array with just the unique ranks in the hand
    is_flush = len({card.suit for card in hand}) == 1
    
    # Straight (including wheel)
    is_straight = (ranks[-1] - ranks[0] == 4) or (ranks == [2,3,4,5,14])
    
    if is_flush and is_straight:
        return 9 + kicker  # Straight flush
    elif 4 in rank_counts.values():
        return 8 + kicker  # Quads
    elif sorted(rank_counts.values()) == [2,3]:
        return 7 + kicker  # Full house
    elif is_flush:
        return 6 + kicker  # Flush
    elif is_straight:
        return 5 + kicker  # Straight
    elif 3 in rank_counts.values():
        return 4 + kicker  # Trips
    elif list(rank_counts.values()).count(2) == 2:
        return 3 + kicker  # Two pair
    elif 2 in rank_counts.values():
        return 2 + kicker  # One pair
    else:
        return 1 + kicker  # High card


# Creating objects of the Card class
card11,card12,card13,card14,card15 = Card(14,"d"),Card(14,"s"),Card(5,"c"),Card(7,"d"),Card(8,"d")
card21,card22,card23,card24,card25 = Card(10,"s"),Card(10,"d"),Card(10,"h"),Card(6,"s"),Card(6,"h")

#Testing
hand1=[card11,card12,card13,card14,card15]
hand2=[card21,card22,card23,card24,card25]

print(Hand_Score(hand1))
print(Hand_Score(hand2))
