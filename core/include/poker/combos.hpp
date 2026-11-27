#pragma once
// The 1326 two-card combos, and the 169 hand classes (AA, AKs, AKo, ...).
//
// Combo index for cards a > b:  a*(a-1)/2 + b   (0..1325)

#include <array>
#include <string>

#include "poker/cards.hpp"

namespace poker {

constexpr int kNumCombos = 1326;
constexpr int kNumClasses = 169;

PK_HD constexpr int combo_index(Card a, Card b) {
    if (a < b) { Card t = a; a = b; b = t; }
    return a * (a - 1) / 2 + b;
}

struct ComboTable {
    std::array<Card, kNumCombos> hi{}, lo{};  // hi > lo as card indices
    constexpr ComboTable() {
        int i = 0;
        for (int a = 1; a < 52; ++a)
            for (int b = 0; b < a; ++b) {
                hi[i] = Card(a);
                lo[i] = Card(b);
                ++i;
            }
    }
};
inline constexpr ComboTable kCombos{};

inline constexpr HandMask combo_mask(int c) { return card_bit(kCombos.hi[c]) | card_bit(kCombos.lo[c]); }

// 169-class grid id = row*13 + col, with row/col 0 = Ace.
// Pairs on the diagonal, suited above it (row < col), offsuit below.
inline constexpr int class_id(int hi_rank, int lo_rank, bool suited) {
    int r = 12 - hi_rank, c = 12 - lo_rank;
    if (hi_rank == lo_rank) return r * 13 + r;
    return suited ? r * 13 + c : c * 13 + r;
}

inline constexpr int combo_class(int combo) {
    Card a = kCombos.hi[combo], b = kCombos.lo[combo];
    int ra = rank_of(a), rb = rank_of(b);
    int hi = ra > rb ? ra : rb, lo = ra > rb ? rb : ra;
    return class_id(hi, lo, suit_of(a) == suit_of(b));
}

inline std::string class_name(int cls) {
    int r = cls / 13, c = cls % 13;
    if (r == c) return {kRankChars[12 - r], kRankChars[12 - r]};
    if (r < c) return {kRankChars[12 - r], kRankChars[12 - c], 's'};
    return {kRankChars[12 - c], kRankChars[12 - r], 'o'};
}

inline std::string combo_name(int c) { return to_string(kCombos.hi[c]) + to_string(kCombos.lo[c]); }

}  // namespace poker
