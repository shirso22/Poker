#pragma once
// Board-relative hand classification: "top pair good kicker + backdoor flush draw",
// not raw evaluator categories. The strategy layer's rule tables consume these,
// and range summaries aggregate them ("his range is 30% top pair+").
//
// Category boundaries are first-pass choices meant to be reviewed by a human player.

#include <array>

#include "poker/combos.hpp"
#include "poker/eval.hpp"
#include "poker/range.hpp"

namespace poker {

enum class Made : uint8_t {
    Nothing = 0,   // no pair, no hole-card contribution beyond high cards
    BoardPlays,    // river: our hole cards don't change the board's 5-card hand
    WeakPair,      // pairs 3rd+ board card
    Underpair,     // pocket pair below the top board card
    SecondPair,    // pairs 2nd highest board rank
    TopPair,
    Overpair,
    TwoPair,       // both hole cards pair the board
    Trips,         // one hole card + a board pair
    Set,           // pocket pair + one board card
    Straight,
    Flush,
    FullHouse,
    Quads,
    StraightFlush,
    Count
};

inline const char* to_string(Made m) {
    static const char* n[] = {"nothing", "board plays", "weak pair", "underpair", "second pair",
                              "top pair", "overpair",    "two pair",  "trips",     "set",
                              "straight", "flush",       "full house", "quads",    "straight flush"};
    return n[int(m)];
}

struct HandClass {
    Made made = Made::Nothing;
    int kicker = -1;          // pairs made with one hole card: rank of the other hole card
    bool flush_draw = false;  // 4 to a flush using >= 1 hole card (flop/turn)
    bool nut_flush_draw = false;
    bool backdoor_fd = false; // flop only: 3 to a flush using >= 1 hole card
    int straight_outs = 0;    // number of ranks that complete a straight using a hole card
    bool overcards = false;   // no pair, both hole cards above the board
    bool oesd() const { return straight_outs >= 2; }  // includes double gutters
    bool gutshot() const { return straight_outs == 1; }
    bool any_draw() const { return flush_draw || straight_outs > 0; }
};

inline HandClass classify(Card h1, Card h2, HandMask board) {
    using namespace detail;
    HandClass hc;
    const HandMask hole = card_bit(h1) | card_bit(h2);
    const int nb = __builtin_popcountll(board);
    const uint32_t full = evaluate(board | hole);
    const Category cat = category_of(full);

    auto suit_ranks = [](HandMask m, int s) { return uint32_t(m >> (16 * s)) & 0x1FFF; };
    uint32_t br = 0;  // board rank set
    std::array<int, 13> bcount{};
    for (int s = 0; s < 4; ++s) {
        uint32_t x = suit_ranks(board, s);
        br |= x;
        for (int r = 0; r < 13; ++r) bcount[r] += (x >> r) & 1;
    }
    const int top = top_index(br);
    int r1 = rank_of(h1), r2 = rank_of(h2);
    if (r1 < r2) std::swap(r1, r2);
    const bool pocket = r1 == r2;

    // ---- made hand ----
    if (nb == 5 && evaluate(board) == full) {
        hc.made = Made::BoardPlays;
    } else if (cat >= kStraight) {
        static constexpr Made map[] = {Made::Straight, Made::Flush, Made::FullHouse, Made::Quads,
                                       Made::StraightFlush};
        hc.made = map[cat - kStraight];
    } else if (pocket && bcount[r1] >= 1) {
        hc.made = Made::Set;
    } else if (!pocket && ((bcount[r1] >= 2) || (bcount[r2] >= 2))) {
        hc.made = Made::Trips;
        hc.kicker = bcount[r1] >= 2 ? r2 : r1;
    } else if (!pocket && bcount[r1] && bcount[r2]) {
        hc.made = Made::TwoPair;
    } else if (pocket) {
        hc.made = r1 > top ? Made::Overpair : Made::Underpair;
    } else if (bcount[r1] || bcount[r2]) {
        int p = bcount[r1] ? r1 : r2;
        hc.kicker = p == r1 ? r2 : r1;
        int pos = __builtin_popcount(br >> (p + 1));  // distinct board ranks above p
        hc.made = pos == 0 ? Made::TopPair : pos == 1 ? Made::SecondPair : Made::WeakPair;
    } else {
        hc.made = Made::Nothing;
        hc.overcards = r2 > top;
    }

    // ---- draws (flop/turn only) ----
    if (nb < 5) {
        if (cat < kFlush) {
            for (int s = 0; s < 4; ++s) {
                int hs = __builtin_popcount(suit_ranks(hole, s));
                int bs = __builtin_popcount(suit_ranks(board, s));
                if (hs == 0) continue;
                if (hs + bs == 4) {
                    hc.flush_draw = true;
                    uint32_t missing = 0x1FFF & ~suit_ranks(board, s);
                    hc.nut_flush_draw = (suit_ranks(hole, s) & top_bit(missing)) != 0;
                }
                if (nb == 3 && hs + bs == 3) hc.backdoor_fd = true;
            }
        }
        if (cat < kStraight) {
            uint32_t all = br | (1u << rank_of(h1)) | (1u << rank_of(h2));
            for (int r = 0; r < 13; ++r) {
                if (straight_high(all | (1u << r)) >= 0 && straight_high(br | (1u << r)) < 0) ++hc.straight_outs;
            }
        }
    }
    return hc;
}

// Weighted composition of a range on a board.
struct RangeSummary {
    double total = 0;
    std::array<double, int(Made::Count)> made{};
    double flush_draw = 0, oesd = 0, gutshot = 0, combo_draw = 0, backdoor_fd = 0;

    double frac(Made m) const { return total > 0 ? made[int(m)] / total : 0; }
    double frac_at_least(Made m) const {
        double s = 0;
        for (int i = int(m); i < int(Made::Count); ++i) s += made[i];
        return total > 0 ? s / total : 0;
    }
};

inline RangeSummary summarize(const Range& r, HandMask board) {
    RangeSummary s;
    for (int c = 0; c < kNumCombos; ++c) {
        if (r.w[c] <= 0 || (combo_mask(c) & board)) continue;
        double w = r.w[c];
        HandClass hc = classify(kCombos.hi[c], kCombos.lo[c], board);
        s.total += w;
        s.made[int(hc.made)] += w;
        if (hc.flush_draw) s.flush_draw += w;
        if (hc.oesd()) s.oesd += w;
        if (hc.gutshot()) s.gutshot += w;
        if (hc.flush_draw && hc.straight_outs > 0) s.combo_draw += w;
        if (hc.backdoor_fd) s.backdoor_fd += w;
    }
    return s;
}

}  // namespace poker
