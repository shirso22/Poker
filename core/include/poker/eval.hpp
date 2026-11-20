#pragma once
// Table-free hand evaluator for 5..7 cards. Host + device.
//
// Returns a uint32 strength; higher is better, equal means a tie.
// Layout:  [category:4][major:13][minor:13]
//   major = bitmask of the ranks that define the hand (quad/trip/pair ranks,
//           straight high card), minor = bitmask of kickers.
// Comparing two equal-popcount rank masks numerically is the same as comparing
// their descending rank lists lexicographically, so plain integer comparison
// gives correct poker ordering with no lookup tables. (Tables need memory
// bandwidth, which matters on a GPU; this needs a handful of ALU ops instead.)

#include "poker/cards.hpp"

namespace poker {

enum Category : uint32_t {
    kHighCard = 0,
    kPair,
    kTwoPair,
    kTrips,
    kStraight,
    kFlush,
    kFullHouse,
    kQuads,
    kStraightFlush,
};

PK_HD constexpr Category category_of(uint32_t strength) { return Category(strength >> 26); }

namespace detail {

PK_HD int popcount(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}

// Index of highest set bit (x != 0).
PK_HD int top_index(uint32_t x) {
#ifdef __CUDA_ARCH__
    return 31 - __clz(x);
#else
    return 31 - __builtin_clz(x);
#endif
}

PK_HD uint32_t top_bit(uint32_t x) { return x ? (1u << top_index(x)) : 0u; }

// Keep only the n highest set bits.
PK_HD uint32_t keep_top(uint32_t x, int n) {
    while (popcount(x) > n) x &= x - 1;  // clear lowest set bit
    return x;
}

// Rank index of the straight's high card, or -1. Handles the wheel (A-5).
PK_HD int straight_high(uint32_t ranks) {
    // ext bit 0 = ace-low, bit k+1 = rank k
    uint32_t ext = (ranks << 1) | ((ranks >> 12) & 1u);
    uint32_t s = ext & (ext >> 1) & (ext >> 2) & (ext >> 3) & (ext >> 4);
    // bit j of s: ext bits j..j+4 all set -> high card is ext bit j+4 = rank j+3
    return s ? top_index(s) + 3 : -1;
}

PK_HD constexpr uint32_t pack(Category c, uint32_t major, uint32_t minor) {
    return (uint32_t(c) << 26) | (major << 13) | minor;
}

}  // namespace detail

PK_HD uint32_t evaluate(HandMask h) {
    using namespace detail;
    const uint32_t s0 = uint32_t(h) & 0x1FFF;
    const uint32_t s1 = uint32_t(h >> 16) & 0x1FFF;
    const uint32_t s2 = uint32_t(h >> 32) & 0x1FFF;
    const uint32_t s3 = uint32_t(h >> 48) & 0x1FFF;

    const uint32_t ranks = s0 | s1 | s2 | s3;

    // Flush suit. With <= 9 cards at most one suit can have 5+.
    uint32_t fl = 0;
    if (popcount(s0) >= 5) fl = s0;
    else if (popcount(s1) >= 5) fl = s1;
    else if (popcount(s2) >= 5) fl = s2;
    else if (popcount(s3) >= 5) fl = s3;

    if (fl) {
        int sh = straight_high(fl);
        if (sh >= 0) return pack(kStraightFlush, 1u << sh, 0);
    }

    // Multiplicity masks.
    const uint32_t quads = s0 & s1 & s2 & s3;
    const uint32_t three_plus = (s0 & s1 & s2) | (s0 & s1 & s3) | (s0 & s2 & s3) | (s1 & s2 & s3);
    const uint32_t two_plus = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);
    const uint32_t trips = three_plus & ~quads;
    const uint32_t pairs = two_plus & ~three_plus;

    if (quads) {
        uint32_t q = top_bit(quads);
        return pack(kQuads, q, top_bit(ranks & ~q));
    }

    if (trips) {
        uint32_t t = top_bit(trips);
        uint32_t rest = (trips & ~t) | pairs;  // second trips counts as the pair
        if (rest) return pack(kFullHouse, t, top_bit(rest));
    }

    if (fl) return pack(kFlush, keep_top(fl, 5), 0);

    {
        int sh = straight_high(ranks);
        if (sh >= 0) return pack(kStraight, 1u << sh, 0);
    }

    if (trips) {
        uint32_t t = top_bit(trips);
        return pack(kTrips, t, keep_top(ranks & ~t, 2));
    }

    if (popcount(pairs) >= 2) {
        uint32_t p = keep_top(pairs, 2);
        return pack(kTwoPair, p, top_bit(ranks & ~p));
    }

    if (pairs) return pack(kPair, pairs, keep_top(ranks & ~pairs, 3));

    return pack(kHighCard, keep_top(ranks, 5), 0);
}

}  // namespace poker
