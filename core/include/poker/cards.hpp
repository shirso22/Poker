#pragma once
// Card representation shared by host and device code.
//
// A card is an index 0..51:  card = rank * 4 + suit
//   rank: 0 = '2' ... 12 = 'A'
//   suit: 0 = c, 1 = d, 2 = h, 3 = s
//
// A hand (any number of cards) is a 64-bit mask with one 16-bit lane per suit:
//   bit (suit * 16 + rank)
// so the 13-bit rank set of suit s is (mask >> (16 * s)) & 0x1FFF.

#include <cstdint>
#include <string>
#include <string_view>

#ifdef __CUDACC__
#define PK_HD __host__ __device__ __forceinline__
#else
#define PK_HD inline
#endif

namespace poker {

using Card = uint8_t;
using HandMask = uint64_t;

constexpr int kNumCards = 52;
constexpr int kNumRanks = 13;
constexpr int kNumSuits = 4;

PK_HD constexpr int rank_of(Card c) { return c >> 2; }
PK_HD constexpr int suit_of(Card c) { return c & 3; }
PK_HD constexpr Card make_card(int rank, int suit) { return Card(rank * 4 + suit); }

PK_HD constexpr HandMask card_bit(Card c) {
    return HandMask(1) << (suit_of(c) * 16 + rank_of(c));
}

// ---- string helpers (host only) ----

inline constexpr std::string_view kRankChars = "23456789TJQKA";
inline constexpr std::string_view kSuitChars = "cdhs";

inline std::string to_string(Card c) {
    return {kRankChars[rank_of(c)], kSuitChars[suit_of(c)]};
}

// Parses "Ah", "Td", ... Returns 255 on bad input.
inline Card parse_card(std::string_view s) {
    if (s.size() != 2) return 255;
    auto r = kRankChars.find(s[0]);
    auto u = kSuitChars.find(s[1]);
    if (r == std::string_view::npos || u == std::string_view::npos) return 255;
    return make_card(int(r), int(u));
}

// Parses "AhKd7c..." (no separators) into a mask. Returns 0 on bad input or duplicates.
inline HandMask parse_hand(std::string_view s) {
    if (s.size() % 2) return 0;
    HandMask m = 0;
    for (size_t i = 0; i < s.size(); i += 2) {
        Card c = parse_card(s.substr(i, 2));
        if (c == 255 || (m & card_bit(c))) return 0;
        m |= card_bit(c);
    }
    return m;
}

}  // namespace poker
