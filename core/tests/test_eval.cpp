// Evaluator tests:
//  1. Exhaustive 5-card and 7-card enumeration against the known category
//     counts and number of distinct equivalence classes.
//  2. Randomized pairwise ordering checks against an independent, deliberately
//     naive reference evaluator (sort-based, best-of-21 subsets).
//  3. Hand-picked edge cases, including the tiebreak bug in legacy/hand_eval_v0.py.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

#include "poker/eval.hpp"

using namespace poker;

static int g_failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++g_failures;                                             \
            std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

static const char* kCatNames[] = {"high card", "pair",  "two pair",   "trips",         "straight",
                                  "flush",     "full house", "quads", "straight flush"};

// ---------------------------------------------------------------------------
// Reference evaluator: obviously-correct and slow.
// Returns {category, tiebreak ranks...} compared lexicographically.
// ---------------------------------------------------------------------------
using RefKey = std::array<int, 6>;

static RefKey ref_eval5(const Card* c) {
    std::array<int, 13> cnt{};
    bool flush = true;
    for (int i = 0; i < 5; ++i) {
        cnt[rank_of(c[i])]++;
        if (suit_of(c[i]) != suit_of(c[0])) flush = false;
    }
    // groups sorted by (count desc, rank desc)
    std::vector<std::pair<int, int>> g;
    for (int r = 12; r >= 0; --r)
        if (cnt[r]) g.push_back({cnt[r], r});
    std::stable_sort(g.begin(), g.end(), [](auto a, auto b) { return a.first > b.first; });

    int straight_hi = -1;
    if (g.size() == 5) {
        if (g[0].second - g[4].second == 4) straight_hi = g[0].second;
        else if (g[0].second == 12 && g[1].second == 3) straight_hi = 3;  // wheel
    }

    RefKey k{};
    auto fill = [&](int cat) {
        k[0] = cat;
        for (size_t i = 0; i < g.size(); ++i) k[1 + i] = g[i].second;
    };
    if (flush && straight_hi >= 0) { k = {kStraightFlush, straight_hi}; }
    else if (g[0].first == 4) fill(kQuads);
    else if (g[0].first == 3 && g[1].first == 2) fill(kFullHouse);
    else if (flush) fill(kFlush);
    else if (straight_hi >= 0) { k = {kStraight, straight_hi}; }
    else if (g[0].first == 3) fill(kTrips);
    else if (g[0].first == 2 && g[1].first == 2) fill(kTwoPair);
    else if (g[0].first == 2) fill(kPair);
    else fill(kHighCard);
    return k;
}

static RefKey ref_eval(const std::vector<Card>& cards) {
    RefKey best{-1};
    int n = int(cards.size());
    std::array<Card, 5> sub;
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            for (int c = b + 1; c < n; ++c)
                for (int d = c + 1; d < n; ++d)
                    for (int e = d + 1; e < n; ++e) {
                        sub = {cards[a], cards[b], cards[c], cards[d], cards[e]};
                        best = std::max(best, ref_eval5(sub.data()));
                    }
    return best;
}

static HandMask to_mask(const std::vector<Card>& cards) {
    HandMask m = 0;
    for (Card c : cards) m |= card_bit(c);
    return m;
}

// ---------------------------------------------------------------------------

static void test_exhaustive_5() {
    std::array<uint64_t, 9> counts{};
    std::unordered_set<uint32_t> distinct;
    for (int a = 0; a < 52; ++a)
        for (int b = a + 1; b < 52; ++b)
            for (int c = b + 1; c < 52; ++c)
                for (int d = c + 1; d < 52; ++d)
                    for (int e = d + 1; e < 52; ++e) {
                        HandMask m = card_bit(a) | card_bit(b) | card_bit(c) | card_bit(d) | card_bit(e);
                        uint32_t v = evaluate(m);
                        counts[category_of(v)]++;
                        distinct.insert(v);
                    }
    const std::array<uint64_t, 9> expected = {1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40};
    for (int i = 0; i < 9; ++i)
        CHECK(counts[i] == expected[i], "5-card %s: got %llu want %llu", kCatNames[i],
              (unsigned long long)counts[i], (unsigned long long)expected[i]);
    CHECK(distinct.size() == 7462, "5-card distinct classes: got %zu want 7462", distinct.size());
    std::printf("[5-card exhaustive] 2,598,960 hands, %zu classes\n", distinct.size());
}

static void test_exhaustive_7() {
    std::array<uint64_t, 9> counts{};
    std::unordered_set<uint32_t> distinct;
    uint64_t checksum = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int a = 0; a < 52; ++a) {
        HandMask ma = card_bit(a);
        for (int b = a + 1; b < 52; ++b) {
            HandMask mb = ma | card_bit(b);
            for (int c = b + 1; c < 52; ++c) {
                HandMask mc = mb | card_bit(c);
                for (int d = c + 1; d < 52; ++d) {
                    HandMask md = mc | card_bit(d);
                    for (int e = d + 1; e < 52; ++e) {
                        HandMask me = md | card_bit(e);
                        for (int f = e + 1; f < 52; ++f) {
                            HandMask mf = me | card_bit(f);
                            for (int g = f + 1; g < 52; ++g) {
                                uint32_t v = evaluate(mf | card_bit(g));
                                counts[category_of(v)]++;
                                checksum += v;
                                if (distinct.size() < 5000) distinct.insert(v);
                            }
                        }
                    }
                }
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::array<uint64_t, 9> expected = {23294460, 58627800, 31433400, 6461620, 6180020,
                                              4047644,  3473184,  224848,   41584};
    for (int i = 0; i < 9; ++i)
        CHECK(counts[i] == expected[i], "7-card %s: got %llu want %llu", kCatNames[i],
              (unsigned long long)counts[i], (unsigned long long)expected[i]);
    CHECK(distinct.size() == 4824, "7-card distinct classes: got %zu want 4824", distinct.size());
    std::printf("[7-card exhaustive] 133,784,560 hands in %.2fs (%.1f M hands/s, single thread, "
                "incl. hash-set overhead), %zu classes, checksum %llu\n",
                secs, 133784560 / secs / 1e6, distinct.size(), (unsigned long long)checksum);
}

static void test_random_vs_reference(int n_pairs) {
    std::mt19937_64 rng(12345);
    std::vector<Card> deck(52);
    for (int i = 0; i < 52; ++i) deck[i] = Card(i);
    int mismatches = 0;
    for (int it = 0; it < n_pairs; ++it) {
        int n = 5 + int(rng() % 3);  // 5, 6 or 7 cards
        std::shuffle(deck.begin(), deck.end(), rng);
        std::vector<Card> a(deck.begin(), deck.begin() + n);
        // Share a board half the time so ties and near-ties are common.
        std::vector<Card> b;
        if (it % 2 && n >= 5) {
            b.assign(deck.begin() + 2, deck.begin() + n);  // same n-2 "board" cards
            b.push_back(deck[n]);
            b.push_back(deck[n + 1]);
        } else {
            b.assign(deck.begin() + n, deck.begin() + 2 * n);
        }
        uint32_t ea = evaluate(to_mask(a)), eb = evaluate(to_mask(b));
        RefKey ra = ref_eval(a), rb = ref_eval(b);
        int fast = (ea > eb) - (ea < eb);
        int ref = (ra > rb) - (ra < rb);
        if (fast != ref || int(category_of(ea)) != ra[0] || int(category_of(eb)) != rb[0]) {
            if (++mismatches <= 5) {
                std::string sa, sb;
                for (Card c : a) sa += to_string(c);
                for (Card c : b) sb += to_string(c);
                std::printf("  mismatch: %s vs %s fast=%d ref=%d\n", sa.c_str(), sb.c_str(), fast, ref);
            }
        }
    }
    CHECK(mismatches == 0, "%d / %d random comparisons disagree with reference", mismatches, n_pairs);
    std::printf("[random vs reference] %d pairs, %d mismatches\n", n_pairs, mismatches);
}

static uint32_t ev(const char* s) {
    HandMask m = parse_hand(s);
    if (!m) std::printf("bad hand string %s\n", s);
    return evaluate(m);
}

static void test_edge_cases() {
    // The legacy v0 evaluator ranked 22AKQ above 33456 (float kicker overflow).
    CHECK(ev("3c3d4h5s6c") > ev("2c2dAhKsQc"), "33456 must beat 22AKQ");
    CHECK(ev("AcKdQhJsTc") > ev("KcQdJhTs9c"), "broadway > K-high straight");
    CHECK(ev("6c5d4h3s2c") > ev("Ac2d3h4s5c"), "6-high straight > wheel");
    CHECK(category_of(ev("Ac2d3h4s5c")) == kStraight, "wheel is a straight");
    CHECK(category_of(ev("Ac2c3c4c5c")) == kStraightFlush, "steel wheel is a straight flush");
    CHECK(category_of(ev("QcKcAc2c3c4d5h")) == kFlush, "no wraparound straight flush (QKA23)");
    CHECK(category_of(ev("KcAc2c3d4h")) == kHighCard, "KA234 is not a straight");
    // Two trips -> full house using the second trips as the pair.
    CHECK(ev("AcAdAh2c2d2hKs") == ev("AcAdAh2c2dKsQs"), "AAA222K == AAA22KQ (both AAA22)");
    // Three pairs -> best two pair, kicker may come from the third pair.
    CHECK(ev("AcAdKcKdQcQd2s") == ev("AhAsKhKsQh3c4d"), "AAKKQQ2 == AAKKQ34");
    CHECK(ev("AcAdKcKd5c5dQs") > ev("AhAsKhKs5h5s4d"), "AAKK55Q > AAKK554");
    // Quads with a pair: kicker is the best remaining card.
    CHECK(ev("9c9d9h9sAcAd2c") > ev("9c9d9h9sKcKdQc"), "9999A beats 9999K");
    // Flush beats straight when both are present.
    CHECK(category_of(ev("2h5h9hJhKh8cTc")) == kFlush, "flush over straight");
    // Flush kickers: all five cards matter.
    CHECK(ev("AhKh9h7h3h") > ev("AhKh9h7h2h"), "flush 5th kicker");
    // Pair kickers.
    CHECK(ev("AcAdKhQs3c") > ev("AcAdKhJsTc"), "AAKQ3 > AAKJT");
    // Board plays: tie.
    CHECK(ev("AcKdQhJsTc2d3h") == ev("AcKdQhJsTc4d5h"), "board broadway ties");
}

int main() {
    test_edge_cases();
    test_random_vs_reference(200000);
    test_exhaustive_5();
    test_exhaustive_7();
    if (g_failures) {
        std::printf("\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("\nALL TESTS PASSED\n");
    return 0;
}
