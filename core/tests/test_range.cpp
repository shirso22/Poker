// Range engine tests: combos, parsing, card removal, Bayes updates/replay,
// equity (vs brute force, known matchups, symmetry, MC), multiway, classification.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "poker/classify.hpp"
#include "poker/equity.hpp"

using namespace poker;

static int g_failures = 0;
#define CHECK(cond, ...)                                                 \
    do {                                                                 \
        if (!(cond)) {                                                   \
            ++g_failures;                                                \
            std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                    \
            std::printf("\n");                                           \
        }                                                                \
    } while (0)

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static Range R(const char* s) {
    Range r;
    std::string err;
    if (!Range::parse(s, r, &err)) std::printf("parse error: %s\n", err.c_str());
    return r;
}
static int C(const char* s) { return combo_index(parse_card({s, 2}), parse_card({s + 2, 2})); }

// ---------------------------------------------------------------------------

static void test_combos() {
    bool seen[kNumCombos] = {};
    int bad = 0;
    for (int a = 0; a < 52; ++a)
        for (int b = 0; b < 52; ++b) {
            if (a == b) continue;
            int i = combo_index(Card(a), Card(b));
            if (i < 0 || i >= kNumCombos || combo_mask(i) != (card_bit(Card(a)) | card_bit(Card(b)))) ++bad;
            seen[i] = true;
        }
    int covered = 0;
    for (bool s : seen) covered += s;
    CHECK(bad == 0 && covered == kNumCombos, "combo index round trip");

    int per_class[kNumClasses] = {};
    for (int c = 0; c < kNumCombos; ++c) per_class[combo_class(c)]++;
    int pairs = 0, suited = 0, offsuit = 0;
    for (int k = 0; k < kNumClasses; ++k) {
        int r = k / 13, col = k % 13;
        int expect = r == col ? 6 : r < col ? 4 : 12;
        CHECK(per_class[k] == expect, "class %s has %d combos", class_name(k).c_str(), per_class[k]);
        (r == col ? pairs : r < col ? suited : offsuit)++;
    }
    CHECK(pairs == 13 && suited == 78 && offsuit == 78, "169 classes");
    CHECK(class_name(combo_class(C("AhKh"))) == "AKs" && class_name(combo_class(C("Kd7c"))) == "K7o" &&
              class_name(combo_class(C("2c2d"))) == "22",
          "class names");
}

static void test_parse() {
    struct { const char* s; int n; } cases[] = {
        {"AA", 6},       {"AKs", 4},      {"AKo", 12},    {"AK", 16},       {"QQ+", 18},
        {"ATs+", 16},    {"A5s-A2s", 16}, {"77-22", 36},  {"22-77", 36},    {"AhKh", 1},
        {"random", 1326}, {"KTo+", 36},   {"QQ+, AKs, AKo", 34},
    };
    for (auto& t : cases) CHECK(R(t.s).support() == t.n, "'%s' -> %d combos, want %d", t.s, R(t.s).support(), t.n);
    Range r = R("AK:0.5, AKs");
    CHECK(r.w[C("AhKh")] == 1.0f && r.w[C("AhKd")] == 0.5f, "later tokens override, weights");
    Range bad;
    CHECK(!Range::parse("AKx", bad) && !Range::parse("A5s-K2s", bad) && !Range::parse("AA:", bad), "rejects junk");
}

static void test_removal_and_bayes() {
    Range aa = R("AA");
    aa.remove(parse_hand("As7d2c"));
    CHECK(aa.support() == 3, "AA with As on board: 3 combos, got %d", aa.support());

    // Prior: AA and KK equally likely. P(raise | AA) = 0.9, P(raise | KK) = 0.1.
    Range r = R("AA, KK");
    std::array<float, kNumCombos> lik{};
    for (int c = 0; c < kNumCombos; ++c) lik[c] = class_name(combo_class(c)) == "AA" ? 0.9f : 0.1f;
    bayes_update(r, lik.data());
    double aa_w = 0, tot = 0;
    for (int c = 0; c < kNumCombos; ++c) {
        tot += r.w[c];
        if (class_name(combo_class(c)) == "AA") aa_w += r.w[c];
    }
    CHECK(std::fabs(aa_w / tot - 0.9) < 1e-6, "posterior P(AA) = 0.9, got %f", aa_w / tot);
    CHECK(r.support() == 12, "support unchanged");

    // Impossible action under the model: floor keeps the range alive, uniform over support.
    Range z = R("AA, KK");
    std::array<float, kNumCombos> zero{};
    bayes_update(z, zero.data(), 1e-3f);
    CHECK(z.support() == 12 && z.w[C("AhAd")] == z.w[C("KhKd")], "all-zero likelihood -> uniform over support");

    // Floor: a near-impossible combo survives at the floor.
    Range f = R("AA, KK");
    for (int c = 0; c < kNumCombos; ++c) lik[c] = class_name(combo_class(c)) == "AA" ? 1.0f : 0.0f;
    bayes_update(f, lik.data(), 1e-4f);
    CHECK(f.w[C("KhKd")] == 1e-4f && f.w[C("AhAd")] == 1.0f, "floored, rescaled");

    // Replay == incremental updates.
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<Observation> obs(6);
    HandMask dead_seq[6] = {0, parse_hand("Ah7c2d"), 0, parse_hand("Ks"), 0, parse_hand("3h")};
    for (int k = 0; k < 6; ++k) {
        obs[k].dead = dead_seq[k];
        for (auto& x : obs[k].likelihood) x = U(rng);
    }
    Range inc = Range::uniform();
    for (auto& o : obs) {
        inc.remove(o.dead);
        bayes_update(inc, o.likelihood.data());
    }
    Range rep = replay(Range::uniform(), 6, [&](int k) -> const Observation& { return obs[k]; });
    float maxdiff = 0;
    for (int c = 0; c < kNumCombos; ++c) maxdiff = std::max(maxdiff, std::fabs(inc.w[c] - rep.w[c]));
    CHECK(maxdiff == 0.0f, "replay matches incremental, diff %g", maxdiff);
}

// Brute-force equity of one hero combo vs a range over all runouts of `board`.
static double brute_equity(int hero, HandMask board, const Range& v) {
    double num = 0, den = 0;
    HandMask hm = combo_mask(hero);
    if (hm & board) return std::nan("");
    std::vector<HandMask> runouts;
    int need = 5 - __builtin_popcountll(board);
    if (need == 0) runouts.push_back(board);
    else if (need == 1) {
        for (int t = 0; t < 52; ++t)
            if (!(board & card_bit(Card(t)))) runouts.push_back(board | card_bit(Card(t)));
    } else {
        for (int t = 0; t < 52; ++t)
            for (int u = t + 1; u < 52; ++u)
                if (!(board & (card_bit(Card(t)) | card_bit(Card(u)))))
                    runouts.push_back(board | card_bit(Card(t)) | card_bit(Card(u)));
    }
    for (HandMask b : runouts) {
        if (b & hm) continue;
        uint32_t hs = evaluate(b | hm);
        for (int c = 0; c < kNumCombos; ++c) {
            HandMask vm = combo_mask(c);
            if (v.w[c] <= 0 || (vm & (b | hm))) continue;
            uint32_t vs = evaluate(b | vm);
            num += v.w[c] * (hs > vs ? 1.0 : hs == vs ? 0.5 : 0.0);
            den += v.w[c];
        }
    }
    return num / den;
}

static void test_river_vs_brute_force() {
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    double maxerr = 0;
    for (int trial = 0; trial < 20; ++trial) {
        std::array<Card, 52> d;
        for (int i = 0; i < 52; ++i) d[i] = Card(i);
        std::shuffle(d.begin(), d.end(), rng);
        HandMask board = 0;
        for (int k = 0; k < 5; ++k) board |= card_bit(d[k]);
        Range v;
        for (int c = 0; c < kNumCombos; ++c) v.w[c] = U(rng) < 0.3f ? U(rng) : 0.0f;  // sparse, uneven
        EquityResult e = equity_vs_range(board, v);
        for (int c = 0; c < kNumCombos; ++c) {
            if (combo_mask(c) & board) continue;
            double b = brute_equity(c, board, v);
            maxerr = std::max(maxerr, std::fabs(e.equity(c) - b));
        }
    }
    CHECK(maxerr < 1e-9, "river sweep vs brute force: max error %g", maxerr);
    std::printf("[river] sweep matches brute force on 20 boards x 1081 combos (max err %.1e)\n", maxerr);
}

static void test_turn_flop_vs_brute_force() {
    Range v = R("QQ+, AKs, AQs, KQs, JTs, 98s, AKo:0.6, 77-55:0.5");
    const char* hero[] = {"AhKd", "JcTc", "8h7h", "5s5d", "QdJd"};
    double maxerr = 0;
    for (const char* bs : {"Kc9h4c", "Tc9h4c2s"}) {
        HandMask board = parse_hand(bs);
        auto t0 = std::chrono::steady_clock::now();
        EquityResult e = equity_vs_range(board, v);
        double dt = secs_since(t0);
        for (const char* h : hero) {
            int c = C(h);
            double b = brute_equity(c, board, v);
            maxerr = std::max(maxerr, std::fabs(e.equity(c) - b));
        }
        std::printf("[%s] exact all-combo equity on %s: %.1f ms (CPU, 1 thread)\n",
                    __builtin_popcountll(board) == 3 ? "flop" : "turn", bs, dt * 1e3);
    }
    CHECK(maxerr < 1e-9, "turn/flop enumeration vs brute force: max error %g", maxerr);
}

static void test_symmetry() {
    Range a = R("TT+, AQs+, AKo"), b = R("88-55, KQs, QJs, JTs, T9s, AJo");
    for (const char* bs : {"9h8h2c3d4s", "Kh8h2c5d"}) {
        HandMask board = parse_hand(bs);
        double ab = range_equity(a, equity_vs_range(board, b));
        double ba = range_equity(b, equity_vs_range(board, a));
        CHECK(std::fabs(ab + ba - 1.0) < 1e-9, "eq(A vs B) + eq(B vs A) = 1 on %s: %f + %f", bs, ab, ba);
    }
}

static double exact_preflop(const char* hero, const char* vil) {
    HandMask h = parse_hand(hero), v = parse_hand(vil);
    Card rest[48];
    int n = 0;
    for (int c = 0; c < 52; ++c)
        if (!((h | v) & card_bit(Card(c)))) rest[n++] = Card(c);
    double s = 0;
    long cnt = 0;
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            for (int c = b + 1; c < n; ++c)
                for (int d = c + 1; d < n; ++d)
                    for (int e = d + 1; e < n; ++e) {
                        HandMask bd = card_bit(rest[a]) | card_bit(rest[b]) | card_bit(rest[c]) |
                                      card_bit(rest[d]) | card_bit(rest[e]);
                        uint32_t x = evaluate(bd | h), y = evaluate(bd | v);
                        s += x > y ? 1 : x == y ? 0.5 : 0;
                        ++cnt;
                    }
    return s / cnt;
}

static void test_known_matchups_and_mc() {
    double aakk = exact_preflop("AsAh", "KdKc");
    double aks_qq = exact_preflop("AsKs", "QhQd");
    std::printf("[preflop exact] AsAh vs KdKc = %.4f, AsKs vs QhQd = %.4f\n", aakk, aks_qq);
    CHECK(aakk > 0.81 && aakk < 0.83, "AA vs KK ~82%%");
    CHECK(aks_qq > 0.45 && aks_qq < 0.47, "AKs vs QQ ~46%%");

    // Monte Carlo preflop, all hero combos at once vs a single-combo range.
    std::mt19937_64 rng(3);
    Range kk;
    kk.w[C("KdKc")] = 1;
    auto t0 = std::chrono::steady_clock::now();
    EquityResult e = equity_vs_range_mc(0, kk, 20000, rng);
    double dt = secs_since(t0);
    double mc = e.equity(C("AsAh"));
    CHECK(std::fabs(mc - aakk) < 0.01, "MC preflop AA vs KK %f vs exact %f", mc, aakk);
    std::printf("[preflop MC] 20k boards, all 1326 hero combos: %.0f ms; AA vs KK %.4f\n", dt * 1e3, mc);
}

static void test_multiway() {
    // 3-way on a flop with single-combo ranges vs exact enumeration of runouts.
    HandMask board = parse_hand("Th9h2c");
    int hero = C("AhKh");  // nut flush draw + overs
    HandMask v1 = parse_hand("TdTs"), v2 = parse_hand("QcJc");
    Range r1, r2;
    r1.w[C("TdTs")] = 1;
    r2.w[C("QcJc")] = 1;
    double s = 0;
    int cnt = 0;
    HandMask used = board | combo_mask(hero) | v1 | v2;
    for (int t = 0; t < 52; ++t)
        for (int u = t + 1; u < 52; ++u) {
            HandMask tu = card_bit(Card(t)) | card_bit(Card(u));
            if (used & tu) continue;
            HandMask b = board | tu;
            uint32_t h = evaluate(b | combo_mask(hero)), a = evaluate(b | v1), c = evaluate(b | v2);
            uint32_t best = std::max({h, a, c});
            int w = (h == best) + (a == best) + (c == best);
            s += h == best ? 1.0 / w : 0;
            ++cnt;
        }
    double exact = s / cnt;
    std::mt19937_64 rng(11);
    std::array<Range, 2> vs = {r1, r2};
    auto res = multiway_equity(hero, board, std::span<const Range>(vs), 200000, rng);
    CHECK(std::fabs(res.equity - exact) < 0.005, "multiway MC %f vs exact %f", res.equity, exact);

    // With real ranges: rejection sampling must stay efficient.
    std::array<Range, 2> wide = {R("22+, A2s+, KTs+, QTs+, JTs, ATo+, KQo"), R("random")};
    auto res2 = multiway_equity(hero, board, std::span<const Range>(wide), 50000, rng);
    std::printf("[multiway] AhKh on Th9h2c vs TT & QJs: MC %.4f, exact %.4f | vs two ranges: %.4f "
                "(rejected %.1f%%)\n",
                res.equity, exact, res2.equity, 100.0 * res2.rejected / (res2.samples + res2.rejected));
}

static void test_classify() {
    struct Case { const char* hole; const char* board; Made made; };
    Case cases[] = {
        {"AhKd", "Ks7c2d", Made::TopPair},        {"QhQd", "Js7c2d", Made::Overpair},
        {"9s9h", "Kc7d2h", Made::Underpair},      {"7s7h", "Kc7d2h", Made::Set},
        {"Kh7s", "Kc7d2h", Made::TwoPair},        {"7s6s", "7c7dKh", Made::Trips},
        {"Ah7s", "Kc7d2h", Made::SecondPair},     {"Ah2s", "Kc7d2h", Made::WeakPair},
        {"JhTs", "AcKdQh", Made::Straight},       {"5h4h", "AsKsQsJsTs", Made::BoardPlays},
        {"AhQd", "AsKsQsJsTs", Made::BoardPlays}, {"9s8d", "AsKsQsJsTs", Made::BoardPlays},
        {"2s3d", "AcKsQhJdTc", Made::BoardPlays}, {"AhAd", "AcAs7d", Made::Quads},
        {"Ah5h", "Kh7h2c", Made::Nothing},        {"KsKh", "Kd7c7h", Made::FullHouse},
    };
    for (auto& t : cases) {
        HandClass hc = classify(parse_card({t.hole, 2}), parse_card({t.hole + 2, 2}), parse_hand(t.board));
        CHECK(hc.made == t.made, "%s on %s: got '%s' want '%s'", t.hole, t.board, to_string(hc.made),
              to_string(t.made));
    }
    auto cls = [](const char* hole, const char* board) {
        return classify(parse_card({hole, 2}), parse_card({hole + 2, 2}), parse_hand(board));
    };
    HandClass x = cls("Ah5h", "Kh7h2c");
    CHECK(x.flush_draw && x.nut_flush_draw && !x.overcards, "A5hh: nut flush draw");
    CHECK(!cls("Qh5h", "Kh7h2c").nut_flush_draw && cls("Qh5h", "Kh7h2c").flush_draw, "Q-high FD not nut");
    CHECK(cls("9c8c", "7d6s2h").oesd(), "98 on 762: OESD");
    CHECK(cls("9c8c", "7d5s2h").gutshot(), "98 on 752: gutshot");
    CHECK(cls("9c7c", "Jd8sKh").gutshot(), "97 on J8K: gutshot (T)");
    CHECK(cls("9c7c", "Jd8s5h").oesd() && cls("9c7c", "Jd8s5h").straight_outs == 2, "97 on J85: double gutter (6, T)");
    CHECK(cls("Ac2c", "3d4sKh").gutshot(), "A2 on 34K: wheel gutshot");
    CHECK(cls("AdQd", "Td7s2c").overcards && cls("AdQd", "Td7s2c").backdoor_fd, "AQ overs + BDFD");
    CHECK(cls("AhKd", "Ks7c2d").kicker == 12, "top pair, ace kicker");
    CHECK(cls("6h5h", "7h8h9c").flush_draw && cls("6h5h", "7h8h9c").made == Made::Straight,
          "made straight keeps its flush-draw redraw flag");

    // Range summary sanity: AA,KK on K72 -> sets 3/9 of KK... weights.
    RangeSummary s = summarize(R("AA, KK"), parse_hand("Kc7d2h"));
    CHECK(std::fabs(s.frac(Made::Set) - 3.0 / 9.0) < 1e-9 && std::fabs(s.frac(Made::Overpair) - 6.0 / 9.0) < 1e-9,
          "summary: 3 KK sets + 6 AA overpairs");
}

static void bench() {
    Range v = R("random");
    HandMask board = parse_hand("Kc9h4c7d2s");
    EquityResult e;
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 2000;
    for (int i = 0; i < reps; ++i) accumulate_river(board, v, e);
    std::printf("[bench] river all-combo equity: %.1f us per board (CPU, 1 thread)\n", secs_since(t0) / reps * 1e6);
}

int main() {
    test_combos();
    test_parse();
    test_removal_and_bayes();
    test_river_vs_brute_force();
    test_turn_flop_vs_brute_force();
    test_symmetry();
    test_known_matchups_and_mc();
    test_multiway();
    test_classify();
    bench();
    if (g_failures) {
        std::printf("\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("\nALL TESTS PASSED\n");
    return 0;
}
