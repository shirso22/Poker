// Engine tests: hand-built rule scenarios + randomized invariant fuzzing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "poker/engine.hpp"

using namespace poker;

static int g_failures = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #cond);    \
            std::printf(__VA_ARGS__);                                       \
            std::printf("\n");                                              \
        }                                                                   \
    } while (0)
#define OK(expr) CHECK((expr) == ApplyError::Ok, "apply failed")

static std::vector<Card> cards(const char* s) {
    std::vector<Card> v;
    for (const char* p = s; p[0] && p[1]; p += 2) {
        if (*p == ' ') { --p; continue; }
        v.push_back(parse_card(std::string_view(p, 2)));
    }
    return v;
}

static Hand make(int n, std::vector<Chips> stacks, int button, const char* holes = "",
                 const char* board = "", Chips sb = 1, Chips bb = 2, Chips ante = 0) {
    TableConfig cfg{n, sb, bb, ante};
    auto h = cards(holes), b = cards(board);
    return Hand(cfg, stacks, button, make_deck(n, h, b));
}

// ---------------------------------------------------------------------------

static void test_heads_up_order() {
    Hand h = make(2, {100, 100}, 0);
    CHECK(h.small_blind_seat() == 0 && h.big_blind_seat() == 1, "HU: button posts SB");
    CHECK(h.to_act() == 0, "HU: button acts first preflop");
    auto la = h.legal_actions();
    CHECK(la.call_cost == 1 && la.can_raise && la.min_raise_to == 4 && la.max_raise_to == 100,
          "SB open: call 1, min raise to 4, got call=%lld min=%lld", (long long)la.call_cost,
          (long long)la.min_raise_to);
    OK(h.apply(Action::call()));
    CHECK(h.to_act() == 1 && h.street() == Street::Preflop, "BB gets option");
    la = h.legal_actions();
    CHECK(la.can_check && !la.can_fold && la.can_raise && la.min_raise_to == 4, "BB option legal set");
    CHECK(h.apply(Action::fold()) == ApplyError::IllegalAction, "cannot fold when check is free");
    OK(h.apply(Action::check()));
    CHECK(h.street() == Street::Flop && h.board_size() == 3, "flop dealt");
    CHECK(h.to_act() == 1, "HU: BB acts first postflop");
    OK(h.apply(Action::check()));
    OK(h.apply(Action::check()));
    CHECK(h.street() == Street::Turn && h.board_size() == 4, "turn");
    OK(h.apply(Action::check()));
    OK(h.apply(Action::check()));
    OK(h.apply(Action::check()));
    OK(h.apply(Action::check()));
    CHECK(h.is_over() && h.went_to_showdown(), "check-down reaches showdown");
    CHECK(h.final_stack(0) + h.final_stack(1) == 200, "chips conserved");
}

static void test_multiway_order_and_min_raise() {
    Hand h = make(6, {500, 500, 500, 500, 500, 500}, 0);
    // button 0, SB 1, BB 2, UTG 3
    CHECK(h.to_act() == 3, "UTG first preflop, got %d", h.to_act());
    OK(h.apply(Action::raise_to(6)));  // increment 4
    CHECK(h.legal_actions().min_raise_to == 10, "min re-raise to 10, got %lld",
          (long long)h.legal_actions().min_raise_to);
    CHECK(h.apply(Action::raise_to(9)) == ApplyError::BadAmount, "raise below min rejected");
    OK(h.apply(Action::raise_to(20)));  // increment 14
    CHECK(h.legal_actions().min_raise_to == 34, "min re-raise to 34, got %lld",
          (long long)h.legal_actions().min_raise_to);
    OK(h.apply(Action::fold()));   // 5
    OK(h.apply(Action::fold()));   // 0 (button)
    OK(h.apply(Action::fold()));   // 1 SB
    OK(h.apply(Action::fold()));   // 2 BB
    CHECK(h.to_act() == 3, "back to opener");
    OK(h.apply(Action::call()));
    CHECK(h.street() == Street::Flop && h.to_act() == 3, "first live left of button acts postflop, got %d",
          h.to_act());
    CHECK(h.legal_actions().min_raise_to == 2, "postflop min bet = BB");
}

static void test_short_allin_does_not_reopen() {
    // button 0 acts first 3-handed; seat1 has only 150.
    Hand h = make(3, {1000, 150, 1000}, 0);
    OK(h.apply(Action::raise_to(100)));   // seat0, increment 98
    OK(h.apply(Action::raise_to(150)));   // seat1 all-in, short (increment 50)
    CHECK(h.legal_actions().min_raise_to == 248, "next min raise = 150 + 98, got %lld",
          (long long)h.legal_actions().min_raise_to);
    OK(h.apply(Action::call()));          // seat2
    CHECK(h.to_act() == 0, "back to seat0");
    auto la = h.legal_actions();
    CHECK(la.can_call && la.call_cost == 50 && !la.can_raise, "short all-in must not reopen raising");
    CHECK(h.apply(Action::raise_to(400)) == ApplyError::IllegalAction, "raise rejected");
    OK(h.apply(Action::call()));
    CHECK(h.street() == Street::Flop, "street closes");
}

static void test_cumulative_short_allins_reopen() {
    Hand h = make(4, {1000, 150, 210, 1000}, 0);
    // button 0, SB 1, BB 2, seat3 first.
    OK(h.apply(Action::raise_to(100)));   // seat3, increment 98
    OK(h.apply(Action::fold()));          // seat0
    OK(h.apply(Action::raise_to(150)));   // seat1 all-in (short +50)
    OK(h.apply(Action::raise_to(210)));   // seat2 all-in (short +60)
    CHECK(h.to_act() == 3, "back to opener");
    auto la = h.legal_actions();
    CHECK(la.can_raise && la.min_raise_to == 308, "cumulative 110 >= 98 reopens; min to 308, got %lld",
          (long long)la.min_raise_to);
}

static void test_side_pots() {
    // Seat0 AA (50 chips), seat1 KK (100), seat2 QQ (200), seat3 45 (200).
    Hand h = make(4, {50, 100, 200, 200}, 0, "AsAh KsKh QsQh 4d5d", "2c7d9hJs3c");
    OK(h.apply(Action::raise_to(200)));  // seat3 all-in
    OK(h.apply(Action::call()));         // seat0 all-in 50
    OK(h.apply(Action::call()));         // seat1 all-in 100
    OK(h.apply(Action::call()));         // seat2 calls 200, all-in
    CHECK(h.is_over() && h.board_size() == 5, "runs out");
    CHECK(h.final_stack(0) == 200, "main pot 4x50 -> AA, got %lld", (long long)h.final_stack(0));
    CHECK(h.final_stack(1) == 150, "side pot 3x50 -> KK, got %lld", (long long)h.final_stack(1));
    CHECK(h.final_stack(2) == 200, "side pot 2x100 -> QQ, got %lld", (long long)h.final_stack(2));
    CHECK(h.final_stack(3) == 0, "45 loses everything");
}

static void test_uncalled_bet_returned() {
    Hand h = make(2, {1000, 100}, 0, "2c3d AsAh", "KdQc9s5h4c");
    OK(h.apply(Action::raise_to(1000)));  // SB shoves
    OK(h.apply(Action::call()));          // BB calls 100 all-in
    CHECK(h.is_over(), "over");
    CHECK(h.final_stack(1) == 200 && h.final_stack(0) == 900,
          "AA wins 200, uncalled 900 returned; got %lld / %lld", (long long)h.final_stack(1),
          (long long)h.final_stack(0));
}

static void test_split_pot_odd_chip() {
    Hand h = make(3, {100, 100, 100}, 0, "2c3d 4c5d 6c7d", "TsJsQdKcAh");
    OK(h.apply(Action::call()));   // button limps 2
    OK(h.apply(Action::fold()));   // SB folds (1 dead)
    OK(h.apply(Action::check()));  // BB
    for (int k = 0; k < 6; ++k) OK(h.apply(Action::check()));
    CHECK(h.is_over() && h.pot() == 5, "pot 5");
    CHECK(h.won(2) == 3 && h.won(0) == 2, "odd chip to first winner left of button (BB): got %lld/%lld",
          (long long)h.won(2), (long long)h.won(0));
}

static void test_fold_to_bb() {
    Hand h = make(4, {100, 100, 100, 100}, 1);  // button 1, SB 2, BB 3, first 0
    OK(h.apply(Action::fold()));
    OK(h.apply(Action::fold()));
    OK(h.apply(Action::fold()));
    CHECK(h.is_over() && !h.went_to_showdown(), "walk");
    CHECK(h.net(3) == 1 && h.net(2) == -1, "BB wins the SB");
}

static void test_short_blind_and_ante() {
    // BB has 1 chip (blind 2): posts all-in short; SB completes to the full BB.
    Hand h = make(2, {100, 1}, 0, "AsAh 2c3d", "KdQc9s5h4c");
    CHECK(h.all_in(1), "BB all-in from the blind");
    CHECK(h.legal_actions().call_cost == 1, "SB owes the full BB");
    OK(h.apply(Action::call()));
    CHECK(h.is_over(), "runs out");
    CHECK(h.final_stack(0) == 101 && h.final_stack(1) == 0, "SB wins 1 net; its extra 1 returned");

    Hand a = make(3, {100, 100, 100}, 0, "", "", 1, 2, 1);
    CHECK(a.pot() == 6, "3 antes + blinds = 6, got %lld", (long long)a.pot());
    CHECK(a.legal_actions().call_cost == 2, "antes are dead, not part of the bet");
}

static void test_bb_option_raise() {
    Hand h = make(3, {100, 100, 100}, 0);
    OK(h.apply(Action::call()));
    OK(h.apply(Action::call()));
    CHECK(h.to_act() == 2 && h.legal_actions().can_raise, "BB option to raise");
    OK(h.apply(Action::raise_to(8)));
    CHECK(h.to_act() == 0 && h.legal_actions().call_cost == 6, "limpers face the raise");
}

// ---------------------------------------------------------------------------
// Fuzz: random legal play, check invariants and compare showdown payouts with
// an independent per-chip-layer reference (exact up to odd-chip rounding).

static void fuzz(int hands) {
    std::mt19937_64 rng(2026);
    long long actions = 0, showdowns = 0, max_err_hands = 0;
    for (int t = 0; t < hands; ++t) {
        int n = 2 + int(rng() % 9);
        std::vector<Chips> stacks(n);
        for (auto& s : stacks) s = 1 + Chips(rng() % 400);
        Chips ante = (rng() % 4 == 0) ? 1 : 0;
        Deck d;
        for (int i = 0; i < 52; ++i) d[i] = Card(i);
        std::shuffle(d.begin(), d.end(), rng);
        int button = int(rng() % n);
        Hand h(TableConfig{n, 1, 2, ante}, stacks, button, d);
        Chips total0 = 0;
        for (auto s : stacks) total0 += s;

        int steps = 0;
        int last_street = 0;
        while (!h.is_over()) {
            if (++steps > 2000) { CHECK(false, "hand did not terminate"); break; }
            auto la = h.legal_actions();
            CHECK(la.can_check || la.can_call, "player to act must be able to check or call");
            if (la.can_raise)
                CHECK(la.min_raise_to > h.current_bet() && la.min_raise_to <= la.max_raise_to,
                      "raise bounds sane");
            Action a;
            double u = std::uniform_real_distribution<>(0, 1)(rng);
            if (la.can_fold && u < 0.2) a = Action::fold();
            else if (la.can_raise && u > 0.7) {
                Chips amt = (u > 0.95) ? la.max_raise_to
                                       : la.min_raise_to + Chips(rng() % (la.max_raise_to - la.min_raise_to + 1));
                a = Action::raise_to(amt);
            } else a = la.can_check ? Action::check() : Action::call();
            CHECK(h.apply(a) == ApplyError::Ok, "legal action rejected");
            ++actions;
            Chips sum = h.pot();
            for (int i = 0; i < n; ++i) {
                sum += h.stack(i);
                CHECK(h.stack(i) >= 0, "negative stack");
            }
            // Once over, stacks already include winnings while pot() still reports contributions.
            if (!h.is_over()) CHECK(sum == total0, "chips not conserved mid-hand");
            CHECK(int(h.street()) >= last_street, "street went backwards");
            last_street = int(h.street());
        }
        Chips total1 = 0;
        for (int i = 0; i < n; ++i) total1 += h.final_stack(i);
        CHECK(total1 == total0, "chips not conserved at end: %lld vs %lld", (long long)total1, (long long)total0);

        if (!h.went_to_showdown()) {
            int live = 0;
            for (int i = 0; i < n; ++i) live += !h.folded(i);
            CHECK(live == 1, "no showdown but %d live players", live);
            continue;
        }
        ++showdowns;
        // Reference: walk chip layers 1..max; each layer's chips (one per contributor)
        // split equally among the best live hands that reach that layer.
        HandMask board = 0;
        for (int k = 0; k < 5; ++k) board |= card_bit(h.board(k));
        std::vector<uint32_t> str(n, 0);
        for (int i = 0; i < n; ++i)
            if (!h.folded(i)) str[i] = evaluate(board | card_bit(h.hole(i)[0]) | card_bit(h.hole(i)[1]));
        Chips maxc = 0;
        for (int i = 0; i < n; ++i) maxc = std::max(maxc, h.total_commit(i));
        Chips max_live = 0;
        for (int i = 0; i < n; ++i)
            if (!h.folded(i)) max_live = std::max(max_live, h.total_commit(i));
        std::vector<double> ref(n, 0.0);
        bool any_split = false;
        for (Chips x = 1; x <= maxc; ++x) {
            int contributors = 0;
            for (int i = 0; i < n; ++i) contributors += h.total_commit(i) >= x;
            Chips lvl = std::min(x, max_live);  // dead money above the top live level joins the top pot
            uint32_t best = 0;
            for (int i = 0; i < n; ++i)
                if (!h.folded(i) && h.total_commit(i) >= lvl) best = std::max(best, str[i]);
            int w = 0;
            for (int i = 0; i < n; ++i) w += (!h.folded(i) && h.total_commit(i) >= lvl && str[i] == best);
            any_split |= w > 1;
            for (int i = 0; i < n; ++i)
                if (!h.folded(i) && h.total_commit(i) >= lvl && str[i] == best) ref[i] += double(contributors) / w;
        }
        bool bad = false;
        for (int i = 0; i < n; ++i)
            // No ties anywhere: must match exactly. Ties: odd chips allow < 1 chip per pot.
            if (std::fabs(double(h.won(i)) - ref[i]) > (any_split ? n : 0)) bad = true;
        if (bad) {
            ++max_err_hands;
            if (max_err_hands <= 3) {
                std::printf("  payout mismatch (n=%d):", n);
                for (int i = 0; i < n; ++i) std::printf(" [%lld vs %.1f]", (long long)h.won(i), ref[i]);
                std::printf("\n");
            }
        }
    }
    CHECK(max_err_hands == 0, "%lld showdown payouts disagree with reference", max_err_hands);
    std::printf("[fuzz] %d hands, %lld actions, %lld showdowns, %lld payout mismatches\n", hands, actions,
                showdowns, max_err_hands);
}

int main() {
    test_heads_up_order();
    test_multiway_order_and_min_raise();
    test_short_allin_does_not_reopen();
    test_cumulative_short_allins_reopen();
    test_side_pots();
    test_uncalled_bet_returned();
    test_split_pot_odd_chip();
    test_fold_to_bb();
    test_short_blind_and_ante();
    test_bb_option_raise();
    std::printf("[scenarios] done\n");
    fuzz(200000);
    if (g_failures) {
        std::printf("\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("\nALL TESTS PASSED\n");
    return 0;
}
