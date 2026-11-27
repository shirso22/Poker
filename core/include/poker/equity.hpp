#pragma once
// Equity of every hero combo against a villain range.
//
// River core (accumulate_river): sort all combos by strength once, sweep in
// strength order keeping running weight totals overall and per card. For hero
// combo h = {x, y}, villain weight that doesn't collide with h is
//   S - S[x] - S[y] + w[h]        (inclusion–exclusion; w[h] was subtracted twice)
// applied separately to "weaker than h", "tied with h", and "all". That gives
// all 1326 equities in O(n log n) instead of O(n^2).
//
// Results accumulate as numerator/denominator per hero combo:
//   num[h] = sum over (villain combo v, runout) of w[v] * (win + tie/2)
//   den[h] = sum over (v, runout) of w[v]
// so summing over runouts is exact, and range-vs-range equity is
//   sum_h wh*num[h] / sum_h wh*den[h].

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "poker/eval.hpp"
#include "poker/range.hpp"

namespace poker {

struct EquityResult {
    std::array<double, kNumCombos> num{}, den{};

    double equity(int c) const { return den[c] > 0 ? num[c] / den[c] : std::nan(""); }

    void add(const EquityResult& o) {
        for (int c = 0; c < kNumCombos; ++c) {
            num[c] += o.num[c];
            den[c] += o.den[c];
        }
    }
};

// Exact hand-vs-range for a complete 5-card board; adds into `out`.
inline void accumulate_river(HandMask board, const Range& villain, EquityResult& out) {
    uint64_t keys[kNumCombos];
    int n = 0;
    for (int c = 0; c < kNumCombos; ++c) {
        if (combo_mask(c) & board) continue;
        keys[n++] = (uint64_t(evaluate(board | combo_mask(c))) << 11) | uint64_t(c);
    }
    std::sort(keys, keys + n);

    double total = 0, card_all[52] = {};
    for (int k = 0; k < n; ++k) {
        int c = int(keys[k] & 0x7FF);
        double w = villain.w[c];
        total += w;
        card_all[kCombos.hi[c]] += w;
        card_all[kCombos.lo[c]] += w;
    }

    double less = 0, card_less[52] = {}, grp_card[52] = {};
    int i = 0;
    while (i < n) {
        int j = i;
        uint64_t s = keys[i] >> 11;
        double grp = 0;
        while (j < n && (keys[j] >> 11) == s) {
            int c = int(keys[j] & 0x7FF);
            double w = villain.w[c];
            grp += w;
            grp_card[kCombos.hi[c]] += w;
            grp_card[kCombos.lo[c]] += w;
            ++j;
        }
        for (int k = i; k < j; ++k) {
            int c = int(keys[k] & 0x7FF);
            Card x = kCombos.hi[c], y = kCombos.lo[c];
            double self = villain.w[c];
            double weaker = less - card_less[x] - card_less[y];
            double tied = grp - grp_card[x] - grp_card[y] + self;
            double all = total - card_all[x] - card_all[y] + self;
            out.num[c] += weaker + 0.5 * tied;
            out.den[c] += all;
        }
        for (int k = i; k < j; ++k) {
            int c = int(keys[k] & 0x7FF);
            Card x = kCombos.hi[c], y = kCombos.lo[c];
            card_less[x] += grp_card[x];
            card_less[y] += grp_card[y];
            grp_card[x] = grp_card[y] = 0;
        }
        less += grp;
        i = j;
    }
}

inline int popcount64(HandMask m) { return __builtin_popcountll(m); }

// Exact: enumerates all remaining board cards (board must have 3, 4 or 5 cards).
inline EquityResult equity_vs_range(HandMask board, const Range& villain) {
    EquityResult r;
    int k = popcount64(board);
    if (k == 5) {
        accumulate_river(board, villain, r);
    } else if (k == 4) {
        for (int t = 0; t < 52; ++t)
            if (!(board & card_bit(Card(t)))) accumulate_river(board | card_bit(Card(t)), villain, r);
    } else if (k == 3) {
        for (int t = 0; t < 52; ++t) {
            if (board & card_bit(Card(t))) continue;
            for (int u = t + 1; u < 52; ++u)
                if (!(board & card_bit(Card(u))))
                    accumulate_river(board | card_bit(Card(t)) | card_bit(Card(u)), villain, r);
        }
    }
    return r;
}

// Monte Carlo over the missing board cards (any board size 0..5, e.g. preflop).
template <class Rng>
EquityResult equity_vs_range_mc(HandMask board, const Range& villain, int samples, Rng& rng) {
    EquityResult r;
    Card avail[52];
    int na = 0;
    for (int c = 0; c < 52; ++c)
        if (!(board & card_bit(Card(c)))) avail[na++] = Card(c);
    int need = 5 - popcount64(board);
    for (int s = 0; s < samples; ++s) {
        HandMask b = board;
        for (int k = 0; k < need; ++k) {  // partial Fisher–Yates
            int j = k + int(rng() % uint64_t(na - k));
            std::swap(avail[k], avail[j]);
            b |= card_bit(avail[k]);
        }
        accumulate_river(b, villain, r);
    }
    return r;
}

// Range-vs-range equity from a hero range and a per-combo EquityResult.
inline double range_equity(const Range& hero, const EquityResult& e) {
    double num = 0, den = 0;
    for (int c = 0; c < kNumCombos; ++c) {
        if (hero.w[c] <= 0) continue;
        num += hero.w[c] * e.num[c];
        den += hero.w[c] * e.den[c];
    }
    return den > 0 ? num / den : std::nan("");
}

// ---------------------------------------------------------------------------
// Multiway: one hero combo vs several villain ranges, Monte Carlo.
// Villain hands are drawn jointly by rejection (independent draws from each
// range, rejected on card collisions), which samples exactly from
//   P(v1..vk) ∝ Π w_i(v_i) · [no shared cards].
// Hero's share of the pot: 1/(number of tied winners) if hero wins, else 0.

struct MultiwayResult {
    double equity = 0;
    int samples = 0;
    int rejected = 0;
};

template <class Rng>
MultiwayResult multiway_equity(int hero_combo, HandMask board, std::span<const Range> villains, int samples,
                               Rng& rng) {
    MultiwayResult res;
    const HandMask hero = combo_mask(hero_combo);
    const HandMask dead = board | hero;
    const int k = int(villains.size());
    std::vector<std::vector<double>> cdf(k);
    std::vector<std::vector<int>> ids(k);
    for (int i = 0; i < k; ++i) {
        double acc = 0;
        for (int c = 0; c < kNumCombos; ++c) {
            if (villains[i].w[c] <= 0 || (combo_mask(c) & dead)) continue;
            acc += villains[i].w[c];
            cdf[i].push_back(acc);
            ids[i].push_back(c);
        }
        if (ids[i].empty()) return res;  // villain range empty after card removal
    }
    std::uniform_real_distribution<double> U(0, 1);
    double total_share = 0;
    std::vector<int> pick(k);
    while (res.samples < samples) {
        HandMask used = dead;
        bool ok = true;
        for (int i = 0; i < k && ok; ++i) {
            double u = U(rng) * cdf[i].back();
            int j = int(std::upper_bound(cdf[i].begin(), cdf[i].end(), u) - cdf[i].begin());
            if (j >= int(ids[i].size())) j = int(ids[i].size()) - 1;
            pick[i] = ids[i][j];
            HandMask m = combo_mask(pick[i]);
            if (m & used) ok = false;
            used |= m;
        }
        if (!ok) {
            ++res.rejected;
            if (res.rejected > 1000 * (samples + 1)) break;  // ranges nearly incompatible
            continue;
        }
        HandMask b = board;
        int need = 5 - popcount64(board);
        while (need > 0) {
            Card c = Card(rng() % 52);
            if ((used | b) & card_bit(c)) continue;
            b |= card_bit(c);
            --need;
        }
        uint32_t hs = evaluate(b | hero), best = hs;
        int winners = 1;
        bool hero_best = true;
        for (int i = 0; i < k; ++i) {
            uint32_t vs = evaluate(b | combo_mask(pick[i]));
            if (vs > best) { best = vs; winners = 1; hero_best = false; }
            else if (vs == best) ++winners;
        }
        total_share += hero_best ? 1.0 / winners : 0.0;
        ++res.samples;
    }
    res.equity = res.samples ? total_share / res.samples : std::nan("");
    return res;
}

}  // namespace poker
