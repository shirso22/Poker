#pragma once
// A range: one non-negative weight per combo (unnormalized relative likelihood).
//
// Bayesian updates multiply by P(action | combo). After each update weights are
// rescaled so the max is 1 (no underflow over long histories), and any combo that
// was possible before the update is floored at `floor`, so a surprising action
// never wipes out the range. Card-blocked combos stay exactly 0.

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

#include "poker/combos.hpp"

namespace poker {

struct Range {
    std::array<float, kNumCombos> w{};

    static Range uniform() {
        Range r;
        r.w.fill(1.0f);
        return r;
    }

    // Parses e.g. "QQ+, AKs, AKo:0.5, A5s-A2s, 77-22, KTs+, AhKh, random".
    // Later tokens override earlier ones. Returns false (and sets *err) on bad input.
    static bool parse(std::string_view text, Range& out, std::string* err = nullptr);

    float total() const {
        double t = 0;
        for (float x : w) t += x;
        return float(t);
    }
    int support() const {
        int n = 0;
        for (float x : w) n += x > 0;
        return n;
    }
    void normalize() {
        float t = total();
        if (t > 0)
            for (float& x : w) x /= t;
    }
    // Zero every combo that uses a dead card (board, our hole cards, ...).
    void remove(HandMask dead) {
        for (int c = 0; c < kNumCombos; ++c)
            if (combo_mask(c) & dead) w[c] = 0;
    }

};

// range[c] *= likelihood[c], rescale to max 1, floor combos that were in support.
// If the likelihood zeroes everything, the floor leaves a uniform range over the
// prior support ("he did something impossible under our model": know nothing).
inline void bayes_update(Range& r, const float* likelihood, float floor = 1e-4f) {
    bool in_support[kNumCombos];
    float mx = 0;
    for (int c = 0; c < kNumCombos; ++c) {
        in_support[c] = r.w[c] > 0;
        if (!in_support[c]) continue;
        r.w[c] *= std::max(likelihood[c], 0.0f);
        mx = std::max(mx, r.w[c]);
    }
    for (int c = 0; c < kNumCombos; ++c) {
        if (!in_support[c]) continue;
        float v = mx > 0 ? r.w[c] / mx : 0.0f;
        r.w[c] = std::max(v, floor);
    }
}

// One observed step: cards that became visible, and P(action | combo).
struct Observation {
    HandMask dead = 0;
    std::array<float, kNumCombos> likelihood{};
};

// Rebuilds a range from its prior and the full observation sequence. Used when
// the opponent model (and therefore every likelihood) changes mid-hand.
template <class ObsAt>
Range replay(Range prior, int steps, ObsAt&& obs_at, float floor = 1e-4f) {
    for (int k = 0; k < steps; ++k) {
        const Observation& o = obs_at(k);
        prior.remove(o.dead);
        bayes_update(prior, o.likelihood.data(), floor);
    }
    return prior;
}

// ---------------------------------------------------------------------------
// Range string parsing.

namespace detail {

inline int rank_from_char(char ch) {
    auto p = kRankChars.find(char(std::toupper(ch)));
    return p == std::string_view::npos ? -1 : int(p);
}

// kind: 's' suited, 'o' offsuit, 0 = both. Pairs ignore kind.
inline void set_class(Range& r, int hi, int lo, char kind, float weight) {
    for (int c = 0; c < kNumCombos; ++c) {
        Card a = kCombos.hi[c], b = kCombos.lo[c];
        int ra = rank_of(a), rb = rank_of(b);
        int h = std::max(ra, rb), l = std::min(ra, rb);
        if (h != hi || l != lo) continue;
        bool suited = suit_of(a) == suit_of(b);
        if (hi != lo && ((kind == 's' && !suited) || (kind == 'o' && suited))) continue;
        r.w[c] = weight;
    }
}

struct Shape {
    int hi = -1, lo = -1;
    char kind = 0;
};

inline bool parse_shape(std::string_view s, Shape& out) {
    if (s.size() < 2 || s.size() > 3) return false;
    out.hi = rank_from_char(s[0]);
    out.lo = rank_from_char(s[1]);
    if (out.hi < 0 || out.lo < 0) return false;
    if (out.hi < out.lo) std::swap(out.hi, out.lo);
    out.kind = 0;
    if (s.size() == 3) {
        char k = char(std::tolower(s[2]));
        if (k != 's' && k != 'o') return false;
        if (out.hi == out.lo) return false;
        out.kind = k;
    }
    return true;
}

inline std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.remove_prefix(1);
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.remove_suffix(1);
    return s;
}

}  // namespace detail

inline bool Range::parse(std::string_view text, Range& out, std::string* err) {
    using namespace detail;
    out = Range{};
    auto fail = [&](std::string_view tok) {
        if (err) *err = "bad range token: '" + std::string(tok) + "'";
        return false;
    };
    while (!text.empty()) {
        size_t comma = text.find(',');
        std::string_view tok = trim(text.substr(0, comma));
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        if (tok.empty()) continue;

        float weight = 1.0f;
        if (size_t colon = tok.find(':'); colon != std::string_view::npos) {
            try {
                weight = std::stof(std::string(tok.substr(colon + 1)));
            } catch (...) {
                return fail(tok);
            }
            tok = trim(tok.substr(0, colon));
        }

        if (tok == "random" || tok == "*") {
            out.w.fill(weight);
            continue;
        }
        // Specific combo, e.g. "AhKh".
        if (tok.size() == 4 && kSuitChars.find(tok[1]) != std::string_view::npos &&
            kSuitChars.find(tok[3]) != std::string_view::npos) {
            Card a = parse_card(tok.substr(0, 2)), b = parse_card(tok.substr(2, 2));
            if (a == 255 || b == 255 || a == b) return fail(tok);
            out.w[combo_index(a, b)] = weight;
            continue;
        }
        // Span: "A5s-A2s", "77-22".
        if (size_t dash = tok.find('-'); dash != std::string_view::npos) {
            Shape x, y;
            if (!parse_shape(tok.substr(0, dash), x) || !parse_shape(tok.substr(dash + 1), y)) return fail(tok);
            if (x.kind != y.kind) return fail(tok);
            if (x.hi == x.lo && y.hi == y.lo) {
                for (int r = std::min(x.hi, y.hi); r <= std::max(x.hi, y.hi); ++r) set_class(out, r, r, 0, weight);
            } else if (x.hi == y.hi && x.hi != x.lo && y.hi != y.lo) {
                for (int l = std::min(x.lo, y.lo); l <= std::max(x.lo, y.lo); ++l)
                    set_class(out, x.hi, l, x.kind, weight);
            } else {
                return fail(tok);
            }
            continue;
        }
        // Plus: "QQ+", "ATs+".
        bool plus = tok.back() == '+';
        if (plus) tok.remove_suffix(1);
        Shape s;
        if (!parse_shape(tok, s)) return fail(tok);
        if (!plus) set_class(out, s.hi, s.lo, s.kind, weight);
        else if (s.hi == s.lo)
            for (int r = s.hi; r <= 12; ++r) set_class(out, r, r, 0, weight);
        else
            for (int l = s.lo; l < s.hi; ++l) set_class(out, s.hi, l, s.kind, weight);
    }
    return true;
}

}  // namespace poker
