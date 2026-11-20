# Design

## Goal

Play full-ring (9-handed) live NLHE cash at 100–300bb the way a strong reg plays
against unknowns. The bot doesn't need to be solver-exact. It needs a sensible
baseline and quick, well-calibrated adaptation.

## Architecture

```
            ┌───────────────────────────────┐
            │ Opponent model (per player)   │  θ ~ posterior over strategy params
            │ archetype-mixture prior       │  updated from every observed action
            └──────────────┬────────────────┘
                           │ π_θ(action | combo, state)
┌──────────────┐   ┌───────▼────────┐   ┌──────────────────────────────┐
│ Game engine  │──▶│ Range engine   │──▶│ Decision                     │
│ rules, pots  │   │ 1326-vectors,  │   │ baseline strategy, shifted    │
└──────────────┘   │ Bayes updates  │   │ toward best response by       │
                   └────────────────┘   │ confidence in the reads       │
                                        └──────────────────────────────┘
```

### Range engine
Every player's range is a 1326-weight vector, updated after each action:
`range[c] *= P(action | c, spot)`. The likelihood comes from the opponent model.
This is the single connection point between the baseline and the adaptation layer.
Weights are floored at ε so a range never collapses after a surprising action.
We track our own range with the same machinery.

### Baseline strategy
- **Preflop:** charts keyed by scenario (RFI, vs open, vs 3-bet/4-bet, iso vs N
  limpers, overlimp, squeeze), position, and stack-depth bucket, with live
  sizing rules. Anything off-chart falls back to: continue with the top X% by
  equity against the estimated range, where X comes from pot odds.
- **Postflop (range-level):** features (range and nut advantage, texture, SPR,
  position, initiative, number of players) → rule table picks size and frequency
  → each combo is assigned by equity against the *continuing* range. Bluffs fill
  the ratio implied by the size, chosen by blockers and lack of showdown value.
  Soft (sigmoid) thresholds, then sample.
- Build order: **river first, then turn, then flop.**

### Opponent model
- θ = knobs of the same parametric strategy function (looseness, aggression,
  per-street bluff propensity, calling-threshold shift, sizing–strength
  correlation, ...). "Plays like our baseline" is one particular θ.
- Prior: a mixture of archetypes (see below). Fast stats (VPIP/PFR) pin down
  the archetype quickly, which gives informative priors for slow postflop stats.
- v1: Beta-Binomial stat counts → archetype weights → θ.
  v2: particle filter over θ on GPU, marginalizing hidden hole cards.

### Decision
Best response to the posterior predictive opponent strategy, regularized toward
the baseline. The regularization protects against a misspecified model and
against opponents who adapt. Thompson-style sampling is a candidate for probing
lines. Which variant wins is decided by simulation.

### Tuning and evaluation
- Simulator with synthetic opponents drawn from known θ.
- Baseline knobs tuned with TuRBO / block-coordinate BO (river knobs first).
- Variance reduction is mandatory: duplicate deals, common random numbers,
  AIVAT-style control variates.
- Exploiter bots (always-bluff-river, station, check-raise-happy) as an
  exploitability proxy.
- Decision logs with reasoning, for human review of hand histories.

## Archetypes (draft, numbers to be corrected by the author)

| Archetype | VPIP/PFR | 3-bet | Fold to c-bet | River bluff | Notes |
|---|---|---|---|---|---|
| Maniac | 55/35 | 15% | low | high | random sizing, barrels |
| Nit | 15/10 | ~3% (QQ+/AK) | high | very low | weak-tight, folds to pressure |
| Reg | 24/18 | ~7% | medium | low | competent, autopilot leaks |
| Old man coffee | 14/5 | ~2% (KK+) | low (sticky) | ~0 | raise = nuts |
| Fish | 45/8 | rare | low | erratic | limp/call, sizing tells |
| Strong player | 22/17 | ~9% | balanced | balanced | ≈ our baseline |

## Hardware notes
Developed on Linux. CUDA code is written to compile with `nvcc`
(`PK_HD` = `__host__ __device__`) and is benchmarked on a separate NVIDIA box.
