# Poker

An exploitative No-Limit Hold'em cash-game bot, built to play like a strong live
reg sitting down with strangers: a reasonable baseline strategy, population
priors, and fast Bayesian reads that move it toward exploiting each opponent.

Also a vehicle for learning GPU programming. The hot paths (hand evaluation,
range-vs-range equity, opponent-model inference, batched simulation) are written
to run on both CPU and CUDA.

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture.

## Status

| Component | State |
|---|---|
| 7-card hand evaluator (host + device) | ✅ done, exhaustively tested |
| Game engine (2–10 players, side pots, TDA reopen rules) | ✅ done, fuzz-tested (straddles TODO) |
| Range engine | next |
| Baseline strategy (river → turn → flop) | planned |
| Opponent model | planned |
| GPU kernels / simulator | planned |

The original 2025 prototype lives in [`legacy/`](legacy/).

## Build and test

```bash
cmake -S . -B build
cmake --build build -j
./build/test_eval
./build/test_engine
```

`test_eval` enumerates all 2,598,960 five-card and 133,784,560 seven-card hands
and checks category counts and equivalence-class counts (7462 / 4824). It also
compares 200k random hand pairs against an independent reference evaluator.

`test_engine` covers rule scenarios (heads-up order, BB option, min-raise,
short all-ins that do or don't reopen betting, side pots, uncalled bets, odd
chips, short blinds, antes). It then plays 200k random hands, checking chip
conservation and comparing every showdown payout against an independent
per-chip reference.
