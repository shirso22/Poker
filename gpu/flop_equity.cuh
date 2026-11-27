#pragma once
// GPU flop equity: every hero combo vs a villain range, summed over all
// 1081 turn+river runouts of a 3-card board.
//
// Output contract (same as the CPU reference poker::equity_vs_range):
//   num[h] = sum over runouts r, villain combos v (disjoint from h and the board)
//            of w[v] * (1 if h wins, 0.5 if tie, 0 if loses)
//   den[h] = sum over the same (r, v) of w[v]
//   equity(h) = num[h] / den[h]
// Hero combos that collide with the board get num = den = 0.
//
// All pointers are device pointers to kNumCombos (1326) floats.

#include <cuda_runtime.h>

#include "poker/cards.hpp"

namespace poker::gpu {

void flop_equity(HandMask board, const float* d_villain, float* d_num, float* d_den, cudaStream_t stream = 0);

}  // namespace poker::gpu
