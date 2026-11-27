// YOUR KERNEL GOES HERE.
//
// Contract: see flop_equity.cuh. Check it with:
//   cmake -S . -B build -DPOKER_CUDA=ON && cmake --build build -j && ./build/bench_flop_equity
//
// Useful facts:
//  - poker::evaluate() (eval.hpp) is __host__ __device__: call it straight from a kernel.
//  - card_bit(), rank_of(), suit_of() (cards.hpp) are device-callable too.
//  - combo_index() is constexpr and works on device thanks to --expt-relaxed-constexpr.
//    The kCombos lookup table (combos.hpp) is a HOST array, so it is NOT visible
//    to device code. Getting the combo -> cards mapping onto the GPU is part of the exercise.
//  - The CPU reference is accumulate_river() / equity_vs_range() in equity.hpp.

#include "flop_equity.cuh"

namespace poker::gpu {

void flop_equity(HandMask board, const float* d_villain, float* d_num, float* d_den, cudaStream_t stream) {
    (void)board;
    (void)d_villain;
    // TODO: launch your kernel(s). Until then this zero-fills, and the benchmark reports FAIL.
    cudaMemsetAsync(d_num, 0, 1326 * sizeof(float), stream);
    cudaMemsetAsync(d_den, 0, 1326 * sizeof(float), stream);
}

}  // namespace poker::gpu
