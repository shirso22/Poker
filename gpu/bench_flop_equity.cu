// Correctness + speed harness for poker::gpu::flop_equity.
//
// For several flops and villain ranges (narrow, wide, sparse random weights):
//   - CPU reference: poker::equity_vs_range (exact, double precision)
//   - GPU: your kernel, timed with CUDA events (warm-up + averaged repetitions)
// PASS requires every hero combo's equity to match within 1e-4, and
// den to match within 1e-4 relative (catches card-removal mistakes that the
// equity ratio can hide).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "flop_equity.cuh"
#include "poker/equity.hpp"

using namespace poker;

#define CUDA_CHECK(x)                                                                         \
    do {                                                                                      \
        cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                              \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)

struct Case {
    std::string name;
    HandMask board;
    Range villain;
};

int main(int argc, char** argv) {
    int reps = argc > 1 ? std::atoi(argv[1]) : 50;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    std::vector<Case> cases;
    auto parsed = [](const char* s) {
        Range r;
        Range::parse(s, r);
        return r;
    };
    cases.push_back({"Kc9h4c  vs random", parse_hand("Kc9h4c"), Range::uniform()});
    cases.push_back({"Ah7d2s  vs tight", parse_hand("Ah7d2s"), parsed("TT+, AQs+, AKo")});
    cases.push_back({"Th9h8c  vs medium", parse_hand("Th9h8c"),
                     parsed("22+, A2s+, K9s+, QTs+, J9s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s, ATo+, KTo+, QTo+, JTo")});
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    Range sparse;
    for (auto& w : sparse.w) w = U(rng) < 0.25f ? U(rng) : 0.0f;
    cases.push_back({"5s5d5h  vs sparse", parse_hand("5s5d5h"), sparse});  // trips board: lots of ties
    cases.push_back({"QsJsTs  vs sparse", parse_hand("QsJsTs"), sparse});  // monotone, straight-flush heavy

    float *d_villain, *d_num, *d_den;
    CUDA_CHECK(cudaMalloc(&d_villain, kNumCombos * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num, kNumCombos * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_den, kNumCombos * sizeof(float)));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    bool all_ok = true;
    std::printf("%-20s %10s %10s %9s %12s %12s  %s\n", "case", "CPU ms", "GPU ms", "speedup", "max |dEq|",
                "max rel dDen", "result");
    for (auto& c : cases) {
        // CPU reference
        auto c0 = std::chrono::steady_clock::now();
        EquityResult ref = equity_vs_range(c.board, c.villain);
        double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

        // GPU
        CUDA_CHECK(cudaMemcpy(d_villain, c.villain.w.data(), kNumCombos * sizeof(float), cudaMemcpyHostToDevice));
        gpu::flop_equity(c.board, d_villain, d_num, d_den);  // warm-up
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < reps; ++i) gpu::flop_equity(c.board, d_villain, d_num, d_den);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaGetLastError());
        float gpu_total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms, t0, t1));
        double gpu_ms = gpu_total_ms / reps;

        std::vector<float> num(kNumCombos), den(kNumCombos);
        CUDA_CHECK(cudaMemcpy(num.data(), d_num, kNumCombos * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(den.data(), d_den, kNumCombos * sizeof(float), cudaMemcpyDeviceToHost));

        double max_eq = 0, max_den = 0;
        int worst = -1;
        for (int h = 0; h < kNumCombos; ++h) {
            if (combo_mask(h) & c.board) continue;
            if (ref.den[h] <= 0) continue;  // empty villain range for this combo
            double rel = std::fabs(den[h] - ref.den[h]) / ref.den[h];
            double eq = den[h] > 0 ? num[h] / den[h] : -1.0;
            double d = std::fabs(eq - ref.equity(h));
            if (d > max_eq) { max_eq = d; worst = h; }
            max_den = std::max(max_den, rel);
        }
        bool ok = max_eq < 1e-4 && max_den < 1e-4;
        all_ok &= ok;
        std::printf("%-20s %10.2f %10.3f %8.1fx %12.2e %12.2e  %s", c.name.c_str(), cpu_ms, gpu_ms, cpu_ms / gpu_ms,
                    max_eq, max_den, ok ? "PASS" : "FAIL");
        if (!ok && worst >= 0)
            std::printf("  (worst: %s gpu %.5f cpu %.5f)", combo_name(worst).c_str(),
                        den[worst] > 0 ? num[worst] / den[worst] : -1.0, ref.equity(worst));
        std::printf("\n");
    }
    std::printf("\n%s\n", all_ok ? "ALL PASS" : "SOME CASES FAILED");
    return all_ok ? 0 : 1;
}
