//
// Created by Kazem on 2024-09-25.
//


#include <iostream>
#include <benchmark/benchmark.h>
#include "vec_mul.h"


static void BM_VECMUL(benchmark::State &state,
                      void (*vecImpl1)(std::vector<float> a, std::vector<float> b, std::vector<float> &c)) {
  int m = state.range(0);
  std::vector<float> A(m);
  std::vector<float> B(m);
  std::vector<float> C(m);
  for (int i = 0; i < m; ++i) {
    A[i] = 1.0;
  }
  for (int i = 0; i < m; ++i) {
    B[i] = 1.0;
  }

  for (auto _: state) {
    vecImpl1(A, B, C);
  }

}

BENCHMARK_CAPTURE(BM_VECMUL, baseline_vec_mul, swiftware::hpp::vec_mul)->Args({2048})->Args({4096})->Args({8192});
BENCHMARK_CAPTURE(BM_VECMUL, unrolled8_scalarized_vec_mul, swiftware::hpp::vec_mul_unrolled8_scalarized)->Args({2048})->Args({4096})->Args({8192});
#ifdef __AVX__
BENCHMARK_CAPTURE(BM_VECMUL, unrolled_avx_vec_mul, swiftware::hpp::vec_mul_unrolled_avx)->Args({2048})->Args({4096})->Args({8192});
#endif

BENCHMARK_CAPTURE(BM_VECMUL, parallel_vec_mul, swiftware::hpp::vec_mul_parallel)->Args({2048})->Args({4096})->Args({8192})->MeasureProcessCPUTime();
BENCHMARK_CAPTURE(BM_VECMUL, unrolled8_scalarized_parallel_vec_mul, swiftware::hpp::vec_mul_unrolled8_scalarized_parallel)->Args({2048})->Args({4096})->Args({8192});
#ifdef __AVX__
BENCHMARK_CAPTURE(BM_VECMUL, unrolled_avx_parallel_vec_mul, swiftware::hpp::vec_mul_unrolled_avx_parallel)->Args({2048})->Args({4096})->Args({8192});
#endif



BENCHMARK_MAIN();
