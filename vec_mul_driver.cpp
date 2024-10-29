//
// Created by Kazem on 2024-09-25.
//


#include <iostream>
#include <benchmark/benchmark.h>
#include "vec_mul.h"

int num_teach_threads = 12;

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


static void BM_VECMUL_PARALLEL(benchmark::State &state,
                               void (*vecImpl1)(std::vector<float> a, std::vector<float> b, std::vector<float> &c, int num_threads)) {
  int m = state.range(0);
  int num_threads = state.range(1);
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
    vecImpl1(A, B, C, num_threads);
  }
}

BENCHMARK_CAPTURE(BM_VECMUL, baseline_vec_mul, swiftware::hpp::vec_mul)->Ranges({{2<<10, 2<<18}});
BENCHMARK_CAPTURE(BM_VECMUL, unrolled8_scalarized_vec_mul, swiftware::hpp::vec_mul_unrolled8_scalarized)->Ranges({{2<<10, 2<<18}});
#ifdef __AVX__
BENCHMARK_CAPTURE(BM_VECMUL, unrolled_avx_vec_mul, swiftware::hpp::vec_mul_unrolled_avx)->Ranges({{2<<10, 2<<18}});
#endif

BENCHMARK_CAPTURE(BM_VECMUL_PARALLEL, parallel_vec_mul, swiftware::hpp::vec_mul_parallel)->Ranges({{2<<10, 2<<18}, {2, 12}});
BENCHMARK_CAPTURE(BM_VECMUL_PARALLEL, unrolled8_scalarized_parallel_vec_mul, swiftware::hpp::vec_mul_unrolled8_scalarized_parallel)->Ranges({{2<<10, 2<<18}, {2, 12}});
#ifdef __AVX__
BENCHMARK_CAPTURE(BM_VECMUL_PARALLEL, unrolled_avx_parallel_vec_mul, swiftware::hpp::vec_mul_unrolled_avx_parallel)->Ranges({{2<<10, 2<<18}, {2, 12}});
#endif



BENCHMARK_MAIN();
