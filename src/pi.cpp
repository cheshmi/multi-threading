// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2026 SwiftWare Lab


#include "pi.h"
#ifdef __AVX__
#include "immintrin.h"
#endif

namespace swiftware::hpp {

  // Function to compute Pi using a single thread (non-OpenMP)
  double compute_pi(long num_steps) {
    double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    for (long i = 0; i < num_steps; i++) {
      double x = (i + 0.5) * step;
      sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
  }

// Function to compute Pi using OpenMP
  double compute_pi_parallel(long num_steps, int num_threads) {
    double step = 1.0 / (double)num_steps;
    double pi = 0.0;

    // Allocate sum array dynamically based on the number of threads
    double *sum = (double *)malloc(num_threads * sizeof(double));
    if (sum == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return 0.0;
    }

    omp_set_num_threads(num_threads);
    int active_threads = num_threads;

#pragma omp parallel
    {
      int i, id, nthrds;
      double x;

      id = omp_get_thread_num();
      nthrds = omp_get_num_threads();

      // Ensure thread count synchronization if requested
      if (id == 0) {
        active_threads = nthrds;
      }

      sum[id] = 0.0;
      for (i = id; i < num_steps; i += nthrds) {
        x = (i + 0.5) * step;
        sum[id] += 4.0 / (1.0 + x * x);
      }
    }

    // Accumulate results from all threads
    for (int i = 0; i < active_threads; i++) {
      pi += sum[i] * step;
    }

    free(sum);
    return pi;
  }

}
