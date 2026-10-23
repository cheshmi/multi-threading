//
// Created by Kazem on 2026-10-23.
//

#ifndef MULTI_THREADING_PI_H
#define MULTI_THREADING_PI_H

namespace swiftware::hpp {

  double compute_pi(long num_steps);

  double compute_pi_parallel(long num_steps, int num_threads);

}
#endif //MULTI_THREADING_PI_H
