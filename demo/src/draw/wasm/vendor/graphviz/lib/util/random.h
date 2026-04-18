/// @file
/// @brief random number generation

#pragma once

#include <stdint.h>
#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// generate a random permutation of the numbers `[0, bound - 1]`
///
/// The caller is responsible for `free`ing the returned array. This function
/// calls `exit` on memory allocation failure.
///
/// @param bound Exclusive upper bound on the sequence
/// @return A permutation of `[0, bound - 1]`
UTIL_API int *gv_permutation(int bound);

/// generate a random number in the range `[0, bound - 1]`
///
/// This function assumes the caller has previously seeded the `rand` random
/// number generator.
///
/// @param bound Exclusive upper bound on random number generation
/// @return A random number drawn from a uniform distribution
UTIL_API int gv_random(int bound);

/// generate a random 64-bit unsigned number in the range `[0, bound - 1]`
///
/// This function assumes the caller has previously seeded the `rand` random
/// number generator.
///
/// @param bound Exclusive upper bound on random number generation
/// @return A random number drawn from a uniform distribution
UTIL_API uint64_t gv_random_u64(uint64_t bound);

#ifdef __cplusplus
}
#endif
