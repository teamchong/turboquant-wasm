/// @file
/// @brief Implementation of random number generation functionality

#include "config.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <util/alloc.h>
#include <util/gv_math.h>
#include <util/random.h>

int *gv_permutation(int bound) {
  if (bound <= 0) {
    return NULL;
  }

  // initialize a sequence `{0, 1, …, bound - 1}`
  int *const p = gv_calloc((size_t)bound, sizeof(int));
  for (int i = 0; i < bound; i++) {
    p[i] = i;
  }

  // perform a Fisher-Yates shuffle
  for (int i = bound - 1; i > 0; --i) {
    const int j = gv_random(i + 1);
    SWAP(&p[i], &p[j]);
  }

  return p;
}

/// handle random number generation, `bound ≤ RAND_MAX`
static int random_small(int bound) {
  assert(bound > 0);
  assert(bound <= RAND_MAX);

  // The interval `[0, RAND_MAX]` is not necessarily neatly divided into
  // `bound`-sized chunks. E.g. using a bound of 3 with a `RAND_MAX` of 7:
  //   ┌───┬───┬───┬───┬───┬───┬───┬───┐
  //   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
  //   └───┴───┴───┴───┴───┴───┴───┴───┘
  //    ◄─────────► ◄─────────► ◄─────►
  //     3 values    3 values   2 values
  // To guarantee a uniform distribution, derive the upper bound of the last
  // complete chunk (5 in the example above), above which we discard and
  // resample to avoid pulling from the partial trailing chunk.
  const int discard_threshold =
      RAND_MAX - (int)(((unsigned)RAND_MAX + 1) % (unsigned)bound);

  int r;
  do {
    r = rand();
  } while (r > discard_threshold);

  return r % bound;
}

uint64_t gv_random_u64(uint64_t bound) {
  assert(bound > 0);

  // See comment in `random_small`, but note that our maximum generated value
  // here will be `UINT64_MAX` instead of `RAND_MAX`. Note that we need to do a
  // slightly different calculation to avoid the integer overflow that would
  // otherwise result from `UINT64_MAX + 1`.
  const uint64_t discard_threshold =
      UINT64_MAX - (UINT64_MAX - bound + 1) % bound;

  uint64_t r;
  do {
    // generate a random `uint64_t` value
    r = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      // `RAND_MAX ≥ 32767` is guaranteed, so `random_small(256)` is safe
      const uint8_t byte = (uint8_t)random_small((int)UINT8_MAX + 1);
      memcpy((char *)&r + i, &byte, sizeof(byte));
    }
  } while (r > discard_threshold);

  return r % bound;
}

int gv_random(int bound) {
  assert(bound > 0);

  if (bound > RAND_MAX) {
    _Static_assert(INT_MAX <= UINT64_MAX,
                   "the `int` type includes non-negative values that do not "
                   "fit in a `uint64_t`, hence some `int` values can never be "
                   "returned by `gv_random_u64`");
    return (int)gv_random_u64((uint64_t)bound);
  }
  return random_small(bound);
}
