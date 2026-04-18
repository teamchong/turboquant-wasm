/// @file
/// @brief platform abstraction over `flockfile`

#pragma once

#include <stdio.h>

static inline void lockfile(FILE *file) {
#if defined(_WIN32) && !defined(__CYGWIN__)
  _lock_file(file);
#else
  flockfile(file);
#endif
}

static inline void unlockfile(FILE *file) {
#if defined(_WIN32) && !defined(__CYGWIN__)
  _unlock_file(file);
#else
  funlockfile(file);
#endif
}
