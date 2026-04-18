/// @file
/// @ingroup cgraph_utils
#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <iostream>

extern "C" {
#endif

#ifdef __GNUC__
// FIXME: use `[[noreturn]]` for all compilers when we move to C23
#define NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define NORETURN __declspec(noreturn)
#else
#define NORETURN /* nothing */
#endif

static inline NORETURN void graphviz_exit(int status) {
#ifdef _WIN32
  // workaround for https://gitlab.com/graphviz/graphviz/-/issues/2178
  fflush(stdout);
  fflush(stderr);
#ifdef __cplusplus
  std::cout.flush();
  std::cerr.flush();
#endif
#endif
  exit(status);
}

#undef NORETURN

#ifdef __cplusplus
}
#endif
