/// @file
/// @ingroup cgraph_utils
#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/// does the string \p s begin with the string \p prefix?
static inline bool startswith(const char *s, const char *prefix) {
  assert(s != NULL);
  assert(prefix != NULL);

  return strncmp(s, prefix, strlen(prefix)) == 0;
}
