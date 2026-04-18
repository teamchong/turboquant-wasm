/// @file
/// @brief Helpers for dealing with `agstrcanon`

#pragma once

#include <stddef.h>
#include <string.h>
#include <util/alloc.h>

/// how many bytes are needed to canonicalize the given string
static inline size_t agstrcanon_bytes(const char *str) {
  return 2 * strlen(str) + 3;
}

/// get a buffer suitable for passing into `agstrcanon`
static inline char *agstrcanon_buffer(const char *str) {
  return gv_alloc(agstrcanon_bytes(str));
}
