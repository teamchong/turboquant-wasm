/**
 * @file
 * @ingroup cgraph_core
 */

#pragma once

#include <stddef.h>

typedef struct {
  const char *data;
  size_t len;
  size_t cur;
} rdr_t;
