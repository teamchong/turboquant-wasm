/// @file
/// @brief `qsort` with carried along context
/// @ingroup cgraph_utils
///
/// The non-standard `qsort_r`, Windows’ `qsort_s`, and C11’s `qsort_s` provide
/// a `qsort` alternative with an extra state parameter. Unfortunately none of
/// these are easily usable portably in Graphviz. This header implements an
/// alternative that hopefully is.

#pragma once

#include <assert.h>
#include <stdlib.h>

static _Thread_local int (*gv_sort_compar)(const void *, const void *, void *);
static _Thread_local void *gv_sort_arg;

static inline int gv_sort_compar_wrapper(const void *a, const void *b) {
  assert(gv_sort_compar != NULL && "no comparator set in gv_sort");
  return gv_sort_compar(a, b, gv_sort_arg);
}

/// `qsort` with an extra state parameter, ala `qsort_r`
static inline void gv_sort(void *base, size_t nmemb, size_t size,
                           int (*compar)(const void *, const void *, void *),
                           void *arg) {
  assert(gv_sort_compar == NULL && gv_sort_arg == NULL &&
         "unsupported recursive call to gv_sort");

  gv_sort_compar = compar;
  gv_sort_arg = arg;

  if (nmemb > 1) {
    qsort(base, nmemb, size, gv_sort_compar_wrapper);
  }

  gv_sort_compar = NULL;
  gv_sort_arg = NULL;
}
