/// @file
/// @brief Implementation of the arena.h API

#ifndef NO_CONFIG // defined by test_arena.c
#include "config.h"
#endif

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <util/alloc.h>
#include <util/arena.h>
#include <util/asan.h>
#include <util/unused.h>

/// a block of backing memory
///
/// Note that this is only the header metadata. When one of these structures is
/// allocated, it is followed by raw bytes used for allocations themselves.
struct arena_chunk {
  arena_chunk_t *previous; ///< previous chunk that was in use
};

/// `popcount(value) == 1`?
static UNUSED bool is_power_of_2(size_t value) {
  if (value == 0) {
    return false;
  }
  while ((value & 1) != 1) {
    value >>= 1;
  }
  return value == 1;
}

/// get some more memory from the system allocator
///
/// @param arena Arena to install the new memory into
/// @param req_alignment Alignment request that led to this call
/// @param req_size Size request that led to this call
static void more_core(arena_t *arena, size_t req_alignment, size_t req_size) {
  assert(arena != NULL);
  assert(req_alignment != 0);

  // A default number of bytes to allocate in a chunk. The aim is for the
  // resulting allocation to be a multiple of the system page size, to encourage
  // the system to give us entire pages. If this does not work out, it is not
  // critical.
  enum { DEFAULT_CHUNK = 16384 - sizeof(arena_chunk_t) };

  size_t chunk_size = DEFAULT_CHUNK;
  // override the default size if we are allocating something too large
  if (chunk_size < req_size + req_alignment - 1) {
    chunk_size = req_size + req_alignment - 1;
  }

  arena_chunk_t *const more = gv_alloc(sizeof(arena_chunk_t) + chunk_size);

  // mark the newly available space as unused
  ASAN_POISON((char *)more + sizeof(arena_chunk_t), chunk_size);

  // install the new chunk
  more->previous = arena->source;
  arena->source = more;
  arena->remaining = chunk_size;
}

/// allocate new dynamic memory
///
/// @param arena Arena to allocate from
/// @param alignment Requested alignment
/// @param size Requested size in bytes
/// @return Pointer to allocated memory or `NULL` on failure
static void *alloc(arena_t *arena, size_t alignment, size_t size) {
  assert(arena != NULL);
  assert(alignment != 0);
  assert(is_power_of_2(alignment));

  if (arena->remaining < size) {
    return NULL;
  }

  const uintptr_t base = (uintptr_t)arena->source + sizeof(arena_chunk_t);
  const uintptr_t limit = base + arena->remaining;

  // Allocate from the end of the chunk memory, for simplicity. E.g.:
  //
  //                actual allocation ┐     ┌ wasted space
  //                              ┌───┴───┬─┴─┐
  //          ┌────────┬──────────┬───────┬───┬───────────────┐
  //   chunk: │previous│ <free> … │       │   │ <allocated> … │
  //          └────────┴──────────┴───────┴───┴───────────────┘
  //                   ▲          ▲       ▲   ▲
  //              base ┘    start ┘       │   └ limit
  //                                      └ start + size
  const uintptr_t start = (limit - size) & ~(alignment - 1);

  if (start < base) {
    // we had enough bytes, but not enough aligned bytes
    return NULL;
  }

  arena->remaining -= limit - start;

  // Only unpoison the narrow allocation, not the full area we are carving off.
  // Repeating the diagram from above:
  //
  //                 unpoisoning this ┐     ┌ not unpoisoning this
  //                              ┌───┴───┬─┴─┐
  //          ┌────────┬──────────┬───────┬───┬───────────────┐
  //   chunk: │previous│ <free> … │       │   │ <allocated> … │
  //          └────────┴──────────┴───────┴───┴───────────────┘
  void *const ret = (void *)start;
  ASAN_UNPOISON(ret, size);

  return ret;
}

void *gv_arena_alloc(arena_t *arena, size_t alignment, size_t size) {
  assert(arena != NULL);

  if (size == 0) {
    return NULL;
  }

  void *ptr = alloc(arena, alignment, size);

  // if we failed, get some more memory and try again
  if (ptr == NULL) {
    more_core(arena, alignment, size);
    ptr = alloc(arena, alignment, size);
  }

  return ptr;
}

char *gv_arena_strdup(arena_t *arena, const char *s) {
  assert(arena != NULL);
  assert(s != NULL);

  const size_t len = strlen(s);
  char *const ret = gv_arena_alloc(arena, 1, len + 1);
  assert(ret != NULL);
  memcpy(ret, s, len);
  ret[len] = '\0';

  return ret;
}

void gv_arena_free(arena_t *arena, void *ptr, size_t size) {
  assert(arena != NULL);

  if (ptr == NULL) {
    return;
  }

  // teach ASan that this region should no longer be accessible
  ASAN_POISON(ptr, size);

  // we do not actually deallocate the memory, but leave it to be freed when the
  // arena is eventually reset
  (void)arena;
}

void gv_arena_reset(arena_t *arena) {
  assert(arena != NULL);

  while (arena->source != NULL) {
    arena_chunk_t *const previous = arena->source->previous;
    free(arena->source);
    arena->source = previous;
  }

  *arena = (arena_t){0};
}
