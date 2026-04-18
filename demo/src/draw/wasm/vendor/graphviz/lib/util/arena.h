/// @file
/// @brief Region-based memory allocator
///
/// The API below is for interacting with a basic bump-pointer allocator.¹
///
/// ¹ https://en.wikipedia.org/wiki/Region-based_memory_management

#pragma once

#include <stdalign.h>
#include <stddef.h>
#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// a block of backing memory
///
/// This is a private implementation detail and should not be used outside of
/// arena.c
typedef struct arena_chunk arena_chunk_t;

/// an arena from which dynamic memory can be allocated
///
/// This structure is intended to be zero-initialized:
///
///   arena_t new_arena = {0};
///
/// All fields are considered private and should not be used outside of arena.c.
typedef struct {
  arena_chunk_t *source; ///< current chunk being allocated out of
  size_t remaining;      ///< number of free bytes remaining in `source`
} arena_t;

/// allocate new memory
///
/// The requested `alignment` must be a power of two. The returned memory is
/// zero initialized.
///
/// @param arena Arena to allocate from
/// @param alignment Required pointer alignment for the resulting allocation
/// @param size Number of bytes requested
/// @return Pointer to allocated memory
UTIL_API void *gv_arena_alloc(arena_t *arena, size_t alignment, size_t size);

/// allocate new memory for a typed value
///
/// This is shorthand for when you are allocating memory for a specific type.
/// The returned memory is zero initialized.
///
/// @param arena Arena to allocate from
/// @param type Type of value that will be later stored in this memory
/// @return Pointer to allocated memory
#define ARENA_NEW(arena, type)                                                 \
  gv_arena_alloc((arena), alignof(type), sizeof(type))

/// copy a string into new dynamic memory
///
/// @param arena Arena to allocate from
/// @param s Source string to copy
/// @return A copy of the string, hosted in arena-allocated memory
UTIL_API char *gv_arena_strdup(arena_t *arena, const char *s);

/// deallocate memory
///
/// @param arena Arena that was previously used to allocate this memory
/// @param ptr Pointer to memory to release
/// @param size Number of bytes of the original allocation
UTIL_API void gv_arena_free(arena_t *arena, void *ptr, size_t size);

/// deallocate all memory
///
/// Calling this restores the arena to an empty state from which it can be used
/// for more allocations.
///
/// @param arena Arena to reset
UTIL_API void gv_arena_reset(arena_t *arena);

#ifdef __cplusplus
}
#endif
