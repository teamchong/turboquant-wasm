#ifndef NO_CONFIG // defined by test_list.c
#include "config.h"
#endif

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util/alloc.h>
#include <util/asan.h>
#include <util/exit.h>
#include <util/gv_math.h>
#include <util/list-private.h>
#include <util/prisize_t.h>

static const void *slot_from_const_list(const list_t_ *list, size_t index,
                                        size_t stride) {
  assert(list != NULL);
  assert(list->base != NULL || index == 0 || stride == 0);

  const char *const base = list->base;
  return base + index * stride;
}

static void *slot_from_list(list_t_ *list, size_t index, size_t stride) {
  assert(list != NULL);
  assert(list->base != NULL || index == 0 || stride == 0);

  char *const base = list->base;
  return base + index * stride;
}

static const void *slot_from_const_base(const void *base, size_t index,
                                        size_t stride) {
  assert(base != NULL || index == 0 || stride == 0);

  const char *const b = base;
  return b + index * stride;
}

static void *slot_from_base(void *base, size_t index, size_t stride) {
  assert(base != NULL || index == 0 || stride == 0);

  char *const b = base;
  return b + index * stride;
}

#define INDEX_TO(origin, index, stride)                                        \
  (_Generic((origin),                                                          \
       const list_t_ *: slot_from_const_list,                                  \
       list_t_ *: slot_from_list,                                              \
       const void *: slot_from_const_base,                                     \
       void *: slot_from_base)((origin), (index), (stride)))

size_t gv_list_append_slot_(list_t_ *list, size_t item_size) {
  assert(list != NULL);

  // do we need to expand the backing storage?
  if (list->size == list->capacity) {
    const size_t c = list->capacity == 0 ? 1 : (list->capacity * 2);
    gv_list_reserve_(list, c, item_size);
  }

  assert(list->capacity > 0);
  assert(list->size < list->capacity);

  // append the new slot
  const size_t new_slot = (list->head + list->size) % list->capacity;
  void *const slot = INDEX_TO(list, new_slot, item_size);
  ASAN_UNPOISON(slot, item_size);
  ++list->size;

  return new_slot;
}

size_t gv_list_prepend_slot_(list_t_ *list, size_t item_size) {
  assert(list != NULL);

  // do we need to expand the backing storage?
  if (list->size == list->capacity) {
    const size_t c = list->capacity == 0 ? 1 : (list->capacity * 2);
    gv_list_reserve_(list, c, item_size);
  }

  assert(list->capacity > 0);
  assert(list->size < list->capacity);

  // prepend the new slot
  list->head = (list->head + (list->capacity - 1)) % list->capacity;
  void *const slot = INDEX_TO(list, list->head, item_size);
  ASAN_UNPOISON(slot, item_size);
  ++list->size;

  return list->head;
}

static int try_reserve(list_t_ *list, size_t capacity, size_t item_size) {
  assert(list != NULL);

  // if we can already fit enough items, nothing to do
  if (list->capacity >= capacity) {
    return 0;
  }

  // will the arithmetic below overflow?
  assert(capacity > 0);
  if (SIZE_MAX / capacity < item_size) {
    return EOVERFLOW;
  }

  void *const base = realloc(list->base, capacity * item_size);
  if (base == NULL && item_size > 0) {
    return ENOMEM;
  }

  // zero the new memory
  {
    void *const new = INDEX_TO(base, list->capacity, item_size);
    const size_t new_bytes = (capacity - list->capacity) * item_size;
    if (new_bytes > 0) { // `new_bytes` can be 0 if `item_size == 0`
      memset(new, 0, new_bytes);
    }

    // poison the new (conceptually unallocated) memory
    ASAN_POISON(new, new_bytes);
  }

  // Do we need to shuffle the prefix upwards? E.g.
  //
  //        ┌───┬───┬───┬───┐
  //   old: │ 3 │ 4 │ 1 │ 2 │
  //        └───┴───┴─┼─┴─┼─┘
  //                  │   └───────────────┐
  //                  └───────────────┐   │
  //                                  ▼   ▼
  //        ┌───┬───┬───┬───┬───┬───┬───┬───┐
  //   new: │ 3 │ 4 │   │   │   │   │ 1 │ 2 │
  //        └───┴───┴───┴───┴───┴───┴───┴───┘
  //          a   b   c   d   e   f   g   h
  if (list->head + list->size > list->capacity) {
    const size_t prefix = list->capacity - list->head;
    const size_t new_head = capacity - prefix;
    // unpoison target range, slots [g, h] in example
    void *const target = INDEX_TO(base, new_head, item_size);
    ASAN_UNPOISON(target, prefix * item_size);
    const void *const src = INDEX_TO(base, list->head, item_size);
    if (prefix * item_size > 0) {
      // `target` and `src` can be null when `item_size == 0`, and `memmove`
      // would then be Undefined Behavior
      memmove(target, src, prefix * item_size);
    }
    // (re-)poison new gap, slots [c, f] in example
    void *const gap_begin = INDEX_TO(base, list->size - prefix, item_size);
    ASAN_POISON(gap_begin, (list->capacity - list->size) * item_size);
    list->head = new_head;
  }

  list->base = base;
  list->capacity = capacity;
  return 0;
}

bool gv_list_try_append_(list_t_ *list, const void *item, size_t item_size) {
  assert(list != NULL);
  assert(item != NULL);

  // do we need to expand the backing storage?
  if (list->size == list->capacity) {
    do {
      // can we attempt doubling without integer overflow?
      if (SIZE_MAX / 2 >= list->capacity) {
        const size_t c = list->capacity == 0 ? 1 : (list->capacity * 2);
        if (try_reserve(list, c, item_size) == 0) {
          // success
          break;
        }
      }

      // try a more conservative expansion
      if (SIZE_MAX - 1 >= list->capacity) {
        if (try_reserve(list, list->capacity + 1, item_size) == 0) {
          // success
          break;
        }
      }

      // failed to expand the list
      return false;
    } while (0);
  }

  assert(list->size < list->capacity);

  // we can now append, knowing it will not require backing storage expansion
  const size_t new_slot = (list->head + list->size) % list->capacity;
  void *const slot = INDEX_TO(list, new_slot, item_size);
  ASAN_UNPOISON(slot, item_size);
  if (item_size > 0) {
    memcpy(slot, item, item_size);
  }
  ++list->size;

  return true;
}

size_t gv_list_get_(const list_t_ list, size_t index) {
  assert(index < list.size && "index out of bounds");
  return (list.head + index) % list.capacity;
}

size_t gv_list_find_(const list_t_ list, const void *needle, size_t item_size) {

  for (size_t i = 0; i < list.size; ++i) {
    const size_t slot = gv_list_get_(list, i);
    const void *candidate = INDEX_TO(&list, slot, item_size);
    if (item_size == 0 || memcmp(needle, candidate, item_size) == 0) {
      return i;
    }
  }

  return SIZE_MAX;
}

void gv_list_remove_(list_t_ *list, size_t index, size_t item_size) {
  assert(list != NULL);
  assert(index < list->size);

  // shrink the list
  for (size_t i = index + 1; i < list->size; ++i) {
    const size_t dst_slot = gv_list_get_(*list, i - 1);
    void *const dst = INDEX_TO(list, dst_slot, item_size);
    const size_t src_slot = gv_list_get_(*list, i);
    const void *const src = INDEX_TO(list, src_slot, item_size);
    if (item_size > 0) {
      memcpy(dst, src, item_size);
    }
  }
  const size_t truncated_slot = gv_list_get_(*list, list->size - 1);
  void *truncated = INDEX_TO(list, truncated_slot, item_size);
  ASAN_POISON(truncated, item_size);
  --list->size;
}

void gv_list_clear_(list_t_ *list, size_t item_size) {
  assert(list != NULL);

  for (size_t i = 0; i < list->size; ++i) {
    const size_t slot = gv_list_get_(*list, i);
    void *const to_poison = INDEX_TO(list, slot, item_size);
    ASAN_POISON(to_poison, item_size);
  }

  list->size = 0;

  // opportunistically re-sync the list
  list->head = 0;
}

void gv_list_reserve_(list_t_ *list, size_t capacity, size_t item_size) {
  const int err = try_reserve(list, capacity, item_size);
  if (err != 0) {
    fprintf(stderr,
            "failed to reserve %" PRISIZE_T " elements of size %" PRISIZE_T
            " bytes: %s\n",
            capacity, item_size, strerror(err));
    graphviz_exit(EXIT_FAILURE);
  }
}

bool gv_list_contains_(const list_t_ list, const void *needle,
                       size_t item_size) {
  return gv_list_find_(list, needle, item_size) != SIZE_MAX;
}

list_t_ gv_list_copy_(const list_t_ list, size_t item_size) {
  list_t_ ret = {.base = gv_calloc(list.capacity, item_size),
                 .capacity = list.capacity};

  // opportunistically create the new list synced
  for (size_t i = 0; i < list.size; ++i) {
    const size_t slot = gv_list_get_(list, i);
    const void *const src = INDEX_TO(&list, slot, item_size);
    void *const dst = INDEX_TO(&ret, ret.size, item_size);
    assert(ret.size < ret.capacity);
    if (item_size > 0) {
      memcpy(dst, src, item_size);
    }
    ++ret.size;
  }

  // mark the remainder of the allocated space as inaccessible
  void *const to_poison = INDEX_TO(&ret, ret.size, item_size);
  const size_t to_poison_len = (ret.capacity - ret.size) * item_size;
  ASAN_POISON(to_poison, to_poison_len);

  return ret;
}

bool gv_list_is_contiguous_(const list_t_ list) {
  return list.head + list.size <= list.capacity;
}

void gv_list_sync_(list_t_ *list, size_t item_size) {
  assert(list != NULL);

  // Allow unrestricted access. The shuffle below accesses both allocated
  // and unallocated elements, so just let it read and write everything.
  ASAN_UNPOISON(list->base, list->capacity * item_size);

  // Shuffle the list 1-1 until it is aligned. This is not efficient, but
  // we assume this is a relatively rare operation.
  while (list->head != 0) {
    // rotate the list leftwards by 1
    assert(list->capacity > 0);
    // shuffle byte-by-byte to avoid dynamic allocation
    for (size_t i = 0; i < item_size; ++i) {
      uint8_t lowest;
      memcpy(&lowest, list->base, sizeof(lowest));
      const size_t remainder = list->capacity * item_size - sizeof(lowest);
      memmove(list->base, (char *)list->base + sizeof(lowest), remainder);
      memcpy((char *)list->base + remainder, &lowest, sizeof(lowest));
    }
    --list->head;
  }

  /* synchronization should have ensured the list no longer wraps */
  assert(gv_list_is_contiguous_(*list));

  /* re-establish access restrictions */
  void *end = INDEX_TO(list, list->size, item_size);
  ASAN_POISON(end, (list->capacity - list->size) * item_size);
}

void gv_list_sort_(list_t_ *list, int (*cmp)(const void *, const void *),
                   size_t item_size) {
  assert(list != NULL);
  assert(cmp != NULL);

  gv_list_sync_(list, item_size);

  if (list->size > 0 && item_size > 0) {
    qsort(list->base, list->size, item_size, cmp);
  }
}

static void exchange(void *a, void *b, size_t size) {
  assert(a != NULL);
  assert(b != NULL);

  // do a byte-by-byte swap of the two objects
  char *x = a;
  char *y = b;
  for (size_t i = 0; i < size; ++i) {
    SWAP(&x[i], &y[i]);
  }
}

void gv_list_reverse_(list_t_ *list, size_t item_size) {
  assert(list != NULL);

  // move from the outside inwards, swapping elements
  for (size_t i = 0; i < list->size / 2; ++i) {
    const size_t a = gv_list_get_(*list, i);
    const size_t b = gv_list_get_(*list, list->size - i - 1);
    void *const x = INDEX_TO(list, a, item_size);
    void *const y = INDEX_TO(list, b, item_size);
    exchange(x, y, item_size);
  }
}

void gv_list_shrink_to_fit_(list_t_ *list, size_t item_size) {
  assert(list != NULL);

  gv_list_sync_(list, item_size);

  if (list->capacity > list->size) {
    list->base = gv_recalloc(list->base, list->capacity, list->size, item_size);
    list->capacity = list->size;
  }
}

void gv_list_free_(list_t_ *list) {
  assert(list != NULL);
  free(list->base);
  *list = (list_t_){0};
}

void gv_list_pop_front_(list_t_ *list, void *into, size_t item_size) {
  assert(list != NULL);
  assert(list->size > 0);
  assert(into != NULL);

  // find and pop the first slot
  const size_t slot = gv_list_get_(*list, 0);
  void *const to_pop = INDEX_TO(list, slot, item_size);
  if (item_size > 0) {
    memcpy(into, to_pop, item_size);
  }
  ASAN_POISON(to_pop, item_size);
  list->head = (list->head + 1) % list->capacity;
  --list->size;
}

void gv_list_pop_back_(list_t_ *list, void *into, size_t item_size) {
  assert(list != NULL);
  assert(list->size > 0);
  assert(into != NULL);

  // find and pop last slot
  const size_t slot = gv_list_get_(*list, list->size - 1);
  void *const to_pop = INDEX_TO(list, slot, item_size);
  if (item_size > 0) {
    memcpy(into, to_pop, item_size);
  }
  ASAN_POISON(to_pop, item_size);
  --list->size;
}

void gv_list_detach_(list_t_ *list, void *datap, size_t *sizep,
                     size_t item_size) {
  assert(list != NULL);
  assert(datap != NULL);

  gv_list_sync_(list, item_size);
  memcpy(datap, &list->base, sizeof(void *));
  if (sizep != NULL) {
    *sizep = list->size;
  }

  *list = (list_t_){0};
}
