/// @file
/// @ingroup cgraph_utils
#pragma once

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util/alloc.h>
#include <util/exit.h>

#ifdef __GNUC__
#define LIST_UNUSED __attribute__((unused))
#else
#define LIST_UNUSED /* nothing */
#endif

/** create a new list type and its associated member functions
 *
 * \param name Type name to give the list container
 * \param type Type of the elements the list will store
 */
#define DEFINE_LIST(name, type) DEFINE_LIST_WITH_DTOR(name, type, name##_noop_)

/** \p DEFINE_LIST but with a custom element destructor
 *
 * \param name Type name to give the list container
 * \param type Type of the elements the list will store
 * \param dtor Destructor to be called on elements being released
 */
#define DEFINE_LIST_WITH_DTOR(name, type, dtor)                                                                                              \
                                                                                                                                             \
  /** list container                                                                                                                         \
   *                                                                                                                                         \
   * All members of this type are considered private. They should only be                                                                    \
   * accessed through the functions below.                                                                                                   \
   */                                                                                                                                        \
  typedef struct {                                                                                                                           \
    type *base; /* start of the allocation for backing memory */                                                                             \
    /* (base == NULL && capacity == 0) || (base != NULL && capacity > 0) */                                                                  \
    size_t head; /* index of the first element */                                                                                            \
    /* (capacity == 0 && head == 0)  || (capacity > 0 && head < capacity) */                                                                 \
    size_t size; /* number of elements in the list */                                                                                        \
    /* size <= capacity */                                                                                                                   \
    size_t capacity; /* available storage slots */                                                                                           \
  } name##_t;                                                                                                                                \
                                                                                                                                             \
  /* default “do nothing” destructor */                                                                                                  \
  static inline LIST_UNUSED void name##_noop_(type item) { (void)item; }                                                                     \
                                                                                                                                             \
  /** get the number of elements in a list */                                                                                                \
  static inline LIST_UNUSED size_t name##_size(const name##_t *list) {                                                                       \
    assert(list != NULL);                                                                                                                    \
    return list->size;                                                                                                                       \
  }                                                                                                                                          \
                                                                                                                                             \
  /** does this list contain no elements? */                                                                                                 \
  static inline LIST_UNUSED bool name##_is_empty(const name##_t *list) {                                                                     \
    assert(list != NULL);                                                                                                                    \
    return name##_size(list) == 0;                                                                                                           \
  }                                                                                                                                          \
                                                                                                                                             \
  static inline int name##_try_append(name##_t *list, type item) {                                                                           \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    /* do we need to expand the backing storage? */                                                                                          \
    if (list->size == list->capacity) {                                                                                                      \
      const size_t c = list->capacity == 0 ? 1 : (list->capacity * 2);                                                                       \
                                                                                                                                             \
      /* will the calculation of the new size overflow? */                                                                                   \
      if (SIZE_MAX / c < sizeof(type)) {                                                                                                     \
        return ERANGE;                                                                                                                       \
      }                                                                                                                                      \
                                                                                                                                             \
      type *base = (type *)realloc(list->base, c * sizeof(type));                                                                            \
      if (base == NULL) {                                                                                                                    \
        return ENOMEM;                                                                                                                       \
      }                                                                                                                                      \
                                                                                                                                             \
      /* zero the new memory */                                                                                                              \
      memset((char *)base + list->capacity * sizeof(type), 0,                                                                                \
             (c - list->capacity) * sizeof(type));                                                                                           \
                                                                                                                                             \
      /* Do we need to shuffle the prefix upwards? E.g. */                                                                                   \
      /*                                                */                                                                                   \
      /*        ┌───┬───┬───┬───┐                       */                                                 \
      /*   old: │ 3 │ 4 │ 1 │ 2 │                       */                                                                         \
      /*        └───┴───┴─┼─┴─┼─┘                       */                                                 \
      /*                  │   └───────────────┐         */                                               \
      /*                  └───────────────┐   │         */                                               \
      /*                                  ▼   ▼         */                                                                               \
      /*        ┌───┬───┬───┬───┬───┬───┬───┬───┐       */                 \
      /*   new: │ 3 │ 4 │   │   │   │   │ 1 │ 2 │       */                                                                 \
      /*        └───┴───┴───┴───┴───┴───┴───┴───┘       */                 \
      if (list->head + list->size > list->capacity) {                                                                                        \
        const size_t prefix = list->capacity - list->head;                                                                                   \
        const size_t new_head = c - prefix;                                                                                                  \
        memmove(&base[new_head], &base[list->head], prefix * sizeof(type));                                                                  \
        list->head = new_head;                                                                                                               \
      }                                                                                                                                      \
                                                                                                                                             \
      list->base = base;                                                                                                                     \
      list->capacity = c;                                                                                                                    \
    }                                                                                                                                        \
                                                                                                                                             \
    list->base[(list->head + list->size) % list->capacity] = item;                                                                           \
    ++list->size;                                                                                                                            \
                                                                                                                                             \
    return 0;                                                                                                                                \
  }                                                                                                                                          \
                                                                                                                                             \
  static inline void name##_append(name##_t *list, type item) {                                                                              \
    int rc = name##_try_append(list, item);                                                                                                  \
    if (rc != 0) {                                                                                                                           \
      fprintf(stderr, "realloc failed: %s\n", strerror(rc));                                                                                 \
      graphviz_exit(EXIT_FAILURE);                                                                                                           \
    }                                                                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** retrieve an element from a list                                                                                                        \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \param index Element index to get                                                                                                       \
   * \return Element at the given index                                                                                                      \
   */                                                                                                                                        \
  static inline type name##_get(const name##_t *list, size_t index) {                                                                        \
    assert(list != NULL);                                                                                                                    \
    assert(index < list->size && "index out of bounds");                                                                                     \
    return list->base[(list->head + index) % list->capacity];                                                                                \
  }                                                                                                                                          \
                                                                                                                                             \
  /** access an element in a list for the purpose of modification                                                                            \
   *                                                                                                                                         \
   * Because this acquires an internal pointer into the list structure, `get`                                                                \
   * and `set` should be preferred over this function. `get` and `set` are                                                                   \
   * easier to reason about. In particular, the pointer returned by this                                                                     \
   * function is invalidated by any list operation that may reallocate the                                                                   \
   * backing storage (e.g. `shrink_to_fit`).                                                                                                 \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \param index Element to get a pointer to                                                                                                \
   * \return Pointer to the requested element                                                                                                \
   */                                                                                                                                        \
  static inline type *name##_at(name##_t *list, size_t index) {                                                                              \
    assert(list != NULL);                                                                                                                    \
    assert(index < list->size && "index out of bounds");                                                                                     \
    return &list->base[(list->head + index) % list->capacity];                                                                               \
  }                                                                                                                                          \
                                                                                                                                             \
  /** get a handle to the first element */                                                                                                   \
  static inline LIST_UNUSED type *name##_front(name##_t *list) {                                                                             \
    assert(list != NULL);                                                                                                                    \
    assert(!name##_is_empty(list));                                                                                                          \
    return name##_at(list, 0);                                                                                                               \
  }                                                                                                                                          \
                                                                                                                                             \
  /** get a handle to the last element */                                                                                                    \
  static inline LIST_UNUSED type *name##_back(name##_t *list) {                                                                              \
    assert(list != NULL);                                                                                                                    \
    assert(!name##_is_empty(list));                                                                                                          \
    return name##_at(list, name##_size(list) - 1);                                                                                           \
  }                                                                                                                                          \
                                                                                                                                             \
  /** assign to an element in a list                                                                                                         \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \param index Element to assign to                                                                                                       \
   * \param item Value to assign                                                                                                             \
   */                                                                                                                                        \
  static inline void name##_set(name##_t *list, size_t index, type item) {                                                                   \
    assert(list != NULL);                                                                                                                    \
    assert(index < name##_size(list) && "index out of bounds");                                                                              \
    type *target = name##_at(list, index);                                                                                                   \
    dtor(*target);                                                                                                                           \
    *target = item;                                                                                                                          \
  }                                                                                                                                          \
                                                                                                                                             \
  /** remove an element from a list                                                                                                          \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \param item Value of element to remove                                                                                                  \
   */                                                                                                                                        \
  static inline LIST_UNUSED void name##_remove(name##_t *list,                                                                               \
                                               const type item) {                                                                            \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    for (size_t i = 0; i < list->size; ++i) {                                                                                                \
      /* is this the element we are looking for? */                                                                                          \
      type *candidate = name##_at(list, i);                                                                                                  \
      if (memcmp(candidate, &item, sizeof(type)) == 0) {                                                                                     \
                                                                                                                                             \
        /* destroy the element we are about to remove */                                                                                     \
        dtor(*candidate);                                                                                                                    \
                                                                                                                                             \
        /* shrink the list */                                                                                                                \
        for (size_t j = i + 1; j < list->size; ++j) {                                                                                        \
          type *replacement = name##_at(list, j);                                                                                            \
          *candidate = *replacement;                                                                                                         \
          candidate = replacement;                                                                                                           \
        }                                                                                                                                    \
        --list->size;                                                                                                                        \
        return;                                                                                                                              \
      }                                                                                                                                      \
    }                                                                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** remove all elements from a list */                                                                                                     \
  static inline LIST_UNUSED void name##_clear(name##_t *list) {                                                                              \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    for (size_t i = 0; i < list->size; ++i) {                                                                                                \
      dtor(name##_get(list, i));                                                                                                             \
    }                                                                                                                                        \
                                                                                                                                             \
    list->size = 0;                                                                                                                          \
                                                                                                                                             \
    /* opportunistically re-sync the list */                                                                                                 \
    list->head = 0;                                                                                                                          \
  }                                                                                                                                          \
                                                                                                                                             \
  /** ensure the list can fit a given number of items without reallocation                                                                   \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \param capacity Number of items the list should be able to contain                                                                      \
   */                                                                                                                                        \
  static inline LIST_UNUSED void name##_reserve(name##_t *list,                                                                              \
                                                size_t capacity) {                                                                           \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    /* if we can already fit enough items, nothing to do */                                                                                  \
    if (list->capacity >= capacity) {                                                                                                        \
      return;                                                                                                                                \
    }                                                                                                                                        \
                                                                                                                                             \
    list->base = (type *)gv_recalloc(list->base, list->capacity, capacity,                                                                   \
                                     sizeof(type));                                                                                          \
                                                                                                                                             \
    /* Do we need to shuffle the prefix upwards? E.g. */                                                                                     \
    /*                                                */                                                                                     \
    /*        ┌───┬───┬───┬───┐                       */                                                   \
    /*   old: │ 3 │ 4 │ 1 │ 2 │                       */                                                                           \
    /*        └───┴───┴─┼─┴─┼─┘                       */                                                   \
    /*                  │   └───────────────┐         */                                                 \
    /*                  └───────────────┐   │         */                                                 \
    /*                                  ▼   ▼         */                                                                                 \
    /*        ┌───┬───┬───┬───┬───┬───┬───┬───┐       */                   \
    /*   new: │ 3 │ 4 │   │   │   │   │ 1 │ 2 │       */                                                                   \
    /*        └───┴───┴───┴───┴───┴───┴───┴───┘       */                   \
    if (list->head + list->size > list->capacity) {                                                                                          \
      const size_t prefix = list->capacity - list->head;                                                                                     \
      const size_t new_head = capacity - prefix;                                                                                             \
      memmove(&list->base[new_head], &list->base[list->head],                                                                                \
              prefix * sizeof(type));                                                                                                        \
      list->head = new_head;                                                                                                                 \
    }                                                                                                                                        \
                                                                                                                                             \
    list->capacity = capacity;                                                                                                               \
  }                                                                                                                                          \
                                                                                                                                             \
  /** shrink or grow the list to the given size                                                                                              \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \param size New size of the list                                                                                                        \
   * \param value Default to assign to any new elements                                                                                      \
   */                                                                                                                                        \
  static inline LIST_UNUSED void name##_resize(name##_t *list, size_t size,                                                                  \
                                               type value) {                                                                                 \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    if (list->size < size) {                                                                                                                 \
      /* we are expanding the list */                                                                                                        \
      while (list->size < size) {                                                                                                            \
        name##_append(list, value);                                                                                                          \
      }                                                                                                                                      \
    } else if (list->size > size) {                                                                                                          \
      /* we are shrinking the list */                                                                                                        \
      while (list->size > size) {                                                                                                            \
        dtor(name##_get(list, list->size - 1));                                                                                              \
        --list->size;                                                                                                                        \
      }                                                                                                                                      \
    }                                                                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** is the given element in the list? */                                                                                                   \
  static inline LIST_UNUSED bool name##_contains(                                                                                            \
      const name##_t *haystack, const type needle,                                                                                           \
      bool (*eq)(const type a, const type b)) {                                                                                              \
    assert(haystack != NULL);                                                                                                                \
    assert(eq != NULL);                                                                                                                      \
                                                                                                                                             \
    for (size_t i = 0; i < name##_size(haystack); ++i) {                                                                                     \
      if (eq(name##_get(haystack, i), needle)) {                                                                                             \
        return true;                                                                                                                         \
      }                                                                                                                                      \
    }                                                                                                                                        \
    return false;                                                                                                                            \
  }                                                                                                                                          \
                                                                                                                                             \
  /** replicate a list */                                                                                                                    \
  static inline LIST_UNUSED name##_t name##_copy(const name##_t *source) {                                                                   \
    assert(source != NULL);                                                                                                                  \
                                                                                                                                             \
    name##_t destination = {(type *)gv_calloc(source->capacity, sizeof(type)),                                                               \
                            0, 0, source->capacity};                                                                                         \
    for (size_t i = 0; i < source->size; ++i) {                                                                                              \
      name##_append(&destination, name##_get(source, i));                                                                                    \
    }                                                                                                                                        \
    return destination;                                                                                                                      \
  }                                                                                                                                          \
                                                                                                                                             \
  /** does the list not wrap past its end?                              */                                                                   \
  /**                                                                   */                                                                   \
  /** This checks whether the list is discontiguous in how its elements */                                                                   \
  /** appear in memory:                                                 */                                                                   \
  /**                                                                   */                                                                   \
  /**                         ┌───┬───┬───┬───┬───┬───┬───┬───┐         */ \
  /**   a contiguous list:    │   │   │ w │ x │ y │ z │   │   │         */                                                 \
  /**                         └───┴───┴───┴───┴───┴───┴───┴───┘         */ \
  /**                                   0   1   2   3                   */                                                                   \
  /**                                                                   */                                                                   \
  /**                         ┌───┬───┬───┬───┬───┬───┬───┬───┐         */ \
  /**   a discontiguous list: │ y │ z │   │   │   │   │ x │ y │         */                                                 \
  /**                         └───┴───┴───┴───┴───┴───┴───┴───┘         */ \
  /**                           2   3                   0   1           */                                                                   \
  static inline LIST_UNUSED bool name##_is_contiguous(const name##_t *list) {                                                                \
    assert(list != NULL);                                                                                                                    \
    return list->head + list->size <= list->capacity;                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** shuffle the populated contents to reset `head` to 0 */                                                                                 \
  static inline void name##_sync(name##_t *list) {                                                                                           \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    /* Shuffle the list 1-1 until it is aligned. This is not efficient, but */                                                               \
    /* we assume this is a relatively rare operation. */                                                                                     \
    while (list->head != 0) {                                                                                                                \
      /* rotate the list leftwards by 1 */                                                                                                   \
      assert(list->capacity > 0);                                                                                                            \
      type replacement = list->base[0];                                                                                                      \
      for (size_t i = list->capacity - 1; i != SIZE_MAX; --i) {                                                                              \
        type temp = list->base[i];                                                                                                           \
        list->base[i] = replacement;                                                                                                         \
        replacement = temp;                                                                                                                  \
      }                                                                                                                                      \
      --list->head;                                                                                                                          \
    }                                                                                                                                        \
                                                                                                                                             \
    /* synchronization should have ensured the list no longer wraps */                                                                       \
    assert(name##_is_contiguous(list));                                                                                                      \
  }                                                                                                                                          \
                                                                                                                                             \
  /** sort the list using the given comparator */                                                                                            \
  static inline LIST_UNUSED void name##_sort(                                                                                                \
      name##_t *list, int (*cmp)(const type *a, const type *b)) {                                                                            \
    assert(list != NULL);                                                                                                                    \
    assert(cmp != NULL);                                                                                                                     \
                                                                                                                                             \
    name##_sync(list);                                                                                                                       \
                                                                                                                                             \
    int (*compar)(const void *, const void *) =                                                                                              \
        (int (*)(const void *, const void *))cmp;                                                                                            \
    if (list->size > 0) {                                                                                                                    \
      qsort(list->base, list->size, sizeof(type), compar);                                                                                   \
    }                                                                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** flip the order of elements in the list */                                                                                              \
  static inline LIST_UNUSED void name##_reverse(name##_t *list) {                                                                            \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    for (size_t i = 0; i < name##_size(list) / 2; ++i) {                                                                                     \
      type const temp1 = name##_get(list, i);                                                                                                \
      type const temp2 = name##_get(list, name##_size(list) - i - 1);                                                                        \
      name##_set(list, i, temp2);                                                                                                            \
      name##_set(list, name##_size(list) - i - 1, temp1);                                                                                    \
    }                                                                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** deallocate unused backing storage, shrinking capacity to size */                                                                       \
  static inline LIST_UNUSED void name##_shrink_to_fit(name##_t *list) {                                                                      \
    assert(list != NULL);                                                                                                                    \
                                                                                                                                             \
    name##_sync(list);                                                                                                                       \
                                                                                                                                             \
    if (list->capacity > list->size) {                                                                                                       \
      list->base = (type *)gv_recalloc(list->base, list->capacity, list->size,                                                               \
                                       sizeof(type));                                                                                        \
      list->capacity = list->size;                                                                                                           \
    }                                                                                                                                        \
  }                                                                                                                                          \
                                                                                                                                             \
  /** free resources associated with a list */                                                                                               \
  static inline LIST_UNUSED void name##_free(name##_t *list) {                                                                               \
    assert(list != NULL);                                                                                                                    \
    name##_clear(list);                                                                                                                      \
    free(list->base);                                                                                                                        \
    memset(list, 0, sizeof(*list));                                                                                                          \
  }                                                                                                                                          \
                                                                                                                                             \
  /** alias for append */                                                                                                                    \
  static inline LIST_UNUSED void name##_push_back(name##_t *list,                                                                            \
                                                  type value) {                                                                              \
    name##_append(list, value);                                                                                                              \
  }                                                                                                                                          \
                                                                                                                                             \
  /** remove and return first element */                                                                                                     \
  static inline LIST_UNUSED type name##_pop_front(name##_t *list) {                                                                          \
    assert(list != NULL);                                                                                                                    \
    assert(list->size > 0);                                                                                                                  \
                                                                                                                                             \
    type value = name##_get(list, 0);                                                                                                        \
                                                                                                                                             \
    /* do not call `dtor` because we are transferring ownership of the removed                                                               \
     * element to the caller                                                                                                                 \
     */                                                                                                                                      \
    list->head = (list->head + 1) % list->capacity;                                                                                          \
    --list->size;                                                                                                                            \
                                                                                                                                             \
    return value;                                                                                                                            \
  }                                                                                                                                          \
                                                                                                                                             \
  /** remove and return last element */                                                                                                      \
  static inline LIST_UNUSED type name##_pop_back(name##_t *list) {                                                                           \
    assert(list != NULL);                                                                                                                    \
    assert(list->size > 0);                                                                                                                  \
                                                                                                                                             \
    type value = name##_get(list, list->size - 1);                                                                                           \
                                                                                                                                             \
    /* do not call `dtor` because we are transferring ownership of the removed                                                               \
     * element to the caller                                                                                                                 \
     */                                                                                                                                      \
    --list->size;                                                                                                                            \
                                                                                                                                             \
    return value;                                                                                                                            \
  }                                                                                                                                          \
                                                                                                                                             \
  /** create a new list from a bare array and element count                                                                                  \
   *                                                                                                                                         \
   * This can be useful when receiving data from a caller who does not use                                                                   \
   * this API, but the callee wants to. Note that the backing data for the                                                                   \
   * array must have been heap-allocated.                                                                                                    \
   *                                                                                                                                         \
   * \param data Array of existing elements                                                                                                  \
   * \param size Number of elements pointed to by `data`                                                                                     \
   * \return A managed list containing the provided elements                                                                                 \
   */                                                                                                                                        \
  static inline LIST_UNUSED name##_t name##_attach(type *data, size_t size) {                                                                \
    assert(data != NULL || size == 0);                                                                                                       \
    name##_t list = {data, 0, size, size};                                                                                                   \
    return list;                                                                                                                             \
  }                                                                                                                                          \
                                                                                                                                             \
  /** transform a managed list into a bare array                                                                                             \
   *                                                                                                                                         \
   * This can be useful when needing to pass data to a callee who does not                                                                   \
   * use this API. The managed list is emptied and left in a state where it                                                                  \
   * can be reused for other purposes.                                                                                                       \
   *                                                                                                                                         \
   * \param list List to operate on                                                                                                          \
   * \return A pointer to an array of the `list->size` elements                                                                              \
   */                                                                                                                                        \
  static inline LIST_UNUSED type *name##_detach(name##_t *list) {                                                                            \
    assert(list != NULL);                                                                                                                    \
    name##_sync(list);                                                                                                                       \
    type *data = list->base;                                                                                                                 \
    memset(list, 0, sizeof(*list));                                                                                                          \
    return data;                                                                                                                             \
  }
