/// @file
/// @brief type-generic dynamically expanding list
/// @ingroup cgraph_utils
///
/// The code in this header is structured as a public API made up of macros that
/// do as little as possible before handing off to internal functions.
///
/// If you are familiar with the concept of a dynamically expanding array like
/// C++’s `std::vector`, the only things that will likely throw you off are:
///   1. The genericity of `LIST` is implemented through a `union` that overlaps
///      the `base` member of a `list_t_` (a generic list core) with a typed
///      pointer. This design involves the public macros dealing in the
///      `LIST(foo)` type while the private functions deal in `list_t_`s.
///   2. The start of the list is not always index 0, in order to more
///      efficiently implement a queue. See the diagram and discussion in
///      `try_reserve` to better understand this.
///
/// Some general terminology you may see in function/macro names:
///   base – the start of the underlying heap allocation backing a list
///   dtor – destructor
///   head – slot index of the start of a list
///   item – a list element
///   slot – an item-sized space in the list, offset recorded from base
///
/// Some unorthodox idioms you may see used in this file:
///   • `(void)(foo == bar)` as a way to force the compiler to type-check that
///     `foo` and `bar` have compatible types. This is the best we can do for
///     pointer compatibility checks without `typeof`.
///   • `(void)(sizeof(foo) == sizeof(bar) ? (void)0 : (void)(…,abort())` as an
///     even weaker version of the above, for when we need to delay a check to
///     runtime instead of compile-time. This is a very unreliable check for
///     `foo` and `bar` being the same type, so should be avoided wherever
///     possible.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <util/list-private.h>

#ifdef __cplusplus
extern "C" {
#endif

static_assert(
    offsetof(list_t_, base) == 0,
    "LIST(<type>).base and LIST(<type>).impl.base will not alias each other");

/// list data structure
///
/// Typical usage:
///
///   LIST(int) my_int_list = {0};
#define LIST(type)                                                             \
  struct {                                                                     \
    union {                                                                    \
      type *base;                                                              \
      list_t_ impl;                                                            \
    }; /**< backing storage */                                                 \
    void (*dtor)(type); /**< optional destructor */                            \
    type scratch;       /**< temporary space for storing off-list items */     \
  }

/// sentinel value to indicate you want `free` to be used as a list destructor
///
/// Sample usage:
///
///   LIST(char *) my_strings = {.dtor = LIST_DTOR_FREE};
#define LIST_DTOR_FREE ((void *)1)

/// get the number of elements in a list
///
/// You can think of this macro as having the C type:
///
///   size_t LIST_SIZE(const LIST(<type>) *list);
///
/// @param list List to inspect
/// @return Size of the list
#define LIST_SIZE(list) gv_list_size_((list)->impl)

/// does this list contain no elements?
///
/// You can think of this macro as having the C type:
///
///   bool LIST_IS_EMPTY(const LIST(<type>) *list);
///
/// @param list List to inspect
/// @return True if the list is empty
#define LIST_IS_EMPTY(list) (LIST_SIZE(list) == 0)

/// try to append a new item to a list
///
/// You can think of this macro as having the C type:
///
///   bool LIST_TRY_APPEND(LIST(<type>) *list, <type> item);
///
/// @param list List to operate on
/// @param item Item to append
/// @return True if the append succeeded
#define LIST_TRY_APPEND(list, item)                                            \
  gv_list_try_append_(&(list)->impl,                                           \
                      ((list)->scratch = (item), &(list)->scratch),            \
                      sizeof((list)->base[0]))

/// add an item to the end of a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_APPEND(LIST(<type>) *list, <type> item);
///
/// This macro succeeds or exits on out-of-memory; it never return failure.
///
/// Note, in contrast to `LIST_TRY_APPEND`, `gv_list_append_slot_` and the write
/// to `(list)->base[…]` are in separate statements because the
/// `gv_list_append_slot_` call here _can_ alter `(list)->base`.
///
/// @param list List to operate on
/// @param item Element to append
#define LIST_APPEND(list, item)                                                \
  do {                                                                         \
    (list)->scratch = (item);                                                  \
    const size_t slot_ =                                                       \
        gv_list_append_slot_(&(list)->impl, sizeof((list)->base[0]));          \
    (list)->base[slot_] = (list)->scratch;                                     \
  } while (0)

/// add an item to the beginning of a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_PREPEND(LIST(<type>) *list, <type> item);
///
/// This macro succeeds or exits on out-of-memory; it never return failure.
///
/// @param list List to operate on
/// @param item Element to prepend
#define LIST_PREPEND(list, item)                                               \
  do {                                                                         \
    (list)->scratch = (item);                                                  \
    const size_t slot_ =                                                       \
        gv_list_prepend_slot_(&(list)->impl, sizeof((list)->base[0]));         \
    (list)->base[slot_] = (list)->scratch;                                     \
  } while (0)

/// retrieve an item from a list
///
/// You can think of this macro as having the C type:
///
///   <type> LIST_GET(const LIST(<type>) *list, size_t index);
///
/// @param list List to operate on
/// @param index Item index to get
/// @return Item at the given index
#define LIST_GET(list, index)                                                  \
  ((list)->base[gv_list_get_((list)->impl, (index))])

/// retrieve a pointer to an item from a list
///
/// You can think of this macro as having one of the C types:
///
///   <type> *LIST_AT(LIST(<type>) *list, size_t index);
///   const <type> *LIST_AT(const LIST(<type>) *list, size_t index);
///
/// @param list List to operate on
/// @param index Item index to get
/// @return Pointer to item at the given index
#define LIST_AT(list, index)                                                   \
  (&(list)->base[gv_list_get_((list)->impl, (index))])

/// retrieve a pointer to the first item in a list
///
/// You can think of this macro as having one of the C types:
///
///   <type> *LIST_FRONT(LIST(<type>) *list);
///   const <type> *LIST_FRONT(const LIST(<type>) *list);
///
/// @param list List to operate on
/// @return Pointer to the first item in the list
#define LIST_FRONT(list) LIST_AT((list), 0)

/// retrieve a pointer to the last item in a list
///
/// You can think of this macro as having one of the C types:
///
///   <type> *LIST_BACK(LIST(<type>) *list);
///   const <type> *LIST_BACK(const LIST(<type>) *list);
///
/// @param list List to operate on
/// @return Pointer to the last item in the list
#define LIST_BACK(list) LIST_AT((list), LIST_SIZE(list) - 1)

/// update the value of an item in a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_SET(LIST(<type>) *list, size_t index, <type> item);
///
/// @param list List to operate on
/// @param index Index of item to update
/// @param item New value to set
#define LIST_SET(list, index, item)                                            \
  do {                                                                         \
    (list)->scratch = (item);                                                  \
    const size_t slot_ = gv_list_get_((list)->impl, (index));                  \
    LIST_DTOR_((list), slot_);                                                 \
    (list)->base[slot_] = (list)->scratch;                                     \
  } while (0)

/// remove an item from a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_REMOVE(LIST(<type>) *list, <type> item);
///
/// @param list List to operate on
/// @param item Item to remove
#define LIST_REMOVE(list, item)                                                \
  do {                                                                         \
    /* get something we can take the address of */                             \
    (list)->scratch = (item);                                                  \
                                                                               \
    const size_t found_ = gv_list_find_((list)->impl, &(list)->scratch,        \
                                        sizeof((list)->base[0]));              \
    if (found_ == SIZE_MAX) { /* not found */                                  \
      break;                                                                   \
    }                                                                          \
                                                                               \
    LIST_DTOR_((list), found_);                                                \
    gv_list_remove_(&(list)->impl, found_, sizeof((list)->base[0]));           \
  } while (0)

/// remove all items from a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_CLEAR(LIST(<type>) *list);
///
/// @param list List to clear
#define LIST_CLEAR(list)                                                       \
  do {                                                                         \
    for (size_t i_ = 0; i_ < LIST_SIZE(list); ++i_) {                          \
      const size_t slot_ = gv_list_get_((list)->impl, i_);                     \
      LIST_DTOR_((list), slot_);                                               \
    }                                                                          \
    gv_list_clear_(&(list)->impl, sizeof((list)->base[0]));                    \
  } while (0)

/// reserve space for new items in a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_RESERVE(LIST(<type>) *list, size_t capacity);
///
/// @param list List to operate on
/// @param capacity Total number of item slots to make available
#define LIST_RESERVE(list, capacity)                                           \
  gv_list_reserve_(&(list)->impl, capacity, sizeof((list)->base[0]))

/// does a list contain a given item?
///
/// You can think of this macro as having the C type:
///
///   bool LIST_CONTAINS(const LIST(<type>) *list, <type> needle);
///
/// The `needle` parameter must be an expression that can have its address
/// taken. E.g. `LIST_CONTAINS(my_ints, 2)` is not valid. This can be worked
/// around with C99 compound literals, `LIST_CONTAINS(my_ints, (int){2})`.
///
/// @param list List to search
/// @param needle Item to search for
/// @return True if the item was found
#define LIST_CONTAINS(list, needle)                                            \
  gv_list_contains_((list)->impl,                                              \
                    ((void)((list)->base == &(needle)), &(needle)),            \
                    sizeof((list)->base[0]))

/// copy a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_COPY(LIST(<type>) *dst, const LIST(<type>) *src);
///
/// @param [out] dst Copy of the source list on completion
/// @param src List to copy
#define LIST_COPY(dst, src)                                                    \
  do {                                                                         \
    memset((dst), 0, sizeof(*(dst)));                                          \
    (void)((dst)->base == (src)->base);                                        \
    (dst)->impl = gv_list_copy_((src)->impl, sizeof((src)->base[0]));          \
    (dst)->dtor = (src)->dtor;                                                 \
  } while (0)

/// does the list not wrap past its end?
///
/// This checks whether the list is discontiguous in how its elements
/// appear in memory:
///
///                         ┌───┬───┬───┬───┬───┬───┬───┬───┐
///   a contiguous list:    │   │   │ w │ x │ y │ z │   │   │
///                         └───┴───┴───┴───┴───┴───┴───┴───┘
///                                   0   1   2   3
///
///                         ┌───┬───┬───┬───┬───┬───┬───┬───┐
///   a discontiguous list: │ y │ z │   │   │   │   │ w │ x │
///                         └───┴───┴───┴───┴───┴───┴───┴───┘
///                           2   3                   0   1
///
/// You can think of this macro as having the C type:
///
///   bool LIST_IS_CONTIGUOUS(const LIST(<type>>) *list);
///
/// @param list List to inspect
/// @return True if the list is contiguous
#define LIST_IS_CONTIGUOUS(list) gv_list_is_contiguous_((list)->impl);

/// shuffle the populated contents to reset `head` to 0
///
/// You can think of this macro as having the C type:
///
///   void LIST_SYNC(LIST(<type>) *list);
///
/// See the `LIST_IS_CONTIGUOUS` leading comment for a better understanding of
/// what it means for `head` to be non-zero.
///
/// @param list List to operate on
#define LIST_SYNC(list) gv_list_sync_(&(list)->impl, sizeof((list)->base[0]))

/// sort a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_SORT(LIST(<type>) *list,
///                  int (*cmp)(const void *a, const void *b));
///
/// @param list List to operate on
/// @param cmp How to compare two list items
#define LIST_SORT(list, cmp)                                                   \
  gv_list_sort_(&(list)->impl, (cmp), sizeof((list)->base[0]))

/// reverse the item order of a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_REVERSE(LIST(<type>) *list);
///
/// @param list List to operate on
#define LIST_REVERSE(list)                                                     \
  gv_list_reverse_(&(list)->impl, sizeof((list)->base[0]))

/// decrease the allocated capacity of a list to minimum
///
/// You can think of this macro as having the C type:
///
///   void LIST_SHRINK_TO_FIT(LIST(<type>) *list);
///
/// @param list List to operate on
#define LIST_SHRINK_TO_FIT(list)                                               \
  gv_list_shrink_to_fit_(&(list)->impl, sizeof((list)->base[0]))

/// free resources associated with a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_FREE(LIST(<type>) *list);
///
/// After a call to this function, the list is empty and may be reused.
///
/// @param list List to free
#define LIST_FREE(list)                                                        \
  do {                                                                         \
    LIST_CLEAR(list);                                                          \
    gv_list_free_(&(list)->impl);                                              \
  } while (0)

/// alias for append
///
/// You can think of this macro as having the C type:
///
///   void LIST_PUSH_BACK(LIST(<type>) *list, <type> item);
///
/// @param list List to operate on
/// @param item Item to append
#define LIST_PUSH_BACK(list, item) LIST_APPEND((list), (item))

/// remove and return the first item of a list
///
/// You can think of this macro as having the C type:
///
///   <type> LIST_POP_FRONT(LIST(<type>) *list);
///
/// @param list List to operate on
/// @return Popped item
#define LIST_POP_FRONT(list)                                                   \
  (gv_list_pop_front_(&(list)->impl, &(list)->scratch,                         \
                      sizeof((list)->base[0])),                                \
   (list)->scratch)

/// remove and return the last item of a list
///
/// You can think of this macro as having the C type:
///
///   <type> LIST_POP_BACK(LIST(<type>) *list);
///
/// @param list List to operate on
/// @return Popped item
#define LIST_POP_BACK(list)                                                    \
  (gv_list_pop_back_(&(list)->impl, &(list)->scratch,                          \
                     sizeof((list)->base[0])),                                 \
   (list)->scratch)

/// remove the last item of a list
///
/// You can think of this macro as having the C type:
///
///   void LIST_DROP_BACK(LIST(<type>) *list);
///
/// This can be used to pop the last element when the caller does not need the
/// popped item.
///
/// @param list List to operate on
#define LIST_DROP_BACK(list)                                                   \
  do {                                                                         \
    const size_t slot_ = gv_list_get_((list)->impl, LIST_SIZE(list) - 1);      \
    LIST_DTOR_((list), slot_);                                                 \
    gv_list_pop_back_(&(list)->impl, &(list)->scratch,                         \
                      sizeof((list)->base[0]));                                \
  } while (0)

/// transform a managed list into a bare array
///
/// You can think of this macro as having the C type:
///
///   void LIST_DETACH(LIST(<type>) *list, <type> **data, size_t *size);
///
/// This can be useful when needing to pass data to a callee who does not
/// use this API. The managed list is emptied and left in a state where it
/// can be reused for other purposes.
///
/// @param list List to operate on
/// @param [out] datap The list data on completion
/// @param [out] sizep The list size on completion
#define LIST_DETACH(list, datap, sizep)                                        \
  gv_list_detach_(&(list)->impl, ((void)(&(list)->base == (datap)), (datap)),  \
                  (sizep), sizeof((list)->base[0]))

#ifdef __cplusplus
}
#endif
