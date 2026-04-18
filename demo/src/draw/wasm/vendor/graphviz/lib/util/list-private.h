/// @file
/// @brief internal implementation details of list.h
/// @ingroup cgraph_utils
///
/// Everything in this header should be considered “private” in the sense that
/// it should not be called except by macros in list.h.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// generic list, agnostic to the list item type
///
/// There is no way to know the size of list items from this structure alone.
/// List item size is expected to be supplied externally.
typedef struct {
  void *base; ///< start of the allocation for backing memory
  ///< (base == NULL && capacity == 0) || (base != NULL && capacity > 0)
  size_t head; ///< index of the first element
  ///< (capacity == 0 && head == 0)  || (capacity > 0 && head < capacity)
  size_t size; ///< number of elements in the list
  ///< size <= capacity
  size_t capacity; ///< available storage slots
} list_t_;

/// get the number of elements in a list
///
/// @param list List to inspect
/// @return Size of the list
static inline size_t gv_list_size_(const list_t_ list) { return list.size; }

/// add an empty space for an item to the end of a list
///
/// This function calls `exit` on failure.
///
/// @param list List to operate on
/// @param item_size Byte size of each list item
/// @return Index of the new (empty) slot
UTIL_API size_t gv_list_append_slot_(list_t_ *list, size_t item_size);

/// try to append a new item to a list
///
/// @param list List to operate on
/// @param item Pointer to item to append
/// @param item_size Byte size of each list item
/// @return True if the append succeeded
UTIL_API bool gv_list_try_append_(list_t_ *list, const void *item,
                                  size_t item_size);

/// add an empty space for an item to the beginning of a list
///
/// This function calls `exit` on failure.
///
/// @param list List to operate on
/// @param item_size Byte size of each list item
/// @return Index of the new (empty) slot
UTIL_API size_t gv_list_prepend_slot_(list_t_ *list, size_t item_size);

/// get the slot index of a given list item index
///
/// @param list List to operate on
/// @param index Index of item to lookup
/// @return Slot index corresponding to the given index
UTIL_API size_t gv_list_get_(const list_t_ list, size_t index);

/// run the destructor of a list on a given slot
///
/// Though this uses the public type `LIST(<type>)` defined in list.h, this is
/// an internal API not expected to be called by anything other than the macros
/// in list.h.
///
/// You can think of this macro as having the following C type:
///
///   void LIST_DTOR_(LIST(<type>) *list, size_t slot);
///
/// @param list List to operate on
/// @param slot Slot to destruct
#define LIST_DTOR_(list, slot)                                                 \
  do {                                                                         \
    if ((list)->dtor == LIST_DTOR_FREE) {                                      \
      /* we need to juggle the element into a pointer to avoid compilation */  \
      /* errors from this (untaken) branch when the element type is not a  */  \
      /* pointer */                                                            \
      void *ptr_;                                                              \
      sizeof((list)->base[0]) == sizeof(ptr_)                                  \
          ? (void)0                                                            \
          : (void)(fprintf(stderr, "list element type is not a pointer, but "  \
                                   "`free` used as destructor\n"),             \
                   abort());                                                   \
      memcpy(&ptr_, &(list)->base[slot], sizeof(ptr_));                        \
      free(ptr_);                                                              \
    } else if ((list)->dtor != NULL) {                                         \
      (list)->dtor((list)->base[slot]);                                        \
    }                                                                          \
  } while (0)

/// find the slot containing the given item
///
/// @param list List to search
/// @param needle Item to search for
/// @param item_size Byte size of each list item
/// @return Slot index on success or `SIZE_MAX` if not found
UTIL_API size_t gv_list_find_(const list_t_ list, const void *needle,
                              size_t item_size);

/// remove a slot from a list
///
/// @param list List to operate on
/// @param index Slot index to remove
/// @param item_size Byte size of each list item
UTIL_API void gv_list_remove_(list_t_ *list, size_t index, size_t item_size);

/// remove all items from a list
///
/// @param list List to operate on
/// @param item_size Byte size of list items
UTIL_API void gv_list_clear_(list_t_ *list, size_t item_size);

/// reserve space for new items in a list
///
/// This function is a no-op if sufficient space is already available.
///
/// @param list List to operate on
/// @param capacity Total number of slots to make available
/// @param item_size Byte size of list items
UTIL_API void gv_list_reserve_(list_t_ *list, size_t capacity,
                               size_t item_size);

/// does a list contain a given item?
///
/// @param list List to search
/// @param needle Item to search for
/// @param item_size Byte size of each list item
/// @return True if the item was found
UTIL_API bool gv_list_contains_(const list_t_ list, const void *needle,
                                size_t item_size);

/// make a copy of a list
///
/// This function calls `exit` on failure.
///
/// @param list List to copy
/// @param item_size Byte size of each list item
/// @return A copy of the original list
UTIL_API list_t_ gv_list_copy_(const list_t_ list, size_t item_size);

/// does the list wrap past its end?
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
///   a discontiguous list: │ y │ z │   │   │   │   │ x │ y │
///                         └───┴───┴───┴───┴───┴───┴───┴───┘
///                           2   3                   0   1
///
/// @param list List to inspect
/// @return True if the list is contiguous
UTIL_API bool gv_list_is_contiguous_(const list_t_ list);

/// shuffle the populated contents to reset `head` to 0
///
/// See the `gv_list_is_contiguous_` leading comment for a better understanding
/// of what it means for `head` to be non-zero.
///
/// @param list List to operate on
/// @param item_size Byte size of each list item
UTIL_API void gv_list_sync_(list_t_ *list, size_t item_size);

/// sort a list
///
/// @param list List to operate on
/// @param cmp Comparator for ordering list items
/// @param item_size Byte size of each list item
UTIL_API void gv_list_sort_(list_t_ *list,
                            int (*cmp)(const void *, const void *),
                            size_t item_size);

/// reverse the item order of a list
///
/// @param list List to operate on
/// @param item_size Byte size of each list item
UTIL_API void gv_list_reverse_(list_t_ *list, size_t item_size);

/// decrease the allocated capacity of a list to minimum
///
/// @param list List to operate on
/// @param item_size Byte size of list items
UTIL_API void gv_list_shrink_to_fit_(list_t_ *list, size_t item_size);

/// free resources associated with a list
///
/// @param list List to operate on
UTIL_API void gv_list_free_(list_t_ *list);

/// remove and return the first item of a list
///
/// @param list List to operate on
/// @param [out] into Destination to pop the item into
/// @param item_size Byte size of each list item
UTIL_API void gv_list_pop_front_(list_t_ *list, void *into, size_t item_size);

/// remove and return the last item of a list
///
/// @param list List to operate on
/// @param [out] into Destination to pop the item into
/// @param item_size Byte size of each list item
UTIL_API void gv_list_pop_back_(list_t_ *list, void *into, size_t item_size);

/// transform a managed list into a bare array
///
/// @param list List to operate on
/// @param [out] datap The list data on completion
/// @param [out] sizep The list size on completion; optional
/// @param item_size Byte size of each list item
UTIL_API void gv_list_detach_(list_t_ *list, void *datap, size_t *sizep,
                              size_t item_size);

#ifdef __cplusplus
}
#endif
