/// @file
/// @brief C analog of C++’s `std::optional`

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

/// a container that may or may not contain a `double` value
typedef struct {
  bool has_value; ///< does this have a value?
  double value;   ///< the value if `has_value` is true
} optional_double_t;

/// set the value of an optional
///
/// This utility function is intended to avoid the easy typo of setting the
/// value while forgetting to set the `has_value` member.
///
/// @param me The optional whose value to set
/// @param value The value to assign
static inline void optional_double_set(optional_double_t *me, double value) {
  assert(me != NULL);
  me->has_value = true;
  me->value = value;
}

/// get the value of an optional or a given value if the optional is empty
///
/// @param me The optional whose value to retrieve
/// @param fallback The value to return if the optional is empty
/// @return Value of the optional or `fallback` if it was empty
static inline double optional_double_value_or(optional_double_t me,
                                              double fallback) {
  if (me.has_value) {
    return me.value;
  }
  return fallback;
}
