#pragma once

#include <stdio.h>

/// return type of `itos_`
struct itos_ {
  // space to print a NUL-terminated ≤128-bit signed integer
  char str[41];
};

/// convert a signed number to a string
///
/// This is not expected to be used directly. Callers are expected to use the
/// `ITOS` macro instead.
///
/// @param i Number to convert
/// @return Stringized conversion of the number
static inline struct itos_ itos_(long long i) {
  struct itos_ s;
  snprintf(s.str, sizeof(s.str), "%lld", i);
  return s;
}

/// convert a signed number to a string
///
/// The string returned by this macro has a lifetime that (under ≥C11 semantics)
/// only lasts until the end of the containing full expression. Thus intended
/// usage is something like:
///
///   void foo(char *);
///   foo(ITOS(42));
///
/// In particular, it is incorrect to store the result of this macro anywhere:
///
///   char *p = ITOS(42); // ← WRONG
///   // p is now a dangling pointer, pointing to deallocated stack memory
///
/// You can think of `ITOS` as a C equivalent of the C++
/// `std::to_string(i).c_str()`.
///
/// @param i Number to convert
/// @return Stringized conversion of the number
#define ITOS(i) (itos_(i).str)
