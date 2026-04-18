/// \file
/// \brief replacements for ctype.h functions
/// \ingroup cgraph_utils
///
/// The behavior of the ctype.h functions is locale-dependent, while Graphviz
/// code typically wants to ask about character data specifically interpreted as
/// ASCII. The current locale is frequently irrelevant because Graphviz (1)
/// supports input in encodings different than the user’s locale via the
/// `charset` attribute and (2) is often producing output formats that are
/// implicitly ASCII-only.
///
/// This discrepancy leads to misbehavior when trying to use the ctype.h
/// functions as-is. For example, certain Windows environments with a signed
/// `char` type crash when `isdigit` is called with a `char` that is part of a
/// multi-byte unicode character and has its high bit set.
///
/// There are various solutions to this like using a full internationalization
/// library or constructing an ASCII locale and calling the `*_l` variants. But
/// for simplicity we just implement the exact discriminators we need.

#pragma once

#include <stdbool.h>

static inline bool gv_islower(int c) { return c >= 'a' && c <= 'z'; }

static inline bool gv_isupper(int c) { return c >= 'A' && c <= 'Z'; }

static inline bool gv_isalpha(int c) { return gv_islower(c) || gv_isupper(c); }

static inline bool gv_isblank(int c) { return c == ' ' || c == '\t'; };

static inline bool gv_iscntrl(int c) {
  if (c >= 0 && c <= 31)
    return true;
  if (c == 127)
    return true;
  return false;
}

static inline bool gv_isdigit(int c) { return c >= '0' && c <= '9'; }

static inline bool gv_isalnum(int c) { return gv_isalpha(c) || gv_isdigit(c); }

static inline bool gv_isgraph(int c) { return c > 32 && c < 127; }

static inline bool gv_isprint(int c) { return c > 31 && c < 127; }

static inline bool gv_ispunct(int c) {
  if (gv_isalnum(c))
    return false;
  return c > 32 && c < 127;
}

static inline bool gv_isspace(int c) {
  if (c == '\t')
    return true;
  if (c == '\n')
    return true;
  if (c == '\v')
    return true;
  if (c == '\f')
    return true;
  if (c == '\r')
    return true;
  if (c == ' ')
    return true;
  return false;
}

static inline bool gv_isxdigit(int c) {
  if (gv_isdigit(c))
    return true;
  if (c >= 'A' && c <= 'F')
    return true;
  if (c >= 'a' && c <= 'f')
    return true;
  return false;
}

static inline char gv_tolower(int c) {
  if (gv_isupper(c))
    return (char)c - 'A' + 'a';
  return (char)c;
}

static inline void gv_tolower_str(char *s) {
  for (; *s != '\0'; ++s) {
    *s = gv_tolower(*s);
  }
}

static inline char gv_toupper(int c) {
  if (gv_islower(c))
    return (char)c - 'a' + 'A';
  return (char)c;
}

static inline void gv_toupper_str(char *s) {
  for (; *s != '\0'; ++s) {
    *s = gv_toupper(*s);
  }
}
