/// @file
/// @brief XML escaping functionality

#pragma once

#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// options to tweak the behavior of XML escaping
typedef struct {
  /// assume no embedded escapes, and escape "\n" and "\r"
  unsigned raw : 1;
  /// escape '-'
  unsigned dash : 1;
  /// escape consecutive ' '
  unsigned nbsp : 1;
  /// anticipate non-ASCII characters that need to be encoded
  unsigned utf8 : 1;
} xml_flags_t;

/// XML-escape a string
///
/// @param s Source string to process.
/// @param flags Options of how to configure escaping.
/// @param cb An `fputs` analogue for emitting escaped output.
/// @param state Caller-defined data to pass to `cb`.
/// @return The first negative value `cb` returns or the last return value of
///   `cb`.
UTIL_API int gv_xml_escape(const char *s, xml_flags_t flags,
                           int (*cb)(void *state, const char *s), void *state);

#ifdef __cplusplus
}
#endif
