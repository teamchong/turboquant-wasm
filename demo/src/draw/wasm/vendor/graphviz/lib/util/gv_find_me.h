/// @file
/// @brief platform abstraction for finding the path to yourself

#pragma once

#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// find an absolute path to the current executable
///
/// The caller is responsible for freeing the returned pointer.
///
/// It is assumed the containing executable is an on-disk file. If it is an
/// in-memory executable with no actual path, results are undefined.
///
/// @return An absolute path to the containing executable on success or `NULL`
///   on failure
UTIL_API char *gv_find_me(void);

#ifdef __cplusplus
}
#endif
