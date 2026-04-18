/// @file
/// @brief wrapper around `fopen` for internal library usage

#pragma once

#include <stdio.h>
#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// open a file, setting close-on-exec
///
/// Generally, library code should set close-on-exec (`O_CLOEXEC`) on file
/// descriptors it creates to avoid child processes of concurrent `fork`+`exec`
/// operations accidentally inheriting copies of the descriptors. It is tricky
/// to achieve this without races. This function attempts to avoid the common
/// problems when trying to do this with `fopen`.
///
/// @param filename A filename, as you would pass to `fopen`
/// @param mode A mode, as you would pass to `fopen`
/// @return A file handle with close-on-exit set on success or `NULL` on failure
UTIL_API FILE *gv_fopen(const char *filename, const char *mode);

#ifdef __cplusplus
}
#endif
