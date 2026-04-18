/// @file
/// @brief Base64 encoding
#pragma once

#include <stddef.h>
#include <util/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/// how many bytes does it take to encode a given source data length?
///
/// @param source_size The number of bytes in the source to encoding
/// @return The number of bytes required in the destination of encoding
UTIL_API size_t gv_base64_size(size_t source_size);

/// Base64 encode some data
///
/// This function does not return on failure, like memory allocation. It calls
/// `exit`. The caller is expected to `free` the returned pointer.
///
/// @param source Pointer to the start of data to encode
/// @param size Number of bytes in the source
/// @return A buffer of the encoded data
UTIL_API char *gv_base64(const unsigned char *source, size_t size);

#ifdef __cplusplus
}
#endif
