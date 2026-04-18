/// @file
/// @brief File system path helpers

#pragma once

/// character for separating directory components in a file system path
#if defined(_WIN32) && !defined(__MINGW32__)
#define PATH_SEPARATOR '\\'
#else
#define PATH_SEPARATOR '/'
#endif
