/// @file
/// @brief macros for interacting with Address Sanitizer

#pragma once

#ifdef __has_feature
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define ASAN_POISON(addr, size) ASAN_POISON_MEMORY_REGION((addr), (size))
#define ASAN_UNPOISON(addr, size) ASAN_UNPOISON_MEMORY_REGION((addr), (size))
#endif
#endif

#ifndef ASAN_POISON
#define ASAN_POISON(addr, size)                                                \
  do {                                                                         \
    (void)(addr);                                                              \
    (void)(size);                                                              \
  } while (0)
#endif
#ifndef ASAN_UNPOISON
#define ASAN_UNPOISON(addr, size)                                              \
  do {                                                                         \
    (void)(addr);                                                              \
    (void)(size);                                                              \
  } while (0)
#endif
