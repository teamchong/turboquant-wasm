/// @file
/// @brief helpers for verbose/debug printing

#pragma once

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <util/lockfile.h>

/// print an informational message
///
/// This assumes the `Verbose` global is in scope.
#define GV_INFO(...)                                                           \
  do {                                                                         \
    if (Verbose) {                                                             \
      const char *const name_ = strrchr(__FILE__, '/') == NULL                 \
                                    ? __FILE__                                 \
                                    : strrchr(__FILE__, '/') + 1;              \
      lockfile(stderr);                                                        \
      const time_t now_ = time(NULL);                                          \
      const struct tm *const now_tm_ = localtime(&now_);                       \
      fprintf(stderr, "[Graphviz] %s:%d: %04d-%02d-%02d %02d:%02d:%02d ",      \
              name_, __LINE__, now_tm_->tm_year + 1900, now_tm_->tm_mon + 1,   \
              now_tm_->tm_mday, now_tm_->tm_hour, now_tm_->tm_min,             \
              now_tm_->tm_sec);                                                \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      unlockfile(stderr);                                                      \
    }                                                                          \
  } while (0)

/// print a debug message
///
/// The distinction between this and `GV_INFO` is purely semantic; this is
/// intended for messages for Graphviz developers while `GV_INFO` is intended
/// for messages for Graphviz users. In future, `GV_DEBUG` may use something
/// other than `Verbose` to guard its output.
#define GV_DEBUG(...) GV_INFO(__VA_ARGS__)
