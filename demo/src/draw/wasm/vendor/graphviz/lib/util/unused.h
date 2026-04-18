/// \file
/// \brief abstraction for squashing compiler warnings for unused symbols
/// \ingroup cgraph_utils

#pragma once

/// squash an unused variable/function warning in C
///
/// e.g.
///
///   static UNUSED void my_uncalled_function(void) { }
///   static UNUSED int my_unused_variable;
///
/// Use this sparingly, as the MSVC version applies to everything in both the
/// current and next line, so can end up accidentally masking genuine problems.
#ifdef __GNUC__ // Clang and GCC
#define UNUSED __attribute__((unused))
#elif defined(_MSC_VER) // MSVC
#define UNUSED                                                                 \
  __pragma(warning(suppress : 4100 /* unreferenced formal parameter */         \
                   4101            /* unreferenced local variable */           \
                   4505            /* unreferenced local function */           \
                   ))
#else
#define UNUSED /* nothing */
#endif

/// annotate that a function’s return value should not be ignored
///
/// e.g.
///
///   WUR int foo(void) {
///     return 42;
///   }
///   void bar(void) {
///     foo(); // ← this line will generate a compiler warning
///   }
#ifdef __GNUC__ // Clang and GCC
#define WUR __attribute__((warn_unused_result))
#elif defined(_MSC_VER) // MSVC
#define WUR _Check_return_
#else
#define WUR /* nothing */
#endif
