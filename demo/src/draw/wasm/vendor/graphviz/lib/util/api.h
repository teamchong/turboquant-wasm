/// @file
/// @brief macro for API hiding/exposing

#pragma once

/// hide a symbol outside of its containing library
///
/// This should be used for functions that are only called from within the same
/// containing library. That is, functions that would be `static` if the entire
/// library was within a single translation unit. The purpose of this annotation
/// is to reduce symbol conflicts and symbol table bloat in downstream targets
/// that do not need this symbol exposed.
#if !defined(__CYGWIN__) && defined(__GNUC__) && !defined(__MINGW32__)
#define PRIVATE __attribute__((visibility("hidden")))
#else
#define PRIVATE /* Nothing required. In other toolchains, symbol hiding is the \
                   default. */
#endif

/// use this macro to hide libutil’s symbols by default
///
/// The expectation is that users of this library (applications, shared
/// libraries, or static libraries) want to call some of the exposed functions
/// but not re-export them to their users. This annotation is only correct while
/// the containing library is built statically. If it were built as a shared
/// library, API symbols would need to have `default` visibility (and thus be
/// unavoidably re-exported) in order to be callable.
#define UTIL_API PRIVATE
