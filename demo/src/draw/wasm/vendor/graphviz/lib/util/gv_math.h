/// \file
/// \brief Arithmetic helper functions
/// \ingroup cgraph_utils

#pragma once

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/// comparator for doubles
static inline int fcmp(double a, double b) {
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

/// maximum of two integers
static inline int imax(int a, int b) { return a > b ? a : b; }

/// maximum of two sizes
static inline size_t zmax(size_t a, size_t b) { return a > b ? a : b; }

/// minimum of two integers
static inline int imin(int a, int b) { return a < b ? a : b; }

/**
 * \brief are two values precisely the same?
 *
 * This function should only be used when you know you want comparison with no
 * tolerance, which is rare. Floating-point arithmetic accumulates imprecision,
 * so equality comparisons should generally include a non-zero tolerance to
 * account for this. In general, this function is only applicable for checking
 * things like “is this variable unchanged since a previous assignment from a
 * literal?”
 *
 * \param a First operand to comparison
 * \param b Second operand to comparison
 * \return True if the values are equal
 */
static inline bool is_exactly_equal(double a, double b) {
  return memcmp(&a, &b, sizeof(a)) == 0;
}

/**
 * \brief is a value precisely 0.0?
 *
 * This function should only be used when you know you want comparison with no
 * tolerance, which is rare. Floating-point arithmetic accumulates imprecision,
 * so equality comparisons should generally include a non-zero tolerance to
 * account for this. Valid `double` representations even include -0.0, for which
 * this function will return false. In general, this function is only applicable
 * for checking things like “is this variable unchanged since a previous
 * assignment from the literal `0`?” or “did this value we parsed from user
 * input originate from the string "0.0"?”
 *
 * \param v Value to check
 * \return True if the value is equal to exactly 0.0
 */
static inline bool is_exactly_zero(double v) { return is_exactly_equal(v, 0); }

/**
 * \brief scale up or down a non-negative integer, clamping to \p [0, INT_MAX]
 *
 * \param original Value to scale
 * \param scale Scale factor to apply
 * \return Clamped result
 */
static inline int scale_clamp(int original, double scale) {
  assert(original >= 0);

  if (scale < 0) {
    return 0;
  }

  if (scale > 1 && original > INT_MAX / scale) {
    return INT_MAX;
  }

  return (int)(original * scale);
}

/// byte length of data per pixel in image data buffers
enum { BYTES_PER_PIXEL = 4 };

/// in-place conversion of ARGB32 big endian to RGBA32 little endian
///
/// Image data originating from sources like Cairo comes in a 4-byte-per-pixel
/// format ordered {blue, green, red, alpha}. Some output libraries/devices
/// instead consume a 4-byte-per-pixel format ordered {red, green, blue, alpha}.
/// This function converts the former to the latter.
///
/// It is confusing to refer to these formats as ARGB32 and RGBA32 when the
/// first is big endian and the second is little endian, and thus the mappings
/// of letters in the acronym to bytes in memory are opposites. Nevertheless
/// that appears to be what is in common usage.
///
/// @param width Width of the image in pixels
/// @param height Height of the image in pixels
/// @param data [in,out] Image data in ARGB32 format on entry and RGBA32 format
///   on exit
static inline void argb2rgba(size_t width, size_t height, unsigned char *data) {
  assert(data != NULL || (width == 0 && height == 0));

  // indices to color bytes in each format
  enum { Ba = 0, Ga = 1, Ra = 2, Aa = 3 };
  enum { Rb = 0, Gb = 1, Bb = 2, Ab = 3 };

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const unsigned char red = data[Ra];
      const unsigned char blue = data[Ba];
      data[Rb] = red;
      data[Bb] = blue;
      data += BYTES_PER_PIXEL;
    }
  }
}

/// swap data referenced by two pointers
///
/// You can think of this macro as having the following C type:
///
///   void SWAP(<t1> *a, <t1> *b);
///
/// Both `a` and `b` are expected to be pure expressions.
#define SWAP(a, b)                                                             \
  do {                                                                         \
    /* trigger a -Wcompare-distinct-pointer-types compiler warning if `a` */   \
    /* and `b` have differing types                                       */   \
    (void)((a) == (b));                                                        \
                                                                               \
    /* Swap their targets. Contemporary compilers will optimize the `memcpy`s  \
     * into direct writes for primitive types.                                 \
     */                                                                        \
    char tmp_[sizeof(*(a))];                                                   \
    memcpy(tmp_, (a), sizeof(*(a)));                                           \
    *(a) = *(b);                                                               \
    memcpy((b), tmp_, sizeof(*(b)));                                           \
  } while (0)

/// convert a double-precision floating-point to integer
///
/// Values below or above the range of representable integers are clamped. The
/// caller should verify this lossy conversion is acceptable/desirable.
///
/// @param v Double value to convert
/// @return Closest integer value
static inline int d2i(double v) {
  if (v > INT_MAX) {
    return INT_MAX;
  }
  if (v < INT_MIN) {
    return INT_MIN;
  }
  return (int)v;
}

/// convert a double-precision floating-point to single-precision
///
/// Values below or above the range of representable floats are clamped. There
/// is loss of precision resulting from both this and that single-precision
/// space inherently has less resolution than double-precision space. The caller
/// should verify this loss of precision is acceptable/desirable.
///
/// @param v Double value to convert
/// @return Closest float value
static inline float d2f(double v) {
  if (v > FLT_MAX) {
    return FLT_MAX;
  }
  if (v < -FLT_MAX) {
    return -FLT_MAX;
  }
  return (float)v;
}
