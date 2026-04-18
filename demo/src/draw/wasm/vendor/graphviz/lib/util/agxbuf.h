/**
 * @file
 * @ingroup cgraph_utils
 * @brief Dynamically expanding string buffers
 *
 * Incrementally constructing a string in C is not something easily available
 * out-of-the-box. You either need to statically know the size of the string
 * upfront or rely on non-standard APIs like `asprintf` and `open_memstream`.
 *
 * This header implements a self-contained, reasonably efficient alternative.
 * You can think of this as similar to C++’s `std::ostringstream` or Python’s
 * `io.StringIO`.
 *
 * `agxbuf` includes Short String Optimization (SSO), a technique to pack small
 * strings into the bytes of the structure itself rather than an out-of-line
 * heap buffer. The point of this is to save memory and/or runtime by avoiding
 * allocator calls. In contrast to most other SSO implementations (e.g. those in
 * the `std::string` implementations of many C++ standard libraries), `agxbuf`
 * biases heavily towards saving memory over runtime performance. That is, the
 * maximum length of string is eligible for SSO, at the expense of more runtime
 * code branches. The thinking here (that should periodically be reconsidered)
 * is that expensive Graphviz runs typically hit memory limits before runtime
 * limits, and thus would benefit more from saving memory than running faster.
 */
/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#pragma once

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <util/alloc.h>
#include <util/unused.h>

/// a description of where a buffer is located
typedef enum {
  AGXBUF_INLINE_SIZE_0 = 0,
  AGXBUF_ON_HEAP = 255, ///< buffer is dynamically allocated
  /// other values mean an inline buffer with size N
} agxbuf_loc_t;

/// extensible buffer
///
/// Malloc'ed memory is never released until \p agxbdisown or \p agxbfree is
/// called.
///
/// This has the following layout assuming x86-64.
///
///                                                               located
///                                                                  ↓
///   ┌───────────────┬───────────────┬───────────────┬─────────────┬─┐
///   │      buf      │     size      │   capacity    │   padding   │ │
///   ├───────────────┴───────────────┴───────────────┴─────────────┼─┤
///   │                             store                           │ │
///   └─────────────────────────────────────────────────────────────┴─┘
///   0               8               16              24              32
///
/// \p buf, \p size, and \p capacity are in use when \p located is
/// \p AGXBUF_ON_HEAP. \p store is in use when \p located is <
/// \p AGXBUF_ON_HEAP.
typedef struct {
  union {
    struct {
      char *buf;                        ///< start of buffer
      size_t size;                      ///< number of characters in the buffer
      size_t capacity;                  ///< available bytes in the buffer
      char padding[sizeof(size_t) - 1]; ///< unused; for alignment
      unsigned char
          located; ///< where does the backing memory for this buffer live?
    };
    char store[sizeof(char *) + sizeof(size_t) * 3 -
               1]; ///< inline storage used when \p located is
                   ///< < \p AGXBUF_ON_HEAP
  };
} agxbuf;

static inline bool agxbuf_is_inline(const agxbuf *xb) {
  assert((xb->located == AGXBUF_ON_HEAP || xb->located <= sizeof(xb->store)) &&
         "corrupted agxbuf type");
  return xb->located < AGXBUF_ON_HEAP;
}

/// free any malloced resources
static inline void agxbfree(agxbuf *xb) {
  if (xb->located == AGXBUF_ON_HEAP)
    free(xb->buf);
}

/// return pointer to beginning of buffer
static inline char *agxbstart(agxbuf *xb) {
  return agxbuf_is_inline(xb) ? xb->store : xb->buf;
}

/// return number of characters currently stored
static inline size_t agxblen(const agxbuf *xb) {
  if (agxbuf_is_inline(xb)) {
    return xb->located - AGXBUF_INLINE_SIZE_0;
  }
  return xb->size;
}

/// get the capacity of the backing memory of a buffer
///
/// In contrast to \p agxblen, this is the total number of usable bytes in the
/// backing store, not the total number of currently stored bytes.
///
/// \param xb Buffer to operate on
/// \return Number of usable bytes in the backing store
static inline size_t agxbsizeof(const agxbuf *xb) {
  if (agxbuf_is_inline(xb)) {
    return sizeof(xb->store);
  }
  return xb->capacity;
}

/// removes last character added, if any
static inline int agxbpop(agxbuf *xb) {

  size_t len = agxblen(xb);
  if (len == 0) {
    return -1;
  }

  if (agxbuf_is_inline(xb)) {
    assert(xb->located > AGXBUF_INLINE_SIZE_0);
    int c = xb->store[len - 1];
    --xb->located;
    return c;
  }

  int c = xb->buf[xb->size - 1];
  --xb->size;
  return c;
}

/// expand buffer to hold at least ssz more bytes
static inline void agxbmore(agxbuf *xb, size_t ssz) {
  size_t cnt = 0;   // current no. of characters in buffer
  size_t size = 0;  // current buffer size
  size_t nsize = 0; // new buffer size
  char *nbuf;       // new buffer

  size = agxbsizeof(xb);
  nsize = size == 0 ? BUFSIZ : (2 * size);
  if (size + ssz > nsize)
    nsize = size + ssz;
  cnt = agxblen(xb);

  if (xb->located == AGXBUF_ON_HEAP) {
    nbuf = (char *)gv_recalloc(xb->buf, size, nsize, sizeof(char));
  } else {
    nbuf = (char *)gv_calloc(nsize, sizeof(char));
    memcpy(nbuf, xb->store, cnt);
    xb->size = cnt;
  }
  xb->buf = nbuf;
  xb->capacity = nsize;
  xb->located = AGXBUF_ON_HEAP;
}

/// next position for writing
static inline char *agxbnext(agxbuf *xb) {
  size_t len = agxblen(xb);
  return agxbuf_is_inline(xb) ? &xb->store[len] : &xb->buf[len];
}

/// vprintf-style output to an agxbuf
static inline int vagxbprint(agxbuf *xb, const char *fmt, va_list ap) {
  size_t size;
  int result;

  // determine how many bytes we need to print
  {
    va_list ap2;
    int rc;
    va_copy(ap2, ap);
    rc = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);
    if (rc < 0) {
      return rc;
    }
    size = (size_t)rc + 1; // account for NUL terminator
  }

  // should we use double buffering?
  bool use_stage = false;

  // do we need to expand the buffer?
  {
    size_t unused_space = agxbsizeof(xb) - agxblen(xb);
    if (unused_space < size) {
      size_t extra = size - unused_space;
      if (agxbuf_is_inline(xb) && extra == 1) {
        // The content is currently stored inline, but this print will push it
        // over into being heap-allocated by a single byte. This last byte is a
        // '\0' that `vsnprintf` unavoidably writes but we do not need. So lets
        // avoid this by printing to an intermediate, larger buffer, and then
        // copying the content minus the trailing '\0' to the final destination.
        use_stage = true;
      } else {
        agxbmore(xb, extra);
      }
    }
  }

  // a buffer one byte larger than inline storage to fit the trailing '\0'
  char stage[sizeof(xb->store) + 1] = {0};
  assert(!use_stage || size <= sizeof(stage));

  // we can now safely print into the buffer
  char *dst = use_stage ? stage : agxbnext(xb);
  result = vsnprintf(dst, size, fmt, ap);
  assert(result == (int)(size - 1) || result < 0);
  if (result > 0) {
    if (agxbuf_is_inline(xb)) {
      assert(result <= (int)UCHAR_MAX);
      if (use_stage) {
        memcpy(agxbnext(xb), stage, (size_t)result);
      }
      xb->located += (unsigned char)result;
      assert(agxblen(xb) <= sizeof(xb->store) && "agxbuf corruption");
    } else {
      assert(!use_stage);
      xb->size += (size_t)result;
    }
  }

  return result;
}

/* support for extra API misuse warnings if available */
#ifdef __GNUC__
#define PRINTF_LIKE(index, first) __attribute__((format(printf, index, first)))
#else
#define PRINTF_LIKE(index, first) /* nothing */
#endif

/// Printf-style output to an agxbuf
static inline PRINTF_LIKE(2, 3) int agxbprint(agxbuf *xb, const char *fmt,
                                              ...) {
  va_list ap;
  int result;

  va_start(ap, fmt);

  result = vagxbprint(xb, fmt, ap);

  va_end(ap);
  return result;
}

#undef PRINTF_LIKE

/// append string s of length ssz into xb
static inline size_t agxbput_n(agxbuf *xb, const char *s, size_t ssz) {
  if (ssz == 0) {
    return 0;
  }
  if (ssz > agxbsizeof(xb) - agxblen(xb))
    agxbmore(xb, ssz);
  size_t len = agxblen(xb);
  if (agxbuf_is_inline(xb)) {
    memcpy(&xb->store[len], s, ssz);
    assert(ssz <= UCHAR_MAX);
    xb->located += (unsigned char)ssz;
    assert(agxblen(xb) <= sizeof(xb->store) && "agxbuf corruption");
  } else {
    memcpy(&xb->buf[len], s, ssz);
    xb->size += ssz;
  }
  return ssz;
}

/// append string s into xb
static inline size_t agxbput(agxbuf *xb, const char *s) {
  size_t ssz = strlen(s);

  return agxbput_n(xb, s, ssz);
}

/// add character to buffer
static inline int agxbputc(agxbuf *xb, char c) {
  if (agxblen(xb) >= agxbsizeof(xb)) {
    agxbmore(xb, 1);
  }
  size_t len = agxblen(xb);
  if (agxbuf_is_inline(xb)) {
    xb->store[len] = c;
    ++xb->located;
    assert(agxblen(xb) <= sizeof(xb->store) && "agxbuf corruption");
  } else {
    xb->buf[len] = c;
    ++xb->size;
  }
  return 0;
}

/// resets pointer to data
static inline void agxbclear(agxbuf *xb) {
  if (agxbuf_is_inline(xb)) {
    xb->located = AGXBUF_INLINE_SIZE_0;
  } else {
    xb->size = 0;
  }
}

/* Null-terminates buffer; resets and returns pointer to data. The buffer is
 * still associated with the agxbuf and will be overwritten on the next, e.g.,
 * agxbput. If you want to retrieve and disassociate the buffer, use agxbdisown
 * instead.
 */
static inline WUR char *agxbuse(agxbuf *xb) {
  if (!agxbuf_is_inline(xb) || agxblen(xb) != sizeof(xb->store)) {
    (void)agxbputc(xb, '\0');
  } else {
    // we can skip explicitly null-terminating the buffer because `agxbclear`
    // resets the `xb->located` byte such that it naturally forms a terminator
    assert(AGXBUF_INLINE_SIZE_0 == '\0');
  }

  agxbclear(xb);
  return agxbstart(xb);
}

/* Disassociate the backing buffer from this agxbuf and return it. The buffer is
 * NUL terminated before being returned. If the agxbuf is using stack memory,
 * this will first copy the data to a new heap buffer to then return. If you
 * want to temporarily access the string in the buffer, but have it overwritten
 * and reused the next time, e.g., agxbput is called, use agxbuse instead of
 * agxbdisown.
 */
static inline char *agxbdisown(agxbuf *xb) {
  char *buf;

  if (agxbuf_is_inline(xb)) {
    // the string lives in `store`, so we need to copy its contents to heap
    // memory
    buf = gv_strndup(xb->store, agxblen(xb));
  } else {
    // the buffer is already dynamically allocated, so terminate it and then
    // take it as-is
    agxbputc(xb, '\0');
    buf = xb->buf;
  }

  // reset xb to a state where it is usable
  memset(xb, 0, sizeof(*xb));

  return buf;
}

/** trim extraneous trailing information from a printed floating point value
 *
 * tl;dr:
 *   - “42.00” → “42”
 *   - “42.01” → “42.01”
 *   - “42.10” → “42.1”
 *   - “-0.0” → “0”
 *
 * Printing a \p double or \p float via, for example,
 * \p agxbprint("%.02f", 42.003) can result in output like “42.00”. If this data
 * is destined for something that does generalized floating point
 * parsing/decoding (e.g. SVG viewers) the “.00” is unnecessary. “42” would be
 * interpreted identically. This function can be called after such a
 * \p agxbprint to normalize data.
 *
 * \param xb Buffer to operate on
 */
static inline void agxbuf_trim_zeros(agxbuf *xb) {

  // find last period
  char *start = agxbstart(xb);
  size_t period;
  for (period = agxblen(xb) - 1;; --period) {
    if (period == SIZE_MAX) {
      // we searched the entire string and did not find a period
      return;
    }
    if (start[period] == '.') {
      break;
    }
  }

  // truncate any “0”s that provide no information
  for (size_t follower = agxblen(xb) - 1;; --follower) {
    if (follower == period || start[follower] == '0') {
      // truncate this character
      if (agxbuf_is_inline(xb)) {
        assert(xb->located > AGXBUF_INLINE_SIZE_0);
        --xb->located;
      } else {
        --xb->size;
      }
      if (follower == period) {
        break;
      }
    } else {
      return;
    }
  }

  // is the remainder we have left not “-0”?
  const size_t len = agxblen(xb);
  if (len < 2 || start[len - 2] != '-' || start[len - 1] != '0') {
    return;
  }

  // turn “-0” into “0”
  start[len - 2] = '0';
  if (agxbuf_is_inline(xb)) {
    assert(xb->located > AGXBUF_INLINE_SIZE_0);
    --xb->located;
  } else {
    --xb->size;
  }
}
