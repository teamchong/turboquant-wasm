/**
 * @file
 * @brief API: cgraph.h
 * @ingroup cgraph_utils
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

#include "config.h"

#include <cgraph/cghdr.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <util/agxbuf.h>
#include <util/gv_ctype.h>
#include <util/gv_math.h>
#include <util/streq.h>

static agerrlevel_t agerrno;             /* Last error level */
static agerrlevel_t agerrlevel = AGWARN; /* Report errors >= agerrlevel */
static int agmaxerr;

static agxbuf last;         ///< last message
static agusererrf usererrf; /* User-set error function */

agusererrf agseterrf(agusererrf newf) {
  agusererrf oldf = usererrf;
  usererrf = newf;
  return oldf;
}

agerrlevel_t agseterr(agerrlevel_t lvl) {
  agerrlevel_t oldv = agerrlevel;
  agerrlevel = lvl;
  return oldv;
}

char *aglasterr(void) {
  // Extract a heap-allocated copy of the last message. Note that this resets
  // `last` to an empty buffer ready to be written to again.
  char *buf = agxbdisown(&last);

  // store the message back again so multiple calls to `aglasterr` can be made
  // without losing the last error
  agxbput(&last, buf);

  // was there no last message?
  if (streq(buf, "")) {
    free(buf);
    return NULL;
  }

  return buf;
}

/// default error reporting implementation
static int default_usererrf(char *message) {
  // `fputs`, escaping characters that may interfere with a terminal
  for (const char *p = message; *p != '\0'; ++p) {

    if (gv_iscntrl(*p) && !gv_isspace(*p)) {
      const int rc = fprintf(stderr, "\\%03o", (unsigned)*p);
      if (rc < 0) {
        return rc;
      }
      continue;
    }

    const int rc = putc(*p, stderr);
    if (rc < 0) {
      return rc;
    }
  }
  return 0;
}

/// Report messages using a user-supplied or default write function
static void out(agerrlevel_t level, const char *fmt, va_list args) {
  // find out how much space we need to construct this string
  size_t bufsz;
  {
    va_list args2;
    va_copy(args2, args);
    int rc = vsnprintf(NULL, 0, fmt, args2);
    va_end(args2);
    if (rc < 0) {
      fprintf(stderr, "%s: vsnprintf failure\n", __func__);
      return;
    }
    bufsz = (size_t)rc + 1; // account for NUL terminator
  }

  // allocate a buffer for the string
  char *buf = malloc(bufsz);
  if (buf == NULL) {
    fprintf(stderr, "%s: could not allocate memory\n", __func__);
    return;
  }

  // determine how errors are to be reported
  const agusererrf errf = usererrf ? usererrf : default_usererrf;

  if (level != AGPREV) {
    (void)errf(level == AGERR ? "Error" : "Warning");
    (void)errf(": ");
  }

  // construct the full error in our buffer
  int rc = vsnprintf(buf, bufsz, fmt, args);
  if (rc < 0) {
    free(buf);
    fprintf(stderr, "%s: vsnprintf failure\n", __func__);
    return;
  }

  // yield our constructed error
  (void)errf(buf);

  free(buf);
}

static int agerr_va(agerrlevel_t level, const char *fmt, va_list args) {
  agerrlevel_t lvl;

  /* Use previous error level if continuation message;
   * Convert AGMAX to AGERROR;
   * else use input level
   */
  lvl = (level == AGPREV ? agerrno : (level == AGMAX) ? AGERR : level);

  /* store this error level */
  agerrno = lvl;
  agmaxerr = imax(agmaxerr, (int)agerrno);

  /* We report all messages whose level is bigger than the user set agerrlevel
   * Setting agerrlevel to AGMAX turns off immediate error reporting.
   */
  if (lvl >= agerrlevel) {
    out(level, fmt, args);
    return 0;
  }

  if (level != AGPREV)
    agxbclear(&last);
  vagxbprint(&last, fmt, args);
  return 0;
}

int agerr(agerrlevel_t level, const char *fmt, ...) {
  va_list args;
  int ret;

  va_start(args, fmt);
  ret = agerr_va(level, fmt, args);
  va_end(args);
  return ret;
}

void agerrorf(const char *fmt, ...) {
  va_list args;

  va_start(args, fmt);
  agerr_va(AGERR, fmt, args);
  va_end(args);
}

void agwarningf(const char *fmt, ...) {
  va_list args;

  va_start(args, fmt);
  agerr_va(AGWARN, fmt, args);
  va_end(args);
}

int agerrors(void) { return agmaxerr; }

int agreseterrors(void) {
  int rc = agmaxerr;
  agmaxerr = 0;
  return rc;
}
