/**
 * @file
 * @brief trapezoid elements and utilities for partition.c
 *
 * See [Fast polygon triangulation based on Seidel's algorithm](http://gamma.cs.unc.edu/SEIDEL/)
 *
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

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <util/list.h>

/* Segment attributes */

typedef struct {
  pointf v0, v1;       /* two endpoints */ 
  bool is_inserted;      /* inserted in trapezoidation yet ? */
  size_t root0, root1;   ///< root nodes in Q
  int next;         /* Next logical segment */
  int prev;         /* Previous segment */
} segment_t;


/* Trapezoid attributes */

typedef struct {
  int lseg, rseg;       /* two adjoining segments */
  pointf hi, lo;       /* max/min y-values */ 
  size_t u0, u1;
  size_t d0, d1;
  size_t sink;          ///< pointer to corresponding in Q
  size_t usave;         ///< I forgot what this means
  int uside;            ///< I forgot what this means
  bool is_valid;
} trap_t; 

/// is the given index a reference to an existing trapezoid?
///
/// The indices stored in `trap_t.{u0|u1|d0|d1|usave}` can contain sentinel
/// values that represent an unset or invalid trapezoid. This function can be
/// used to determine the validity of these fields.
static inline bool is_valid_trap(size_t index) {
  return index != 0 && index != SIZE_MAX;
}

/// an array of trapezoids
typedef LIST(trap_t) traps_t;

#define C_EPS 1.0e-7        /* tolerance value: Used for making */
                /* all decisions about collinearity or */
                /* left/right of segment. Decrease */
                /* this value if the input points are */
                /* spaced very close together */

static inline bool fp_equal(double s, double t) { return fabs(s - t) <= C_EPS; }

/**
 * @brief double floating point three-way comparison
 *
 * Returns -1, 0, or 1 if f1 respectively, less than,
 * almost equal with tolerance C_EPS, or greater than f2.
 * The purpose of the function is workaround precision issues.
 */

static inline int dfp_cmp(double f1, double f2) {
  double d = f1 - f2;
  if (d < -C_EPS)
    return -1;
  if (d > C_EPS)
    return 1;
  return 0;
}

static inline bool equal_to(pointf v0, pointf v1) {
  return fp_equal(v0.y, v1.y) && fp_equal(v0.x, v1.x);
}

static inline bool greater_than(pointf v0, pointf v1) {
  return v0.y > v1.y + C_EPS ? true
                             : (v0.y < v1.y - C_EPS ? false : v0.x > v1.x);
}

extern traps_t construct_trapezoids(int, segment_t*, int*);
