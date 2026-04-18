/**
 * @file
 * @ingroup common_utils
 * @brief geometric functions (e.g. on points and boxes)
 *
 * with application to, but no specific dependence on graphs
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
#include <util/unused.h>

#ifdef __cplusplus
extern "C" {
#endif


#include "geom.h"

#ifdef GVDLL
#ifdef GVC_EXPORTS
#define GEOMPROCS_API __declspec(dllexport)
#else
#define GEOMPROCS_API __declspec(dllimport)
#endif
#endif

#ifndef GEOMPROCS_API
#define GEOMPROCS_API /* nothing */
#endif

/// expand box b as needed to enclose point p
static inline void expandbp(boxf *b, pointf p) {
  b->LL.x = fmin(b->LL.x, p.x);
  b->LL.y = fmin(b->LL.y, p.y);
  b->UR.x = fmax(b->UR.x, p.x);
  b->UR.y = fmax(b->UR.y, p.y);
}

/// expand box b0 as needed to enclose box b1
static inline void expandbb(box *b0, box b1) {
  b0->LL.x = MIN(b0->LL.x, b1.LL.x);
  b0->LL.y = MIN(b0->LL.y, b1.LL.y);
  b0->UR.x = MAX(b0->UR.x, b1.UR.x);
  b0->UR.y = MAX(b0->UR.y, b1.UR.y);
}
static inline void expandbbf(boxf *b0, boxf b1) {
  b0->LL.x = fmin(b0->LL.x, b1.LL.x);
  b0->LL.y = fmin(b0->LL.y, b1.LL.y);
  b0->UR.x = fmax(b0->UR.x, b1.UR.x);
  b0->UR.y = fmax(b0->UR.y, b1.UR.y);
}
#define EXPANDBB(b0, b1) \
  (_Generic((b0), box *: expandbb, boxf *: expandbbf)((b0), (b1)))

GEOMPROCS_API boxf flip_rec_boxf(boxf b, pointf p);

GEOMPROCS_API double ptToLine2 (pointf l1, pointf l2, pointf p);

GEOMPROCS_API int lineToBox(pointf p1, pointf p2, boxf b);
GEOMPROCS_API pointf ccwrotatepf(pointf p, int ccwrot);
GEOMPROCS_API pointf cwrotatepf(pointf p, int cwrot);

GEOMPROCS_API void rect2poly(pointf *p);

GEOMPROCS_API int line_intersect (pointf a, pointf b, pointf c, pointf d, pointf* p);

static inline WUR point add_point(point p, point q) {
    point r;

    r.x = p.x + q.x;
    r.y = p.y + q.y;
    return r;
}

static inline WUR pointf add_pointf(pointf p, pointf q) {
    pointf r;

    r.x = p.x + q.x;
    r.y = p.y + q.y;
    return r;
}

static inline WUR pointf sub_pointf(pointf p, pointf q) {
    pointf r;

    r.x = p.x - q.x;
    r.y = p.y - q.y;
    return r;
}

static inline WUR pointf mid_pointf(pointf p, pointf q) {
    pointf r;

    r.x = (p.x + q.x) / 2.;
    r.y = (p.y + q.y) / 2.;
    return r;
}

static inline WUR pointf interpolate_pointf(double t, pointf p, pointf q) {
    pointf r;

    r.x = p.x + t * (q.x - p.x);
    r.y = p.y + t * (q.y - p.y);
    return r;
}

static inline WUR point exch_xy(point p) {
    point r;

    r.x = p.y;
    r.y = p.x;
    return r;
}

static inline WUR pointf exch_xyf(pointf p) {
    pointf r;

    r.x = p.y;
    r.y = p.x;
    return r;
}

static inline WUR bool boxf_overlap(boxf b0, boxf b1) {
    return OVERLAP(b0, b1);
}

static inline WUR pointf perp (pointf p) {
    pointf r;

    r.x = -p.y;
    r.y = p.x;
    return r;
}

static inline WUR pointf scale (double c, pointf p) {
    pointf r;

    r.x = c * p.x;
    r.y = c * p.y;
    return r;
}

#undef GEOMPROCS_API
#ifdef __cplusplus
}
#endif
