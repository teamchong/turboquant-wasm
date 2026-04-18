/// @file
/// API geomprocs.h
/// @ingroup common_utils
/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/* geometric functions (e.g. on points and boxes) with application to, but
 * no specific dependence on graphs */

#include "config.h"
#include <assert.h>
#include <common/geom.h>
#include <common/geomprocs.h>
#include <math.h>
#include <stdbool.h>
#include <util/unreachable.h>

/*
 *--------------------------------------------------------------
 *
 * lineToBox --
 *
 *      Determine whether a line lies entirely inside, entirely
 *      outside, or overlapping a given rectangular area.
 *
 * Results:
 *      -1 is returned if the line given by p and q
 *      is entirely outside the rectangle given by b.
 * 	0 is returned if the polygon overlaps the rectangle, and
 *	1 is returned if the polygon is entirely inside the rectangle.
 *
 * Side effects:
 *      None.
 *
 *--------------------------------------------------------------
 */

/* This code steals liberally from algorithms in tk/generic/tkTrig.c -- jce */

int lineToBox(pointf p, pointf q, boxf b)
{
    /*
     * First check the two points individually to see whether they
     * are inside the rectangle or not.
     */

    bool inside1 = INSIDE(p, b);
    bool inside2 = INSIDE(q, b);
    if (inside1 != inside2) {
        return 0;
    }
    if (inside1 && inside2) {
        return 1;
    }

    /*
     * Both points are outside the rectangle, but still need to check
     * for intersections between the line and the rectangle.  Horizontal
     * and vertical lines are particularly easy, so handle them
     * separately.
     */

    if (p.x == q.x) {
        // vertical line
        if (((p.y >= b.LL.y) ^ (q.y >= b.LL.y)) && BETWEEN(b.LL.x, p.x, b.UR.x)) {
            return 0;
        }
    } else if (p.y == q.y) {
        // horizontal line
        if (((p.x >= b.LL.x) ^ (q.x >= b.LL.x)) && BETWEEN(b.LL.y, p.y, b.UR.y)) {
            return 0;
        }
    } else {
        double m, x, y;

        /*
         * Diagonal line.  Compute slope of line and use
         * for intersection checks against each of the
         * sides of the rectangle: left, right, bottom, top.
         */

        m = (q.y - p.y)/(q.x - p.x);
        double low = fmin(p.x, q.x);
        double high = fmax(p.x, q.x);

        // left edge
        y = p.y + (b.LL.x - p.x)*m;
        if (BETWEEN(low, b.LL.x, high) && BETWEEN(b.LL.y, y, b.UR.y)) {
            return 0;
        }

        // right edge
        y += (b.UR.x - b.LL.x)*m;
        if (BETWEEN(b.LL.y, y, b.UR.y) && BETWEEN(low, b.UR.x, high)) {
            return 0;
        }

        // bottom edge
        low = fmin(p.y, q.y);
        high = fmax(p.y, q.y);
        x = p.x + (b.LL.y - p.y)/m;
        if (BETWEEN(b.LL.x, x, b.UR.x) && BETWEEN(low, b.LL.y, high)) {
            return 0;
        }

        // top edge
        x += (b.UR.y - b.LL.y)/m;
        if (BETWEEN(b.LL.x, x, b.UR.x) && BETWEEN(low, b.UR.y, high)) {
            return 0;
        }
    }
    return -1;
}

void rect2poly(pointf *p)
{
    p[3].x = p[2].x = p[1].x;
    p[2].y = p[1].y;
    p[3].y = p[0].y;
    p[1].x = p[0].x;
}

pointf cwrotatepf(pointf p, int cwrot)
{
    assert(cwrot == 0 || cwrot == 90 || cwrot == 180 || cwrot == 270);
    switch (cwrot) {
    case 0:
	break;
    case 90:
	return (pointf){.x = p.y, .y = -p.x};
    case 180:
	return (pointf){.x = p.x, .y = -p.y};
    case 270:
	return exch_xyf(p);
    default:
	UNREACHABLE();
    }
    return p;
}

pointf ccwrotatepf(pointf p, int ccwrot)
{
    assert(ccwrot == 0 || ccwrot == 90 || ccwrot == 180 || ccwrot == 270);
    switch (ccwrot) {
    case 0:
	break;
    case 90:
	return perp(p);
    case 180:
	return (pointf){.x = p.x, .y = -p.y};
    case 270:
	return exch_xyf(p);
    default:
	UNREACHABLE();
    }
    return p;
}

boxf flip_rec_boxf(boxf b, pointf p)
{
    /* flip box */
    boxf r = {.LL = exch_xyf(b.LL), .UR = exch_xyf(b.UR)};
    /* move box */
    r.LL = add_pointf(r.LL, p);
    r.UR = add_pointf(r.UR, p);
    return r;
}

#define SMALL 0.0000000001

/* ptToLine2:
 * Return distance from point p to line a-b squared.
 */
double ptToLine2 (pointf a, pointf b, pointf p)
{
  double dx = b.x-a.x;
  double dy = b.y-a.y;
  double a2 = (p.y-a.y)*dx - (p.x-a.x)*dy;
  a2 *= a2;   /* square - ensures that it is positive */
  if (a2 < SMALL) return 0.;  /* avoid 0/0 problems */
  return a2 / (dx*dx + dy*dy);
}

#define dot(v,w) (v.x*w.x+v.y*w.y)

/* line_intersect:
 * Computes intersection of lines a-b and c-d, returning intersection
 * point in *p.
 * Returns 0 if no intersection (lines parallel), 1 otherwise.
 */
int line_intersect (pointf a, pointf b, pointf c, pointf d, pointf* p)
{

    pointf mv = sub_pointf(b,a);
    pointf lv = sub_pointf(d,c);
    pointf ln = perp (lv);
    double lc = -dot(ln,c);
    double dt = dot(ln,mv);

    if (fabs(dt) < SMALL) return 0;

    *p = sub_pointf(a,scale((dot(ln,a)+lc)/dt,mv));
    return 1;
}

