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

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pathplan/pathutil.h>
#include <pathplan/tri.h>
#include <util/alloc.h>

static int triangulate(Ppoint_t **pointp, size_t pointn,
                       void (*fn)(void *, const Ppoint_t *), void *vc);

int ccw(Ppoint_t p1, Ppoint_t p2, Ppoint_t p3) {
    double d = (p1.y - p2.y) * (p3.x - p2.x) - (p3.y - p2.y) * (p1.x - p2.x);
    return d > 0 ? ISCW : (d < 0 ? ISCCW : ISON);
}

static Ppoint_t point_indexer(void *base, size_t index) {
  Ppoint_t **b = base;
  return *b[index];
}

/* Ptriangulate:
 * Return 0 on success; non-zero on error.
 */
int Ptriangulate(Ppoly_t *polygon, void (*fn)(void *, const Ppoint_t *),
                 void *vc) {
    Ppoint_t **pointp;

    const size_t pointn = polygon->pn;

    pointp = gv_calloc(pointn, sizeof(Ppoint_t*));

    for (size_t i = 0; i < pointn; i++)
	pointp[i] = &(polygon->ps[i]);

    assert(pointn >= 3);
    if (triangulate(pointp, pointn, fn, vc) != 0) {
	free(pointp);
	return 1;
    }

    free(pointp);
    return 0;
}

/* triangulate:
 * Triangulates the given polygon. 
 * Returns non-zero if no diagonal exists.
 */
static int triangulate(Ppoint_t **pointp, size_t pointn,
                       void (*fn)(void *, const Ppoint_t *), void *vc) {
    assert(pointn >= 3);
    Ppoint_t A[3];
    if (pointn > 3) {
	for (size_t i = 0; i < pointn; i++) {
	    const size_t ip1 = (i + 1) % pointn;
	    const size_t ip2 = (i + 2) % pointn;
	    if (isdiagonal(i, ip2, pointp, pointn, point_indexer)) {
		A[0] = *pointp[i];
		A[1] = *pointp[ip1];
		A[2] = *pointp[ip2];
		fn(vc, A);
		size_t j = 0;
		for (i = 0; i < pointn; i++)
		    if (i != ip1)
			pointp[j++] = pointp[i];
		return triangulate(pointp, pointn - 1, fn, vc);
	    }
	}
	return -1;
    } else {
	A[0] = *pointp[0];
	A[1] = *pointp[1];
	A[2] = *pointp[2];
	fn(vc, A);
    }
    return 0;
}

/// is pb between pa and pc?
static bool between(Ppoint_t pa, Ppoint_t pb, Ppoint_t pc) {
  const Ppoint_t pba = {.x = pb.x - pa.x, .y = pb.y - pa.y};
  const Ppoint_t pca = {.x = pc.x - pa.x, .y = pc.y - pa.y};
  if (ccw(pa, pb, pc) != ISON)
    return false;
  return pca.x * pba.x + pca.y * pba.y >= 0 &&
         pca.x * pca.x + pca.y * pca.y <= pba.x * pba.x + pba.y * pba.y;
}

/// line to line intersection
static bool intersects(Ppoint_t pa, Ppoint_t pb, Ppoint_t pc, Ppoint_t pd) {
  int ccw1, ccw2, ccw3, ccw4;

  if (ccw(pa, pb, pc) == ISON || ccw(pa, pb, pd) == ISON ||
      ccw(pc, pd, pa) == ISON || ccw(pc, pd, pb) == ISON) {
    if (between(pa, pb, pc) || between(pa, pb, pd) || between(pc, pd, pa) ||
        between(pc, pd, pb))
      return true;
  } else {
    ccw1 = ccw(pa, pb, pc) == ISCCW ? 1 : 0;
    ccw2 = ccw(pa, pb, pd) == ISCCW ? 1 : 0;
    ccw3 = ccw(pc, pd, pa) == ISCCW ? 1 : 0;
    ccw4 = ccw(pc, pd, pb) == ISCCW ? 1 : 0;
    return (ccw1 ^ ccw2) && (ccw3 ^ ccw4);
  }
  return false;
}

bool isdiagonal(size_t i, size_t ip2, void *pointp, size_t pointn,
                indexer_t indexer) {
    int res;

    /* neighborhood test */
    const size_t ip1 = (i + 1) % pointn;
    const size_t im1 = (i + pointn - 1) % pointn;
    /* If P[i] is a convex vertex [ i+1 left of (i-1,i) ]. */
    if (ccw(indexer(pointp, im1), indexer(pointp, i), indexer(pointp, ip1)) == ISCCW)
	res = ccw(indexer(pointp, i), indexer(pointp, ip2), indexer(pointp, im1)) == ISCCW &&
	    ccw(indexer(pointp, ip2), indexer(pointp, i), indexer(pointp, ip1)) == ISCCW;
    /* Assume (i - 1, i, i + 1) not collinear. */
    else
	res = ccw(indexer(pointp, i), indexer(pointp, ip2), indexer(pointp, ip1)) == ISCW;
    if (!res) {
	return false;
    }

    /* check against all other edges */
    for (size_t j = 0; j < pointn; j++) {
	const size_t jp1 = (j + 1) % pointn;
	if (!(j == i || jp1 == i || j == ip2 || jp1 == ip2))
	    if (intersects
		(indexer(pointp, i), indexer(pointp, ip2), indexer(pointp, j), indexer(pointp, jp1))) {
		return false;
	    }
    }
    return true;
}
