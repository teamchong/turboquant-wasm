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

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <pathplan/pathutil.h>
#include <pathplan/tri.h>
#include <util/list.h>
#include <util/prisize_t.h>

#define DQ_FRONT 1
#define DQ_BACK  2

#define prerror(msg) \
        fprintf (stderr, "lib/pathplan/%s:%d: %s\n", __FILE__, __LINE__, (msg))

#define POINTSIZE sizeof (Ppoint_t)

typedef struct pointnlink_t {
    Ppoint_t *pp;
    struct pointnlink_t *link;
} pointnlink_t;

#define POINTNLINKSIZE sizeof (pointnlink_t)
#define POINTNLINKPSIZE sizeof (pointnlink_t *)

typedef struct {
    pointnlink_t *pnl0p;
    pointnlink_t *pnl1p;
    size_t right_index; ///< index into \p tris of the triangle to the right
} tedge_t;

typedef struct triangle_t {
    int mark;
    tedge_t e[3];
} triangle_t;

typedef struct deque_t {
    pointnlink_t **pnlps;
    size_t pnlpn, fpnlpi, lpnlpi, apex;
} deque_t;

static LIST(triangle_t) tris;

static Ppoint_t *ops;
static size_t opn;

static int triangulate(pointnlink_t **, size_t);
static int loadtriangle(pointnlink_t *, pointnlink_t *, pointnlink_t *);
static void connecttris(size_t, size_t);
static bool marktripath(size_t, size_t);

static void add2dq(deque_t *dq, int, pointnlink_t*);
static void splitdq(deque_t *dq, int, size_t);
static size_t finddqsplit(const deque_t *dq, pointnlink_t*);

static int pointintri(size_t, Ppoint_t *);

static int growops(size_t);

static Ppoint_t point_indexer(void *base, size_t index) {
  pointnlink_t **b = base;
  return *b[index]->pp;
}

/* Pshortestpath:
 * Find a shortest path contained in the polygon polyp going between the
 * points supplied in eps. The resulting polyline is stored in output.
 * Return 0 on success, -1 on bad input, -2 on memory allocation problem. 
 */
int Pshortestpath(Ppoly_t * polyp, Ppoint_t eps[2], Ppolyline_t * output)
{
    size_t pi, minpi;
    double minx;
    size_t trii, trij, ftrii, ltrii;
    int ei;
    pointnlink_t epnls[2], *lpnlp, *rpnlp, *pnlp;
    triangle_t *trip;

    /* make space */
    pointnlink_t *pnls = calloc(polyp->pn, sizeof(pnls[0]));
    if (polyp->pn > 0 && pnls == NULL) {
	prerror("cannot realloc pnls");
	return -2;
    }
    pointnlink_t **pnlps = calloc(polyp->pn, sizeof(pnlps[0]));
    if (polyp->pn > 0 && pnlps == NULL) {
	prerror("cannot realloc pnlps");
	free(pnls);
	return -2;
    }
    size_t pnll = 0;
    LIST_CLEAR(&tris);

    deque_t dq = {.pnlpn = polyp->pn * 2};
    dq.pnlps = calloc(dq.pnlpn, POINTNLINKPSIZE);
    if (dq.pnlps == NULL) {
	prerror("cannot realloc dq.pnls");
	free(pnlps);
	free(pnls);
	return -2;
    }
    dq.fpnlpi = dq.pnlpn / 2;
    dq.lpnlpi = dq.fpnlpi - 1;

    /* make sure polygon is CCW and load pnls array */
    for (pi = 0, minx = HUGE_VAL, minpi = SIZE_MAX; pi < polyp->pn; pi++) {
	if (minx > polyp->ps[pi].x)
	    minx = polyp->ps[pi].x, minpi = pi;
    }
    const Ppoint_t p2 = polyp->ps[minpi];
    const Ppoint_t p1 = polyp->ps[minpi == 0 ? polyp->pn - 1 : minpi - 1];
    const Ppoint_t p3 = polyp->ps[(minpi + 1) % polyp->pn];
    if ((p1.x == p2.x && p2.x == p3.x && p3.y > p2.y) ||
	ccw(p1, p2, p3) != ISCCW) {
	for (pi = polyp->pn - 1; pi != SIZE_MAX; pi--) {
	    if (pi < polyp->pn - 1
		&& polyp->ps[pi].x == polyp->ps[pi + 1].x
		&& polyp->ps[pi].y == polyp->ps[pi + 1].y)
		continue;
	    pnls[pnll].pp = &polyp->ps[pi];
	    pnls[pnll].link = &pnls[pnll % polyp->pn];
	    pnlps[pnll] = &pnls[pnll];
	    pnll++;
	}
    } else {
	for (pi = 0; pi < polyp->pn; pi++) {
	    if (pi > 0 && polyp->ps[pi].x == polyp->ps[pi - 1].x &&
		polyp->ps[pi].y == polyp->ps[pi - 1].y)
		continue;
	    pnls[pnll].pp = &polyp->ps[pi];
	    pnls[pnll].link = &pnls[pnll % polyp->pn];
	    pnlps[pnll] = &pnls[pnll];
	    pnll++;
	}
    }

#if defined(DEBUG) && DEBUG >= 1
    fprintf(stderr, "points\n%" PRISIZE_T "\n", pnll);
    for (size_t pnli = 0; pnli < pnll; pnli++)
	fprintf(stderr, "%f %f\n", pnls[pnli].pp->x, pnls[pnli].pp->y);
#endif

    /* generate list of triangles */
    if (triangulate(pnlps, pnll)) {
	free(dq.pnlps);
	free(pnlps);
	free(pnls);
	return -2;
    }

#if defined(DEBUG) && DEBUG >= 2
    fprintf(stderr, "triangles\n%" PRISIZE_T "\n", LIST_SIZE(&tris));
    for (trii = 0; trii < LIST_SIZE(&tris); trii++)
	for (ei = 0; ei < 3; ei++)
	    fprintf(stderr, "%f %f\n", LIST_GET(&tris, trii).e[ei].pnl0p->pp->x,
		    LIST_GET(&tris, trii).e[ei].pnl0p->pp->y);
#endif

    /* connect all pairs of triangles that share an edge */
    for (trii = 0; trii < LIST_SIZE(&tris); trii++)
	for (trij = trii + 1; trij < LIST_SIZE(&tris); trij++)
	    connecttris(trii, trij);

    /* find first and last triangles */
    for (trii = 0; trii < LIST_SIZE(&tris); trii++)
	if (pointintri(trii, &eps[0]))
	    break;
    if (trii == LIST_SIZE(&tris)) {
	prerror("source point not in any triangle");
	free(dq.pnlps);
	free(pnlps);
	free(pnls);
	return -1;
    }
    ftrii = trii;
    for (trii = 0; trii < LIST_SIZE(&tris); trii++)
	if (pointintri(trii, &eps[1]))
	    break;
    if (trii == LIST_SIZE(&tris)) {
	prerror("destination point not in any triangle");
	free(dq.pnlps);
	free(pnlps);
	free(pnls);
	return -1;
    }
    ltrii = trii;

    /* mark the strip of triangles from eps[0] to eps[1] */
    if (!marktripath(ftrii, ltrii)) {
	prerror("cannot find triangle path");
	free(dq.pnlps);
	free(pnlps);
	free(pnls);
	/* a straight line is better than failing */
	if (growops(2) != 0)
		return -2;
	output->pn = 2;
	ops[0] = eps[0], ops[1] = eps[1];
	output->ps = ops;
	return 0;
    }

    /* if endpoints in same triangle, use a single line */
    if (ftrii == ltrii) {
	free(dq.pnlps);
	free(pnlps);
	free(pnls);
	if (growops(2) != 0)
		return -2;
	output->pn = 2;
	ops[0] = eps[0], ops[1] = eps[1];
	output->ps = ops;
	return 0;
    }

    /* build funnel and shortest path linked list (in add2dq) */
    epnls[0].pp = &eps[0], epnls[0].link = NULL;
    epnls[1].pp = &eps[1], epnls[1].link = NULL;
    add2dq(&dq, DQ_FRONT, &epnls[0]);
    dq.apex = dq.fpnlpi;
    trii = ftrii;
    while (trii != SIZE_MAX) {
	trip = LIST_AT(&tris, trii);
	trip->mark = 2;

	/* find the left and right points of the exiting edge */
	for (ei = 0; ei < 3; ei++)
	    if (trip->e[ei].right_index != SIZE_MAX && LIST_GET(&tris, trip->e[ei].right_index).mark == 1)
		break;
	if (ei == 3) {		/* in last triangle */
	    if (ccw(eps[1], *dq.pnlps[dq.fpnlpi]->pp,
		    *dq.pnlps[dq.lpnlpi]->pp) == ISCCW)
		lpnlp = dq.pnlps[dq.lpnlpi], rpnlp = &epnls[1];
	    else
		lpnlp = &epnls[1], rpnlp = dq.pnlps[dq.lpnlpi];
	} else {
	    pnlp = trip->e[(ei + 1) % 3].pnl1p;
	    if (ccw(*trip->e[ei].pnl0p->pp, *pnlp->pp,
		    *trip->e[ei].pnl1p->pp) == ISCCW)
		lpnlp = trip->e[ei].pnl1p, rpnlp = trip->e[ei].pnl0p;
	    else
		lpnlp = trip->e[ei].pnl0p, rpnlp = trip->e[ei].pnl1p;
	}

	/* update deque */
	if (trii == ftrii) {
	    add2dq(&dq, DQ_BACK, lpnlp);
	    add2dq(&dq, DQ_FRONT, rpnlp);
	} else {
	    if (dq.pnlps[dq.fpnlpi] != rpnlp
		&& dq.pnlps[dq.lpnlpi] != rpnlp) {
		/* add right point to deque */
		size_t splitindex = finddqsplit(&dq, rpnlp);
		splitdq(&dq, DQ_BACK, splitindex);
		add2dq(&dq, DQ_FRONT, rpnlp);
		/* if the split is behind the apex, then reset apex */
		if (splitindex > dq.apex)
		    dq.apex = splitindex;
	    } else {
		/* add left point to deque */
		size_t splitindex = finddqsplit(&dq, lpnlp);
		splitdq(&dq, DQ_FRONT, splitindex);
		add2dq(&dq, DQ_BACK, lpnlp);
		/* if the split is in front of the apex, then reset apex */
		if (splitindex < dq.apex)
		    dq.apex = splitindex;
	    }
	}
	trii = SIZE_MAX;
	for (ei = 0; ei < 3; ei++)
	    if (trip->e[ei].right_index != SIZE_MAX && LIST_GET(&tris, trip->e[ei].right_index).mark == 1) {
		trii = trip->e[ei].right_index;
		break;
	    }
    }

#if defined(DEBUG) && DEBUG >= 1
    fprintf(stderr, "polypath");
    for (pnlp = &epnls[1]; pnlp; pnlp = pnlp->link)
	fprintf(stderr, " %f %f", pnlp->pp->x, pnlp->pp->y);
    fprintf(stderr, "\n");
#endif

    free(dq.pnlps);
    size_t i;
    for (i = 0, pnlp = &epnls[1]; pnlp; pnlp = pnlp->link)
	i++;
    if (growops(i) != 0) {
	free(pnlps);
	free(pnls);
	return -2;
    }
    output->pn = i;
    for (i = i - 1, pnlp = &epnls[1]; pnlp; i--, pnlp = pnlp->link)
	ops[i] = *pnlp->pp;
    output->ps = ops;
    free(pnlps);
    free(pnls);

    return 0;
}

/* triangulate polygon */
static int triangulate(pointnlink_t **points, size_t point_count) {
	if (point_count > 3)
	{
		for (size_t pnli = 0; pnli < point_count; pnli++)
		{
			const size_t pnlip1 = (pnli + 1) % point_count;
			const size_t pnlip2 = (pnli + 2) % point_count;
			if (isdiagonal(pnli, pnlip2, points, point_count, point_indexer))
			{
				if (loadtriangle(points[pnli], points[pnlip1], points[pnlip2]) != 0)
					return -1;
				for (pnli = pnlip1; pnli < point_count - 1; pnli++)
					points[pnli] = points[pnli + 1];
				return triangulate(points, point_count - 1);
			}
		}
		prerror("triangulation failed");
    } 
	else {
		if (loadtriangle(points[0], points[1], points[2]) != 0)
			return -1;
	}

    return 0;
}

static int loadtriangle(pointnlink_t * pnlap, pointnlink_t * pnlbp,
			 pointnlink_t * pnlcp)
{
    triangle_t trip = {0};
    trip.e[0].pnl0p = pnlap, trip.e[0].pnl1p = pnlbp, trip.e[0].right_index = SIZE_MAX;
    trip.e[1].pnl0p = pnlbp, trip.e[1].pnl1p = pnlcp, trip.e[1].right_index = SIZE_MAX;
    trip.e[2].pnl0p = pnlcp, trip.e[2].pnl1p = pnlap, trip.e[2].right_index = SIZE_MAX;

    if (!LIST_TRY_APPEND(&tris, trip)) {
	prerror("cannot realloc tris");
	return -1;
    }

    return 0;
}

/* connect a pair of triangles at their common edge (if any) */
static void connecttris(size_t tri1, size_t tri2) {
    triangle_t *tri1p, *tri2p;
    int ei, ej;

    for (ei = 0; ei < 3; ei++) {
	for (ej = 0; ej < 3; ej++) {
	    tri1p = LIST_AT(&tris, tri1);
	    tri2p = LIST_AT(&tris, tri2);
	    if ((tri1p->e[ei].pnl0p->pp == tri2p->e[ej].pnl0p->pp &&
		 tri1p->e[ei].pnl1p->pp == tri2p->e[ej].pnl1p->pp) ||
		(tri1p->e[ei].pnl0p->pp == tri2p->e[ej].pnl1p->pp &&
		 tri1p->e[ei].pnl1p->pp == tri2p->e[ej].pnl0p->pp))
		tri1p->e[ei].right_index = tri2, tri2p->e[ej].right_index = tri1;
	}
    }
}

/* find and mark path from trii, to trij */
static bool marktripath(size_t trii, size_t trij) {
    int ei;

    if (LIST_GET(&tris, trii).mark)
	return false;
    LIST_AT(&tris, trii)->mark = 1;
    if (trii == trij)
	return true;
    for (ei = 0; ei < 3; ei++)
	if (LIST_GET(&tris, trii).e[ei].right_index != SIZE_MAX &&
	    marktripath(LIST_GET(&tris, trii).e[ei].right_index, trij))
	    return true;
    LIST_AT(&tris, trii)->mark = 0;
    return false;
}

/* add a new point to the deque, either front or back */
static void add2dq(deque_t *dq, int side, pointnlink_t *pnlp) {
    if (side == DQ_FRONT) {
	if (dq->lpnlpi >= dq->fpnlpi)
	    pnlp->link = dq->pnlps[dq->fpnlpi];	/* shortest path links */
	dq->fpnlpi--;
	dq->pnlps[dq->fpnlpi] = pnlp;
    } else {
	if (dq->lpnlpi >= dq->fpnlpi)
	    pnlp->link = dq->pnlps[dq->lpnlpi];	/* shortest path links */
	dq->lpnlpi++;
	dq->pnlps[dq->lpnlpi] = pnlp;
    }
}

static void splitdq(deque_t *dq, int side, size_t index) {
    if (side == DQ_FRONT)
	dq->lpnlpi = index;
    else
	dq->fpnlpi = index;
}

static size_t finddqsplit(const deque_t *dq, pointnlink_t *pnlp) {
    for (size_t index = dq->fpnlpi; index < dq->apex; index++)
	if (ccw(*dq->pnlps[index + 1]->pp, *dq->pnlps[index]->pp, *pnlp->pp) == ISCCW)
	    return index;
    for (size_t index = dq->lpnlpi; index > dq->apex; index--)
	if (ccw(*dq->pnlps[index - 1]->pp, *dq->pnlps[index]->pp, *pnlp->pp) == ISCW)
	    return index;
    return dq->apex;
}

static int pointintri(size_t trii, Ppoint_t *pp) {
    int ei, sum;

    for (ei = 0, sum = 0; ei < 3; ei++)
	if (ccw(*LIST_GET(&tris, trii).e[ei].pnl0p->pp,
	        *LIST_GET(&tris, trii).e[ei].pnl1p->pp, *pp) != ISCW)
	    sum++;
    return sum == 3 || sum == 0;
}

static int growops(size_t newopn) {
    if (newopn <= opn)
	return 0;
    Ppoint_t *new_ops = realloc(ops, POINTSIZE * newopn);
    if (new_ops == NULL) {
	prerror("cannot realloc ops");
	return -1;
    }
    ops = new_ops;
    opn = newopn;

    return 0;
}
