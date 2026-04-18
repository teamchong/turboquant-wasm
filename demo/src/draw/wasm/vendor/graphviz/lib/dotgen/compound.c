/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/


/* Module for clipping splines to cluster boxes.
 */

#include "config.h"

#include	<dotgen/dot.h>
#include	<math.h>
#include	<stdbool.h>
#include	<stddef.h>
#include	<util/agxbuf.h>
#include	<util/alloc.h>
#include	<util/gv_math.h>

/* Return point where line segment [pp,cp] intersects
 * the box bp. Assume cp is outside the box, and pp is
 * on or in the box. 
 */
static pointf boxIntersectf(pointf pp, pointf cp, boxf * bp)
{
    pointf ipp;
    double ppx = pp.x;
    double ppy = pp.y;
    double cpx = cp.x;
    double cpy = cp.y;

    const pointf ll = bp->LL;
    const pointf ur = bp->UR;
    if (cp.x < ll.x) {
	ipp.x = ll.x;
	ipp.y = pp.y + round((ipp.x - ppx) * (ppy - cpy) / (ppx - cpx));
	if (ipp.y >= ll.y && ipp.y <= ur.y)
	    return ipp;
    }
    if (cp.x > ur.x) {
	ipp.x = ur.x;
	ipp.y = pp.y + round((ipp.x - ppx) * (ppy - cpy) / (ppx - cpx));
	if (ipp.y >= ll.y && ipp.y <= ur.y)
	    return ipp;
    }
    if (cp.y < ll.y) {
	ipp.y = ll.y;
	ipp.x = pp.x + round((ipp.y - ppy) * (ppx - cpx) / (ppy - cpy));
	if (ipp.x >= ll.x && ipp.x <= ur.x)
	    return ipp;
    }
    if (cp.y > ur.y) {
	ipp.y = ur.y;
	ipp.x = pp.x + round((ipp.y - ppy) * (ppx - cpx) / (ppy - cpy));
	if (ipp.x >= ll.x && ipp.x <= ur.x)
	    return ipp;
    }

    /* failure */
    agerrorf(
          "segment [(%.5g, %.5g),(%.5g,%.5g)] does not intersect box "
          "ll=(%.5g,%.5g),ur=(%.5g,%.5g)\n", pp.x, pp.y, cp.x, cp.y, ll.x, ll.y,
          ur.x, ur.y);
    assert(0);

    return ipp;
}

/// returns true if p is on or in box bb
static bool inBoxf(pointf p, boxf *bb) {
    return INSIDE(p, *bb);
}

/* Returns subgraph with given name.
 * Returns NULL if no name is given, or subgraph of
 * that name does not exist.
 */
static graph_t *getCluster(char *cluster_name, Dt_t *map) {
    Agraph_t* sg;

    if (!cluster_name || *cluster_name == '\0')
	return NULL;
    sg = findCluster (map, cluster_name);
    if (sg == NULL) {
	agwarningf("cluster named %s not found\n", cluster_name);
    }
    return sg;
}

/* The following functions are derived from pp. 411-415 (pp. 791-795)
 * of Graphics Gems. In the code there, they use a SGN function to
 * count crossings. This doesn't seem to handle certain special cases,
 * as when the last point is on the line. It certainly didn't work
 * for us when we used int values; see bug 145. We needed to use `fcmp` instead.
 *
 * Possibly unnecessary with double values, but harmless.
 */

/* Return the number of times the Bézier control polygon crosses
 * the vertical line x = xcoord.
 */
static int countVertCross(pointf * pts, double xcoord)
{
    int i;
    int sign, old_sign;
    int num_crossings = 0;

    sign = fcmp(pts[0].x, xcoord);
    if (sign == 0)
	num_crossings++;
    for (i = 1; i <= 3; i++) {
	old_sign = sign;
	sign = fcmp(pts[i].x, xcoord);
	if (sign != old_sign && old_sign != 0)
	    num_crossings++;
    }
    return num_crossings;
}

/* Return the number of times the Bézier control polygon crosses
 * the horizontal line y = ycoord.
 */
static int countHorzCross(pointf * pts, double ycoord)
{
    int i;
    int sign, old_sign;
    int num_crossings = 0;

    sign = fcmp(pts[0].y, ycoord);
    if (sign == 0)
	num_crossings++;
    for (i = 1; i <= 3; i++) {
	old_sign = sign;
	sign = fcmp(pts[i].y, ycoord);
	if (sign != old_sign && old_sign != 0)
	    num_crossings++;
    }
    return num_crossings;
}

/* Given 4 Bézier control points pts, corresponding to the portion
 * of an initial spline with path parameter in the range
 * 0.0 <= tmin <= t <= tmax <= 1.0, return t where the spline 
 * first crosses a vertical line segment
 * [(xcoord,ymin),(xcoord,ymax)]. Return -1 if not found.
 * This is done by binary subdivision. 
 */
static double
findVertical(pointf * pts, double tmin, double tmax,
	     double xcoord, double ymin, double ymax)
{
    pointf Left[4];
    pointf Right[4];
    double t;
    int no_cross;

    if (tmin == tmax)
	return tmin;

    no_cross = countVertCross(pts, xcoord);
    if (no_cross == 0)
	return -1.0;

    /* if 1 crossing and on the line x == xcoord (within 0.005 point) */
    if (no_cross == 1 && fabs(pts[3].x - xcoord) <= 0.005) {
	if (ymin <= pts[3].y && pts[3].y <= ymax) {
	    return tmax;
	} else
	    return -1.0;
    }

    /* split the Bézier into halves, trying the first half first. */
    Bezier(pts, 0.5, Left, Right);
    t = findVertical(Left, tmin, (tmin + tmax) / 2.0, xcoord, ymin, ymax);
    if (t >= 0.0)
	return t;
    return findVertical(Right, (tmin + tmax) / 2.0, tmax, xcoord, ymin,
			ymax);

}

/* Given 4 Bézier control points pts, corresponding to the portion
 * of an initial spline with path parameter in the range
 * 0.0 <= tmin <= t <= tmax <= 1.0, return t where the spline 
 * first crosses a horizontal line segment
 * [(xmin,ycoord),(xmax,ycoord)]. Return -1 if not found.
 * This is done by binary subdivision. 
 */
static double
findHorizontal(pointf * pts, double tmin, double tmax,
	       double ycoord, double xmin, double xmax)
{
    pointf Left[4];
    pointf Right[4];
    double t;
    int no_cross;

    if (tmin == tmax)
	return tmin;

    no_cross = countHorzCross(pts, ycoord);
    if (no_cross == 0)
	return -1.0;

    /* if 1 crossing and on the line y == ycoord (within 0.005 point) */
    if (no_cross == 1 && fabs(pts[3].y - ycoord) <= 0.005) {
	if (xmin <= pts[3].x && pts[3].x <= xmax) {
	    return tmax;
	} else
	    return -1.0;
    }

    /* split the Bézier into halves, trying the first half first. */
    Bezier(pts, 0.5, Left, Right);
    t = findHorizontal(Left, tmin, (tmin + tmax) / 2.0, ycoord, xmin,
		       xmax);
    if (t >= 0.0)
	return t;
    return findHorizontal(Right, (tmin + tmax) / 2.0, tmax, ycoord, xmin,
			  xmax);
}

/* Given four spline control points and a box,
 * find the shortest portion of the spline from
 * pts[0] to the intersection with the box, if any.
 * If an intersection is found, the four points are stored in pts[0..3]
 * with pts[3] being on the box, and 1 is returned. Otherwise, pts
 * is left unchanged and 0 is returned.
 */
static int splineIntersectf(pointf * pts, boxf * bb)
{
    double tmin = 2.0;
    const pointf origpts[] = {pts[0], pts[1], pts[2], pts[3]};

    double t = findVertical(pts, 0.0, 1.0, bb->LL.x, bb->LL.y, bb->UR.y);
    if (t >= 0 && t < tmin) {
	Bezier(origpts, t, pts, NULL);
	tmin = t;
    }
    t = findVertical(pts, 0.0, MIN(1.0, tmin), bb->UR.x, bb->LL.y, bb->UR.y);
    if (t >= 0 && t < tmin) {
	Bezier(origpts, t, pts, NULL);
	tmin = t;
    }
    t = findHorizontal(pts, 0.0, MIN(1.0, tmin), bb->LL.y, bb->LL.x, bb->UR.x);
    if (t >= 0 && t < tmin) {
	Bezier(origpts, t, pts, NULL);
	tmin = t;
    }
    t = findHorizontal(pts, 0.0, MIN(1.0, tmin), bb->UR.y, bb->LL.x, bb->UR.x);
    if (t >= 0 && t < tmin) {
	Bezier(origpts, t, pts, NULL);
	tmin = t;
    }

    return tmin < 2.0 ? 1 : 0;
}

/* If edge e has a cluster head and/or cluster tail,
 * clip spline to outside of cluster. 
 * Requirement: spline is composed of only one part, 
 * with n control points where n >= 4 and n (mod 3) = 1.
 * If edge has arrowheads, reposition them.
 */
static void makeCompoundEdge(edge_t *e, Dt_t *clustMap) {
    size_t starti = 0, endi = 0; // index of first and last control point

    /* find head and tail target clusters, if defined */
    graph_t *lh = getCluster(agget(e, "lhead"), clustMap); // cluster containing head
    graph_t *lt = getCluster(agget(e, "ltail"), clustMap); // cluster containing tail
    if (!lt && !lh)
	return;
    if (!ED_spl(e)) return;

    /* at present, we only handle single spline case */
    if (ED_spl(e)->size > 1) {
	agwarningf("%s -> %s: spline size > 1 not supported\n",
	      agnameof(agtail(e)), agnameof(aghead(e)));
	return;
    }
    bezier *bez = ED_spl(e)->list; // original Bézier for e
    const size_t size = bez->size;

    node_t *head = aghead(e);
    node_t *tail = agtail(e);

    /* allocate new Bézier */
    bezier nbez = {0}; // new Bézier for `e`
    nbez.eflag = bez->eflag;
    nbez.sflag = bez->sflag;

    /* if Bézier has four points, almost collinear,
     * make line - unimplemented optimization?
     */

    /* If head cluster defined, find first Bézier
     * crossing head cluster, and truncate spline to
     * box edge.
     * Otherwise, leave end alone.
     */
    bool fixed = false;
    if (lh) {
	boxf *bb = &GD_bb(lh);
	if (!inBoxf(ND_coord(head), bb)) {
	    agwarningf("%s -> %s: head not inside head cluster %s\n",
		  agnameof(agtail(e)), agnameof(aghead(e)), agget(e, "lhead"));
	} else {
	    /* If first control point is in bb, degenerate case. Spline
	     * reduces to four points between the arrow head and the point 
	     * where the segment between the first control point and arrow head 
	     * crosses box.
	     */
	    if (inBoxf(bez->list[0], bb)) {
		if (inBoxf(ND_coord(tail), bb)) {
		    agwarningf(
			  "%s -> %s: tail is inside head cluster %s\n",
			  agnameof(agtail(e)), agnameof(aghead(e)), agget(e, "lhead"));
		} else if (!inBoxf(bez->sp, bb)) {
		    assert(bez->sflag);	/* must be arrowhead on tail */
		    pointf p = boxIntersectf(bez->list[0], bez->sp, bb);
		    bez->list[3] = p;
		    bez->list[1] = mid_pointf(p, bez->sp);
		    bez->list[0] = mid_pointf(bez->list[1], bez->sp);
		    bez->list[2] = mid_pointf(bez->list[1], p);
		    if (bez->eflag)
			endi = arrowEndClip(e, bez->list,
					 starti, 0, &nbez, bez->eflag);
		    endi += 3;
		    fixed = true;
		}
	    } else {
		for (endi = 0; endi < size - 1; endi += 3) {
		    if (splineIntersectf(&bez->list[endi], bb))
			break;
		}
		if (endi == size - 1) {	/* no intersection */
		    assert(bez->eflag);
		    nbez.ep = boxIntersectf(bez->ep, bez->list[endi], bb);
		} else {
		    if (bez->eflag)
			endi =
			    arrowEndClip(e, bez->list,
					 starti, endi, &nbez, bez->eflag);
		    endi += 3;
		}
		fixed = true;
	    }
	}
    }
    if (!fixed) { // if no lh, or something went wrong, use original head
	endi = size - 1;
	if (bez->eflag)
	    nbez.ep = bez->ep;
    }

    /* If tail cluster defined, find last Bézier
     * crossing tail cluster, and truncate spline to
     * box edge.
     * Otherwise, leave end alone.
     */
    fixed = false;
    if (lt) {
	boxf *bb = &GD_bb(lt);
	if (!inBoxf(ND_coord(tail), bb)) {
	    agwarningf("%s -> %s: tail not inside tail cluster %s\n",
		agnameof(agtail(e)), agnameof(aghead(e)), agget(e, "ltail"));
	} else {
	    /* If last control point is in bb, degenerate case. Spline
	     * reduces to four points between arrow head, and the point
	     * where the segment between the last control point and the 
	     * arrow head crosses box.
	     */
	    if (inBoxf(bez->list[endi], bb)) {
		if (inBoxf(ND_coord(head), bb)) {
		    agwarningf(
			"%s -> %s: head is inside tail cluster %s\n",
		  	agnameof(agtail(e)), agnameof(aghead(e)), agget(e, "ltail"));
		} else if (bez->eflag && !inBoxf(nbez.ep, bb)) {
		    pointf p = boxIntersectf(bez->list[endi], nbez.ep, bb);
		    starti = endi - 3;
		    bez->list[starti] = p;
		    bez->list[starti + 2] = mid_pointf(p, nbez.ep);
		    bez->list[starti + 3] = mid_pointf(bez->list[starti + 2], nbez.ep);
		    bez->list[starti + 1] = mid_pointf(bez->list[starti + 2], p);
		    if (bez->sflag)
			starti = arrowStartClip(e, bez->list, starti,
				endi - 3, &nbez, bez->sflag);
		    fixed = true;
		}
	    } else {
		for (starti = endi; starti > 0; starti -= 3) {
		    pointf pts[4];
		    for (size_t i = 0; i < 4; i++)
			pts[i] = bez->list[starti - i];
		    if (splineIntersectf(pts, bb)) {
			for (size_t i = 0; i < 4; i++)
			    bez->list[starti - i] = pts[i];
			break;
		    }
		}
		if (starti == 0 && bez->sflag) {
		    nbez.sp = boxIntersectf(bez->sp, bez->list[starti], bb);
		} else if (starti != 0) {
		    starti -= 3;
		    if (bez->sflag)
			starti = arrowStartClip(e, bez->list, starti,
				endi - 3, &nbez, bez->sflag);
		}
		fixed = true;
	    }
	}
    }
    if (!fixed) { // if no lt, or something went wrong, use original tail
	/* Note: starti == 0 */
	if (bez->sflag)
	    nbez.sp = bez->sp;
    }

    /* complete Bézier, free garbage and attach new Bézier to edge 
     */
    nbez.size = endi - starti + 1;
    nbez.list = gv_calloc(nbez.size, sizeof(pointf));
    for (size_t i = 0, j = starti; i < nbez.size; i++, j++)
	nbez.list[i] = bez->list[j];
    free(bez->list);
    *ED_spl(e)->list = nbez;
}

void dot_compoundEdges(graph_t * g)
{
    edge_t *e;
    node_t *n;
    Dt_t* clustMap = mkClustMap (g);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    makeCompoundEdge(e, clustMap);
	}
    }
    dtclose(clustMap);
}
