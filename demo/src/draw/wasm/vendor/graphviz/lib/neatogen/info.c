/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <neatogen/neato.h>
#include <stdio.h>
#include <neatogen/info.h>
#include <stddef.h>
#include <util/alloc.h>

Info_t *nodeInfo;		/* Array of node info */

/* compare:
 * returns -1 if p < q.p
 *          0 if p = q.p
 *          1 if p > q.p
 * if q if NULL, returns -1
 * Ordering is by angle from -pi/2 to 3pi/4.
 * For equal angles (which should not happen in our context)
 * ordering is by closeness to origin.
 */
static int compare(Point o, Point p, Point q) {
    double x0;
    double y0;
    double x1;
    double y1;
    double a, b;

    if (p.x == q.x && p.y == q.y)
	return 0;

    x0 = (double)p.x - (double)o.x;
    y0 = (double)p.y - (double)o.y;
    x1 = (double)q.x - (double)o.x;
    y1 = (double)q.y - (double)o.y;
    if (x0 >= 0.0) {
	if (x1 < 0.0)
	    return -1;
	else if (x0 > 0.0) {
	    if (x1 > 0.0) {
		a = y1 / x1;
		b = y0 / x0;
		if (b < a)
		    return -1;
		else if (b > a)
		    return 1;
		else if (x0 < x1)
		    return -1;
		else
		    return 1;
	    } else {		/* x1 == 0.0 */
		if (y1 > 0.0)
		    return -1;
		else
		    return 1;
	    }
	} else {		/* x0 == 0.0 */
	    if (x1 > 0.0) {
		if (y0 <= 0.0)
		    return -1;
		else
		    return 1;
	    } else {		/* x1 == 0.0 */
		if (y0 < y1) {
		    if (y1 <= 0.0)
			return 1;
		    else
			return -1;
		} else {
		    if (y0 <= 0.0)
			return -1;
		    else
			return 1;
		}
	    }
	}
    } else {
	if (x1 >= 0.0)
	    return 1;
	else {
	    a = y1 / x1;
	    b = y0 / x0;
	    if (b < a)
		return -1;
	    else if (b > a)
		return 1;
	    else if (x0 > x1)
		return -1;
	    else
		return 1;
	}
    }
}

void addVertex(Site * s, double x, double y)
{
    Info_t *ip;
    const Point origin_point = s->coord;

    ip = nodeInfo + s->sitenbr;

    const Point tmp = {.x = x, .y = y};

    size_t i;
    for (i = 0; i < ip->n_verts; ++i) {
	int cmp = compare(origin_point, tmp, ip->verts[i]);
	if (cmp == 0) { // we already know this vertex; ignore
	    return;
	}
	if (cmp < 0) { // found where to insert this vertex
	    break;
	}
    }

    ip->verts = gv_recalloc(ip->verts, ip->n_verts, ip->n_verts + 1,
                            sizeof(ip->verts[0]));
    // shuffle existing entries upwards to make room
    memmove(&ip->verts[i + 1], &ip->verts[i],
            (ip->n_verts - i) * sizeof(ip->verts[0]));
    ip->verts[i] = tmp;
    ++ip->n_verts;
}
