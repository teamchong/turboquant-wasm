/// @file
/// @ingroup common_render
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
#include <common/geomprocs.h>
#include <common/render.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pathplan/pathplan.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/debug.h>
#include <util/gv_math.h>
#include <util/list.h>
#include <util/prisize_t.h>

static int nedges; ///< total no. of edges used in routing
static size_t nboxes; ///< total no. of boxes used in routing

static int routeinit;

static int checkpath(size_t, boxf *, path *);
static void printpath(path * pp);
#ifdef DEBUG
static void printboxes(size_t boxn, boxf *boxes) {
    pointf ll, ur;

    for (size_t bi = 0; bi < boxn; bi++) {
	ll = boxes[bi].LL, ur = boxes[bi].UR;
	agxbuf buf = {0};
	agxbprint(&buf, "%.0f %.0f %.0f %.0f pathbox", ll.x, ll.y, ur.x, ur.y);
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
    }
}

#if DEBUG > 1
static void psprintpolypts(Ppoint_t * p, int sz)
{
    int i;

    fprintf(stderr, "%%!\n");
    fprintf(stderr, "%% constraint poly\n");
    fprintf(stderr, "newpath\n");
    for (i = 0; i < sz; i++)
	fprintf(stderr, "%f %f %s\n", p[i].x, p[i].y, i == 0 ? "moveto" : "lineto");
    fprintf(stderr, "closepath stroke\n");
}
static void psprintpoint(point p)
{
    fprintf(stderr, "gsave\n");
    fprintf(stderr,
	    "newpath %d %d moveto %d %d 2 0 360 arc closepath fill stroke\n",
	    p.x, p.y, p.x, p.y);
    fprintf(stderr, "/Times-Roman findfont 4 scalefont setfont\n");
    fprintf(stderr, "%d %d moveto (\\(%d,%d\\)) show\n", p.x + 5, p.y + 5,
	    p.x, p.y);
    fprintf(stderr, "grestore\n");
}
static void psprintpointf(pointf p)
{
    fprintf(stderr, "gsave\n");
    fprintf(stderr,
	    "newpath %.5g %.5g moveto %.5g %.5g 2 0 360 arc closepath fill stroke\n",
	    p.x, p.y, p.x, p.y);
    fprintf(stderr, "/Times-Roman findfont 4 scalefont setfont\n");
    fprintf(stderr, "%.5g %.5g moveto (\\(%.5g,%.5g\\)) show\n", p.x + 5, p.y + 5,
	    p.x, p.y);
    fprintf(stderr, "grestore\n");
}
#endif

static void psprintspline(Ppolyline_t spl)
{
    LIST_APPEND(&Show_boxes, gv_strdup("%%!"));
    LIST_APPEND(&Show_boxes, gv_strdup("%% spline"));
    LIST_APPEND(&Show_boxes, gv_strdup("gsave 1 0 0 setrgbcolor newpath"));
    for (size_t i = 0; i < spl.pn; i++) {
	agxbuf buf = {0};
	agxbprint(&buf, "%f %f %s", spl.ps[i].x, spl.ps[i].y,
	  i == 0 ?  "moveto" : (i % 3 == 0 ? "curveto" : ""));
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
    }
    LIST_APPEND(&Show_boxes, gv_strdup("stroke grestore"));
}

static void psprintline(Ppolyline_t pl)
{
    LIST_APPEND(&Show_boxes, gv_strdup("%%!"));
    LIST_APPEND(&Show_boxes, gv_strdup("%% line"));
    LIST_APPEND(&Show_boxes, gv_strdup("gsave 0 0 1 setrgbcolor newpath"));
    for (size_t i = 0; i < pl.pn; i++) {
	agxbuf buf = {0};
	agxbprint(&buf, "%f %f %s", pl.ps[i].x, pl.ps[i].y,
		i == 0 ? "moveto" : "lineto");
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
    }
    LIST_APPEND(&Show_boxes, gv_strdup("stroke grestore"));
}

static void psprintpoly(Ppoly_t p)
{
    char*  pfx;

    LIST_APPEND(&Show_boxes, gv_strdup("%% poly list"));
    LIST_APPEND(&Show_boxes, gv_strdup("gsave 0 1 0 setrgbcolor"));
    for (size_t bi = 0; bi < p.pn; bi++) {
	const pointf tail = p.ps[bi];
	const pointf head = p.ps[(bi + 1) % p.pn];
	if (fabs(tail.x - head.x) < 1 && fabs(tail.y - head.y) < 1) pfx = "%%";
	else pfx ="";
	agxbuf buf = {0};
	agxbprint(&buf, "%s%.0f %.0f %.0f %.0f makevec", pfx, tail.x, tail.y, head.x,
	          head.y);
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
    }
    LIST_APPEND(&Show_boxes, gv_strdup("grestore"));
}

static void psprintboxes(size_t boxn, boxf *boxes) {
    pointf ll, ur;

    LIST_APPEND(&Show_boxes, gv_strdup("%% box list"));
    LIST_APPEND(&Show_boxes, gv_strdup("gsave 0 1 0 setrgbcolor"));
    for (size_t bi = 0; bi < boxn; bi++) {
	ll = boxes[bi].LL, ur = boxes[bi].UR;
	agxbuf buf = {0};
	agxbprint(&buf, "newpath\n%.0f %.0f moveto", ll.x, ll.y);
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
	agxbprint(&buf, "%.0f %.0f lineto", ll.x, ur.y);
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
	agxbprint(&buf, "%.0f %.0f lineto", ur.x, ur.y);
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
	agxbprint(&buf, "%.0f %.0f lineto", ur.x, ll.y);
	LIST_APPEND(&Show_boxes, agxbdisown(&buf));
	LIST_APPEND(&Show_boxes, gv_strdup("closepath stroke"));
    }
    LIST_APPEND(&Show_boxes, gv_strdup("grestore"));
}

static void psprintinit (int begin)
{
    if (begin)
	LIST_APPEND(&Show_boxes, gv_strdup("dbgstart"));
    else
	LIST_APPEND(&Show_boxes, gv_strdup("grestore"));
}

static bool debugleveln(edge_t* realedge, int i)
{
    return GD_showboxes(agraphof(aghead(realedge))) == i ||
	    GD_showboxes(agraphof(agtail(realedge))) == i ||
	    ED_showboxes(realedge) == i ||
	    ND_showboxes(aghead(realedge)) == i ||
	    ND_showboxes(agtail(realedge)) == i;
}
#endif  /* DEBUG */

/// Given a simple (ccw) polygon, route an edge from tp to hp.
pointf *simpleSplineRoute(pointf tp, pointf hp, Ppoly_t poly, size_t *n_spl_pts,
                          int polyline) {
    Ppolyline_t pl, spl;
    Ppoint_t eps[2];

    eps[0].x = tp.x;
    eps[0].y = tp.y;
    eps[1].x = hp.x;
    eps[1].y = hp.y;
    if (Pshortestpath(&poly, eps, &pl) < 0)
        return NULL;

    if (polyline)
	make_polyline (pl, &spl);
    else {
	// polygon edges passed to Proutespline
	Pedge_t *edges = gv_calloc(poly.pn, sizeof(Pedge_t));
	for (size_t i = 0; i < poly.pn; i++) {
	    edges[i].a = poly.ps[i];
	    edges[i].b = poly.ps[(i + 1) % poly.pn];
	}
	if (Proutespline(edges, poly.pn, pl, (Pvector_t[2]){0}, &spl) < 0) {
            free(edges);
            return NULL;
	}
	free(edges);
    }

    pointf *ps = calloc(spl.pn, sizeof(ps[0]));
    if (ps == NULL) {
	agerrorf("cannot allocate ps\n");
	return NULL;
    }
    for (size_t i = 0; i < spl.pn; i++) {
        ps[i] = spl.ps[i];
    }
    *n_spl_pts = spl.pn;
    return ps;
}

/** Data initialized once until matching call to routeplineterm
 * Allows recursive calls to dot
 */
int
routesplinesinit(void)
{
    if (++routeinit > 1) return 0;
#ifdef DEBUG
    LIST_FREE(&Show_boxes);
#endif
    nedges = 0;
    nboxes = 0;
    if (Verbose)
	start_timer();
    return 0;
}

void routesplinesterm(void)
{
    if (--routeinit > 0) return;
    GV_DEBUG("routesplines: %d edges, %" PRISIZE_T " boxes %.2f sec",
		nedges, nboxes, elapsed_sec());
}

static void limitBoxes(boxf *boxes, size_t boxn, const pointf *pps, size_t pn,
                       double delta) {
    double t;
    pointf sp[4];
    const double num_div = delta * (double)boxn;

    for (size_t splinepi = 0; splinepi + 3 < pn; splinepi += 3) {
	for (double si = 0; si <= num_div; si++) {
	    t = si / num_div;
	    sp[0] = pps[splinepi];
	    sp[1] = pps[splinepi + 1];
	    sp[2] = pps[splinepi + 2];
	    sp[3] = pps[splinepi + 3];
	    sp[0].x += t * (sp[1].x - sp[0].x);
	    sp[0].y += t * (sp[1].y - sp[0].y);
	    sp[1].x += t * (sp[2].x - sp[1].x);
	    sp[1].y += t * (sp[2].y - sp[1].y);
	    sp[2].x += t * (sp[3].x - sp[2].x);
	    sp[2].y += t * (sp[3].y - sp[2].y);
	    sp[0].x += t * (sp[1].x - sp[0].x);
	    sp[0].y += t * (sp[1].y - sp[0].y);
	    sp[1].x += t * (sp[2].x - sp[1].x);
	    sp[1].y += t * (sp[2].y - sp[1].y);
	    sp[0].x += t * (sp[1].x - sp[0].x);
	    sp[0].y += t * (sp[1].y - sp[0].y);
	    for (size_t bi = 0; bi < boxn; bi++) {
/* this tested ok on 64bit machines, but on 32bit we need this FUDGE
 *     or graphs/directed/records.gv fails */
#define FUDGE .0001
		if (sp[0].y <= boxes[bi].UR.y+FUDGE && sp[0].y >= boxes[bi].LL.y-FUDGE) {
		    boxes[bi].LL.x = fmin(boxes[bi].LL.x, sp[0].x);
		    boxes[bi].UR.x = fmax(boxes[bi].UR.x, sp[0].x);
		}
	    }
	}
    }
}

#define INIT_DELTA 10 
#define LOOP_TRIES 15  /* number of times to try to limiting boxes to regain space, using smaller divisions */

/** Route a path using the path info in pp. This includes start and end points
 * plus a collection of contiguous boxes containing the terminal points. The
 * boxes are converted into a containing polygon. A shortest path is constructed
 * within the polygon from between the terminal points. If polyline is true,
 * this path is converted to a spline representation. Otherwise, we call the
 * path planner to convert the polyline into a smooth spline staying within the
 * polygon. In both cases, the function returns an array of the computed control
 * points. The number of these points is given in npoints.
 *
 * During cleanup, the function determines the x-extent of the spline in the box, so
 * the box can be shrunk to the minimum width. The extra space can then be used by other
 * edges. 
 *
 * If a catastrophic error, return NULL and npoints is 0.
 */
static pointf *routesplines_(path *pp, size_t *npoints, int polyline) {
    Ppoly_t poly;
    Ppolyline_t pl, spl;
    Ppoint_t eps[2];
    int prev, next;
    boxf *boxes;
    edge_t* realedge;
    bool flip;
    int loopcnt;
    bool unbounded;

    *npoints = 0;
    nedges++;
    nboxes += pp->nbox;

    for (realedge = pp->data;
	 realedge && ED_edge_type(realedge) != NORMAL;
	 realedge = ED_to_orig(realedge));
    if (!realedge) {
	agerrorf("in routesplines, cannot find NORMAL edge\n");
	return NULL;
    }

    boxes = pp->boxes;
    const size_t boxn = pp->nbox;

    if (checkpath(boxn, boxes, pp))
	return NULL;

#ifdef DEBUG
    if (debugleveln(realedge, 1))
	printboxes(boxn, boxes);
    if (debugleveln(realedge, 3)) {
	psprintinit(1);
	psprintboxes(boxn, boxes);
    }
#endif

    // vertices of polygon defined by boxes
    Ppoint_t *polypoints = gv_calloc(boxn * 8, sizeof(Ppoint_t));

    if (boxn > 1 && boxes[0].LL.y > boxes[1].LL.y) {
        flip = true;
	for (size_t bi = 0; bi < boxn; bi++) {
	    double v = boxes[bi].UR.y;
	    boxes[bi].UR.y = -1*boxes[bi].LL.y;
	    boxes[bi].LL.y = -v;
	}
    }
    else flip = false;

    size_t pi;
    if (agtail(realedge) != aghead(realedge)) {
	/* I assume that the path goes either down only or
	   up - right - down */
	size_t bi;
	for (bi = 0, pi = 0; bi < boxn; bi++) {
	    next = prev = 0;
	    if (bi > 0)
		prev = boxes[bi].LL.y > boxes[bi - 1].LL.y ? -1 : 1;
	    if (bi + 1 < boxn)
		next = boxes[bi + 1].LL.y > boxes[bi].LL.y ? 1 : -1;
	    if (prev != next) {
		if (next == -1 || prev == 1) {
		    polypoints[pi].x = boxes[bi].LL.x;
		    polypoints[pi++].y = boxes[bi].UR.y;
		    polypoints[pi].x = boxes[bi].LL.x;
		    polypoints[pi++].y = boxes[bi].LL.y;
		} else {
		    polypoints[pi].x = boxes[bi].UR.x;
		    polypoints[pi++].y = boxes[bi].LL.y;
		    polypoints[pi].x = boxes[bi].UR.x;
		    polypoints[pi++].y = boxes[bi].UR.y;
		}
	    }
	    else if (prev == 0) { /* single box */
		polypoints[pi].x = boxes[bi].LL.x;
		polypoints[pi++].y = boxes[bi].UR.y;
		polypoints[pi].x = boxes[bi].LL.x;
		polypoints[pi++].y = boxes[bi].LL.y;
	    } 
	    else {
		if (!(prev == -1 && next == -1)) {
		    free(polypoints);
		    agerrorf("in routesplines, illegal values of prev %d and next %d, line %d\n", prev, next, __LINE__);
		    return NULL;
		}
	    }
	}
	for (bi = boxn - 1; bi != SIZE_MAX; bi--) {
	    next = prev = 0;
	    if (bi + 1 < boxn)
		prev = boxes[bi].LL.y > boxes[bi + 1].LL.y ? -1 : 1;
	    if (bi > 0)
		next = boxes[bi - 1].LL.y > boxes[bi].LL.y ? 1 : -1;
	    if (prev != next) {
		if (next == -1 || prev == 1 ) {
		    polypoints[pi].x = boxes[bi].LL.x;
		    polypoints[pi++].y = boxes[bi].UR.y;
		    polypoints[pi].x = boxes[bi].LL.x;
		    polypoints[pi++].y = boxes[bi].LL.y;
		} else {
		    polypoints[pi].x = boxes[bi].UR.x;
		    polypoints[pi++].y = boxes[bi].LL.y;
		    polypoints[pi].x = boxes[bi].UR.x;
		    polypoints[pi++].y = boxes[bi].UR.y;
		}
	    } 
	    else if (prev == 0) { /* single box */
		polypoints[pi].x = boxes[bi].UR.x;
		polypoints[pi++].y = boxes[bi].LL.y;
		polypoints[pi].x = boxes[bi].UR.x;
		polypoints[pi++].y = boxes[bi].UR.y;
	    }
	    else {
		if (!(prev == -1 && next == -1)) {
		    /* it went badly, e.g. degenerate box in boxlist */
		    free(polypoints);
		    agerrorf("in routesplines, illegal values of prev %d and next %d, line %d\n", prev, next, __LINE__);
		    return NULL; /* for correctness sake, it's best to just stop */
		}
		polypoints[pi].x = boxes[bi].UR.x;
		polypoints[pi++].y = boxes[bi].LL.y;
		polypoints[pi].x = boxes[bi].UR.x;
		polypoints[pi++].y = boxes[bi].UR.y;
		polypoints[pi].x = boxes[bi].LL.x;
		polypoints[pi++].y = boxes[bi].UR.y;
		polypoints[pi].x = boxes[bi].LL.x;
		polypoints[pi++].y = boxes[bi].LL.y;
	    }
	}
    }
    else {
	free(polypoints);
	agerrorf("in routesplines, edge is a loop at %s\n", agnameof(aghead(realedge)));
	return NULL;
    }

    if (flip) {
	for (size_t bi = 0; bi < boxn; bi++) {
	    double v = boxes[bi].UR.y;
	    boxes[bi].UR.y = -1*boxes[bi].LL.y;
	    boxes[bi].LL.y = -v;
	}
	for (size_t i = 0; i < pi; i++)
	    polypoints[i].y *= -1;
    }

    static const double INITIAL_LLX = DBL_MAX;
    static const double INITIAL_URX = -DBL_MAX;
    for (size_t bi = 0; bi < boxn; bi++) {
	boxes[bi].LL.x = INITIAL_LLX;
	boxes[bi].UR.x = INITIAL_URX;
    }
    poly.ps = polypoints, poly.pn = pi;
    eps[0].x = pp->start.p.x, eps[0].y = pp->start.p.y;
    eps[1].x = pp->end.p.x, eps[1].y = pp->end.p.y;
    if (Pshortestpath(&poly, eps, &pl) < 0) {
	free(polypoints);
	agerrorf("in routesplines, Pshortestpath failed\n");
	return NULL;
    }
#ifdef DEBUG
    if (debugleveln(realedge, 3)) {
	psprintpoly(poly);
	psprintline(pl);
    }
#endif

    if (polyline) {
	make_polyline (pl, &spl);
    }
    else {
	Pedge_t *edges = gv_calloc(poly.pn, sizeof(Pedge_t));
	for (size_t edgei = 0; edgei < poly.pn; edgei++) {
	    edges[edgei].a = polypoints[edgei];
	    edges[edgei].b = polypoints[(edgei + 1) % poly.pn];
	}
	Pvector_t evs[2] = {0};
	if (pp->start.constrained) {
	    evs[0].x = cos(pp->start.theta);
	    evs[0].y = sin(pp->start.theta);
	}
	if (pp->end.constrained) {
	    evs[1].x = -cos(pp->end.theta);
	    evs[1].y = -sin(pp->end.theta);
	}

	if (Proutespline(edges, poly.pn, pl, evs, &spl) < 0) {
	    free(edges);
	    free(polypoints);
	    agerrorf("in routesplines, Proutespline failed\n");
	    return NULL;
	}
	free(edges);
#ifdef DEBUG
	if (debugleveln(realedge, 3)) {
	    psprintspline(spl);
	    psprintinit(0);
	}
#endif
    }
    pointf *ps = calloc(spl.pn, sizeof(ps[0]));
    if (ps == NULL) {
	free(polypoints);
	agerrorf("cannot allocate ps\n");
	return NULL;  /* Bailout if no memory left */
    }

    unbounded = true;
    for (size_t splinepi = 0; splinepi < spl.pn; splinepi++) {
	ps[splinepi] = spl.ps[splinepi];
    }

    {
	// does the spline go straight left/right?
	bool is_horizontal = spl.pn > 0;
	for (size_t i = 0; i < spl.pn; ++i) {
	    if (fabs(ps[0].y - ps[i].y) > FUDGE) {
		is_horizontal = false;
		break;
	    }
	}
	if (is_horizontal) {
	    GV_INFO("spline [%.03f, %.03f] -- [%.03f, %.03f] is horizontal;"
	            " will be trivially bounded", ps[0].x, ps[0].y, ps[spl.pn - 1].x,
	            ps[spl.pn - 1].y);
	}

	// does the spline go straight up/down?
	bool is_vertical = spl.pn > 0;
	for (size_t i = 0; i < spl.pn; ++i) {
	    if (fabs(ps[0].x - ps[i].x) > FUDGE) {
		is_vertical = false;
		break;
	    }
	}
	if (is_vertical) {
	    GV_INFO("spline [%.03f, %.03f] -- [%.03f, %.03f] is vertical;"
	            " will be trivially bounded", ps[0].x, ps[0].y, ps[spl.pn - 1].x,
	            ps[spl.pn - 1].y);
	}

	// if the spline is horizontal or vertical, the `limitBoxes` loop will not
	// converge, but we know the expected outcome trivially
	if (is_horizontal || is_vertical) {
	    for (size_t i = 0; i < boxn; ++i) {
		boxes[i].LL.x = ps[0].x;
		boxes[i].UR.x = ps[0].x;
	    }
	    unbounded = false;
	}
    }

    double delta = INIT_DELTA;

    for (loopcnt = 0; unbounded && loopcnt < LOOP_TRIES; loopcnt++) {
	limitBoxes(boxes, boxn, ps, spl.pn, delta);

    /* The following check is necessary because if a box is not very 
     * high, it is possible that the sampling above might miss it.
     * Therefore, we make the sample finer until all boxes have
     * valid values. cf. bug 456.
     */
	size_t bi;
	for (bi = 0; bi < boxn; bi++) {
	/* these fp equality tests are used only to detect if the
	 * values have been changed since initialization - ok */
	    if (is_exactly_equal(boxes[bi].LL.x, INITIAL_LLX) ||
	        is_exactly_equal(boxes[bi].UR.x, INITIAL_URX)) {
		delta *= 2; /* try again with a finer interval */
		break;
	    }
	}
	if (bi == boxn)
	    unbounded = false;
    }
    if (unbounded) {  
	/* Either an extremely short, even degenerate, box, or some failure with the path
         * planner causing the spline to miss some boxes. In any case, use the shortest path 
	 * to bound the boxes. This will probably mean a bad edge, but we avoid an infinite
	 * loop and we can see the bad edge, and even use the showboxes scaffolding.
	 */
	Ppolyline_t polyspl;
	agwarningf("Unable to reclaim box space in spline routing for edge \"%s\" -> \"%s\". Something is probably seriously wrong.\n", agnameof(agtail(realedge)), agnameof(aghead(realedge)));
	make_polyline (pl, &polyspl);
	limitBoxes(boxes, boxn, polyspl.ps, polyspl.pn, INIT_DELTA);
    }

    *npoints = spl.pn;

#ifdef DEBUG
    if (GD_showboxes(agraphof(aghead(realedge))) == 2 ||
	GD_showboxes(agraphof(agtail(realedge))) == 2 ||
	ED_showboxes(realedge) == 2 ||
	ND_showboxes(aghead(realedge)) == 2 ||
	ND_showboxes(agtail(realedge)) == 2)
	printboxes(boxn, boxes);
#endif

    free(polypoints);
    return ps;
}

pointf *routesplines(path *pp, size_t *npoints) {
  return routesplines_(pp, npoints, 0);
}

pointf *routepolylines(path *pp, size_t *npoints) {
  return routesplines_(pp, npoints, 1);
}

static double overlap(double i0, double i1, double j0, double j1) {
  if (i1 <= j0)
    return 0;
  if (i0 >= j1)
    return 0;

  // does the first interval subsume the second?
  if (i0 <= j0 && i1 >= j1)
    return i1 - i0;
  // does the second interval subsume the first?
  if (j0 <= i0 && j1 >= i1)
    return j1 - j0;

  if (j0 <= i0 && i0 <= j1)
    return j1 - i0;
  assert(j0 <= i1 && i1 <= j1);
  return i1 - j0;
}


/*
 * repairs minor errors in the boxpath, such as boxes not joining
 * or slightly intersecting.  it's sort of the equivalent of the
 * audit process in the 5E control program - if you've given up on
 * fixing all the bugs, at least try to engineer around them!
 * in postmodern CS, we could call this "self-healing code."
 *
 * Return 1 on failure; 0 on success.
 */
static int checkpath(size_t boxn, boxf *boxes, path *thepath) {
    boxf *ba, *bb;
    int errs, l, r, d, u;

    /* remove degenerate boxes. */
    size_t i = 0;
    for (size_t bi = 0; bi < boxn; bi++) {
	if (fabs(boxes[bi].LL.y - boxes[bi].UR.y) < .01)
	    continue;
	if (fabs(boxes[bi].LL.x - boxes[bi].UR.x) < .01)
	    continue;
	boxes[i] = boxes[bi];
	i++;
    }
    boxn = i;

    ba = &boxes[0];
    if (ba->LL.x > ba->UR.x || ba->LL.y > ba->UR.y) {
	agerrorf("in checkpath, box 0 has LL coord > UR coord\n");
	printpath(thepath);
	return 1;
    }
    for (size_t bi = 0; bi + 1 < boxn; bi++) {
	ba = &boxes[bi], bb = &boxes[bi + 1];
	if (bb->LL.x > bb->UR.x || bb->LL.y > bb->UR.y) {
	    agerrorf("in checkpath, box %" PRISIZE_T " has LL coord > UR coord\n",
	             bi + 1);
	    printpath(thepath);
	    return 1;
	}
	l = ba->UR.x < bb->LL.x ? 1 : 0;
	r = ba->LL.x > bb->UR.x ? 1 : 0;
	d = ba->UR.y < bb->LL.y ? 1 : 0;
	u = ba->LL.y > bb->UR.y ? 1 : 0;
	errs = l + r + d + u;
	if (errs > 0 && Verbose) {
	    fprintf(stderr, "in checkpath, boxes %" PRISIZE_T " and %" PRISIZE_T
	            " don't touch\n", bi, bi + 1);
	    printpath(thepath);
	}
	if (errs > 0) {
	    double xy;

	    if (l == 1)
		xy = ba->UR.x, ba->UR.x = bb->LL.x, bb->LL.x = xy, l = 0;
	    else if (r == 1)
		xy = ba->LL.x, ba->LL.x = bb->UR.x, bb->UR.x = xy, r = 0;
	    else if (d == 1)
		xy = ba->UR.y, ba->UR.y = bb->LL.y, bb->LL.y = xy, d = 0;
	    else if (u == 1)
		xy = ba->LL.y, ba->LL.y = bb->UR.y, bb->UR.y = xy, u = 0;
	    for (int j = 0; j < errs - 1; j++) {
		if (l == 1)
		    xy = (ba->UR.x + bb->LL.x) / 2.0 + 0.5, ba->UR.x =
			bb->LL.x = xy, l = 0;
		else if (r == 1)
		    xy = (ba->LL.x + bb->UR.x) / 2.0 + 0.5, ba->LL.x =
			bb->UR.x = xy, r = 0;
		else if (d == 1)
		    xy = (ba->UR.y + bb->LL.y) / 2.0 + 0.5, ba->UR.y =
			bb->LL.y = xy, d = 0;
		else if (u == 1)
		    xy = (ba->LL.y + bb->UR.y) / 2.0 + 0.5, ba->LL.y =
			bb->UR.y = xy, u = 0;
	    }
	}
	/* check for overlapping boxes */
	double xoverlap = overlap(ba->LL.x, ba->UR.x, bb->LL.x, bb->UR.x);
	double yoverlap = overlap(ba->LL.y, ba->UR.y, bb->LL.y, bb->UR.y);
	if (xoverlap > 0 && yoverlap > 0) {
	    if (xoverlap < yoverlap) {
		if (ba->UR.x - ba->LL.x > bb->UR.x - bb->LL.x) {
		    /* take space from ba */
		    if (ba->UR.x < bb->UR.x)
			ba->UR.x = bb->LL.x;
		    else
			ba->LL.x = bb->UR.x;
		} else {
		    /* take space from bb */
		    if (ba->UR.x < bb->UR.x)
			bb->LL.x = ba->UR.x;
		    else
			bb->UR.x = ba->LL.x;
		}
	    } else {		/* symmetric for y coords */
		if (ba->UR.y - ba->LL.y > bb->UR.y - bb->LL.y) {
		    /* take space from ba */
		    if (ba->UR.y < bb->UR.y)
			ba->UR.y = bb->LL.y;
		    else
			ba->LL.y = bb->UR.y;
		} else {
		    /* take space from bb */
		    if (ba->UR.y < bb->UR.y)
			bb->LL.y = ba->UR.y;
		    else
			bb->UR.y = ba->LL.y;
		}
	    }
	}
    }

    if (thepath->start.p.x < boxes[0].LL.x
	|| thepath->start.p.x > boxes[0].UR.x
	|| thepath->start.p.y < boxes[0].LL.y
	|| thepath->start.p.y > boxes[0].UR.y) {
	thepath->start.p.x = fmax(thepath->start.p.x, boxes[0].LL.x);
	thepath->start.p.x = fmin(thepath->start.p.x, boxes[0].UR.x);
	thepath->start.p.y = fmax(thepath->start.p.y, boxes[0].LL.y);
	thepath->start.p.y = fmin(thepath->start.p.y, boxes[0].UR.y);
    }
    if (thepath->end.p.x < boxes[boxn - 1].LL.x
	|| thepath->end.p.x > boxes[boxn - 1].UR.x
	|| thepath->end.p.y < boxes[boxn - 1].LL.y
	|| thepath->end.p.y > boxes[boxn - 1].UR.y) {
	thepath->end.p.x = fmax(thepath->end.p.x, boxes[boxn - 1].LL.x);
	thepath->end.p.x = fmin(thepath->end.p.x, boxes[boxn - 1].UR.x);
	thepath->end.p.y = fmax(thepath->end.p.y, boxes[boxn - 1].LL.y);
	thepath->end.p.y = fmin(thepath->end.p.y, boxes[boxn - 1].UR.y);
    }
    return 0;
}

static void printpath(path * pp)
{
    fprintf(stderr, "%" PRISIZE_T " boxes:\n", pp->nbox);
    for (size_t bi = 0; bi < pp->nbox; bi++)
	fprintf(stderr, "%" PRISIZE_T " (%.5g, %.5g), (%.5g, %.5g)\n", bi,
		pp->boxes[bi].LL.x, pp->boxes[bi].LL.y,
	       	pp->boxes[bi].UR.x, pp->boxes[bi].UR.y);
    fprintf(stderr, "start port: (%.5g, %.5g), tangent angle: %.5g, %s\n",
	    pp->start.p.x, pp->start.p.y, pp->start.theta,
	    pp->start.constrained ? "constrained" : "not constrained");
    fprintf(stderr, "end port: (%.5g, %.5g), tangent angle: %.5g, %s\n",
	    pp->end.p.x, pp->end.p.y, pp->end.theta,
	    pp->end.constrained ? "constrained" : "not constrained");
}

static pointf get_centroid(Agraph_t *g)
{
    pointf sum = {0.0, 0.0};

    sum.x = (GD_bb(g).LL.x + GD_bb(g).UR.x) / 2.0;
    sum.y = (GD_bb(g).LL.y + GD_bb(g).UR.y) / 2.0;
    return sum;
}

typedef LIST(node_t *) nodes_t;

static void nodes_delete(nodes_t *pvec) {
  if (pvec != NULL) {
    LIST_FREE(pvec);
  }
  free(pvec);
}

typedef LIST(nodes_t *) cycles_t;

static bool cycle_contains_edge(nodes_t *cycle, edge_t *edge) {
	node_t* start = agtail(edge);
	node_t* end = aghead(edge);

	const size_t cycle_len = LIST_SIZE(cycle);

	for (size_t i=0; i < cycle_len; ++i) {
		const node_t *c_start = LIST_GET(cycle, i == 0 ? cycle_len - 1 : i - 1);
		const node_t *c_end = LIST_GET(cycle, i);

		if (c_start == start && c_end == end)
			return true;
	}


	return false;
}

static bool is_cycle_unique(cycles_t *cycles, nodes_t *cycle) {
	const size_t cycle_len = LIST_SIZE(cycle);
	size_t i; //node counter

	bool all_items_match;

	for (size_t c = 0; c < LIST_SIZE(cycles); ++c) {
		nodes_t *cur_cycle = LIST_GET(cycles, c);
		const size_t cur_cycle_len = LIST_SIZE(cur_cycle);

		//if all the items match in equal length cycles then we're not unique
		if (cur_cycle_len == cycle_len) {
			all_items_match = true;
			for (i=0; i < cur_cycle_len; ++i) {
				node_t *cur_cycle_item = LIST_GET(cur_cycle, i);
				if (!LIST_CONTAINS(cycle, cur_cycle_item)) {
					all_items_match = false;
					break;
				}
			}
			if (all_items_match)
				return false;
		}
	}

	return true;
}

static void dfs(graph_t *g, node_t *search, nodes_t *visited, node_t *end,
                cycles_t *cycles) {
	edge_t* e;
	node_t* n;

	if (LIST_CONTAINS(visited, search)) {
		if (search == end) {
			if (is_cycle_unique(cycles, visited)) {
				nodes_t *cycle = gv_alloc(sizeof(nodes_t));
				LIST_COPY(cycle, visited);
				LIST_APPEND(cycles, cycle);
			}
		}
	} else {
		LIST_APPEND(visited, search);
		for (e = agfstout(g, search); e; e = agnxtout(g, e)) {
			n = aghead(e);
			dfs(g, n, visited, end, cycles);
		}
		if (!LIST_IS_EMPTY(visited)) {
			LIST_DROP_BACK(visited);
		}
	}
}

// Returns a vec of vec of nodes (aka a vector of cycles)
static cycles_t find_all_cycles(graph_t *g) {
    node_t *n;

    // vector of vectors of nodes -- AKA cycles to delete
    cycles_t alloced_cycles = {.dtor = nodes_delete};
    cycles_t cycles = {.dtor = nodes_delete}; // vector of vectors of nodes AKA a vector of cycles

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
		nodes_t *cycle = gv_alloc(sizeof(nodes_t));
		// keep track of all items we allocate to clean up at the end of this function
		LIST_APPEND(&alloced_cycles, cycle);
		
		dfs(g, n, cycle, n, &cycles);
	}
	
	LIST_FREE(&alloced_cycles); // cycles contains copied vecs
    return cycles;
}

static nodes_t *find_shortest_cycle_with_edge(cycles_t *cycles, edge_t *edge,
                                              size_t min_size) {
	nodes_t *shortest = NULL;

	for (size_t c = 0; c < LIST_SIZE(cycles); ++c) {
		nodes_t *cycle = LIST_GET(cycles, c);
		size_t cycle_len = LIST_SIZE(cycle);

		if (cycle_len < min_size)
			continue;

		if (shortest == NULL || LIST_SIZE(shortest) > cycle_len) {
			if (cycle_contains_edge(cycle, edge)) {
				shortest = cycle;
			}
		}
	}
	return shortest;
}

static pointf get_cycle_centroid(graph_t *g, edge_t* edge)
{
	cycles_t cycles = find_all_cycles(g);

	//find the center of the shortest cycle containing this edge
	//cycles of length 2 do their own thing, we want 3 or
	nodes_t *cycle = find_shortest_cycle_with_edge(&cycles, edge, 3);
    pointf sum = {0.0, 0.0};

	if (cycle == NULL) {
		LIST_FREE(&cycles);
		return get_centroid(g);
	}

	double cnt = 0;
	for (size_t idx = 0; idx < LIST_SIZE(cycle); ++idx) {
		node_t *n = LIST_GET(cycle, idx);
		sum.x += ND_coord(n).x;
        sum.y += ND_coord(n).y;
        cnt++;
	}

	LIST_FREE(&cycles);

	sum.x /= cnt;
    sum.y /= cnt;
    return sum;
}

static void bend(pointf spl[4], pointf centroid)
{
    pointf  a;
    double  r;

    pointf midpt = mid_pointf(spl[0], spl[3]);
    double dist = DIST(spl[3], spl[0]);
    r = dist/5.0;
    {
        double vX = centroid.x - midpt.x;
        double vY = centroid.y - midpt.y;
        double magV = hypot(vX, vY);
	if (magV == 0) return;  /* if midpoint == centroid, don't divide by zero */
        a.x = midpt.x - vX / magV * r;      /* + would be closest point */
        a.y = midpt.y - vY / magV * r;
    }
    /* this can be improved */
    spl[1].x = spl[2].x = a.x;
    spl[1].y = spl[2].y = a.y;
}

// FIX: handle ports on boundary?
void 
makeStraightEdge(graph_t * g, edge_t * e, int et, splineInfo* sinfo)
{
    edge_t *e0;

    size_t e_cnt = 1;
    e0 = e;
    while (e0 != ED_to_virt(e0) && (e0 = ED_to_virt(e0))) e_cnt++;

    edge_t **edge_list = gv_calloc(e_cnt, sizeof(edge_t *));
    e0 = e;
    for (size_t i = 0; i < e_cnt; i++) {
	edge_list[i] = e0;
	e0 = ED_to_virt(e0);
    }
    assert(e_cnt <= INT_MAX);
    makeStraightEdges(g, edge_list, e_cnt, et, sinfo);
    free(edge_list);
}

void makeStraightEdges(graph_t *g, edge_t **edge_list, size_t e_cnt, int et,
                       splineInfo *sinfo) {
    pointf dumb[4];
    bool curved = et == EDGETYPE_CURVED;
    pointf del;

    edge_t *e = edge_list[0];
    node_t *n = agtail(e);
    node_t *head = aghead(e);
    dumb[1] = dumb[0] = add_pointf(ND_coord(n), ED_tail_port(e).p);
    dumb[2] = dumb[3] = add_pointf(ND_coord(head), ED_head_port(e).p);
    if (e_cnt == 1 || Concentrate) {
	if (curved) bend(dumb,get_cycle_centroid(g, edge_list[0]));
	clip_and_install(e, aghead(e), dumb, 4, sinfo);
	addEdgeLabels(e);
	return;
    }

    if (APPROXEQPT(dumb[0], dumb[3], MILLIPOINT)) {
	/* degenerate case */
	dumb[1] = dumb[0];
	dumb[2] = dumb[3];
	del.x = 0;
	del.y = 0;
    }
    else {
        pointf perp = {
          .x = dumb[0].y - dumb[3].y,
          .y = dumb[3].x - dumb[0].x
        };
	double l_perp = hypot(perp.x, perp.y);
	int xstep = GD_nodesep(g->root);
	assert(e_cnt - 1 <= INT_MAX);
	int dx = xstep * (int)(e_cnt - 1) / 2;
	dumb[1].x = dumb[0].x + dx * perp.x / l_perp;
	dumb[1].y = dumb[0].y + dx * perp.y / l_perp;
	dumb[2].x = dumb[3].x + dx * perp.x / l_perp;
	dumb[2].y = dumb[3].y + dx * perp.y / l_perp;
	del.x = -xstep * perp.x / l_perp;
	del.y = -xstep * perp.y / l_perp;
    }

    for (size_t i = 0; i < e_cnt; i++) {
	edge_t *e0 = edge_list[i];
	pointf dumber[4];
	if (aghead(e0) == head) {
	    for (size_t j = 0; j < 4; j++) {
		dumber[j] = dumb[j];
	    }
	} else {
	    for (size_t j = 0; j < 4; j++) {
		dumber[3 - j] = dumb[j];
	    }
	}
	if (et == EDGETYPE_PLINE) {
	    Ppoint_t pts[] = {dumber[0], dumber[1], dumber[2], dumber[3]};
	    Ppolyline_t spl, line = {.pn = 4, .ps = pts};
	    make_polyline (line, &spl);
	    clip_and_install(e0, aghead(e0), spl.ps, spl.pn, sinfo);
	}
	else
	    clip_and_install(e0, aghead(e0), dumber, 4, sinfo);

	addEdgeLabels(e0);
	dumb[1] = add_pointf(dumb[1], del);
	dumb[2] = add_pointf(dumb[2], del);
    }
}
