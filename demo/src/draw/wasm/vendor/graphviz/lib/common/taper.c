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

/*
 * Tapered edges, based on lines.ps written by Denis Moskowitz.
 */

#include "config.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <types.h>
#include <common/render.h>
#include <common/utils.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/list.h>
#include <util/prisize_t.h>

#ifdef DEBUG
enum { debug = 1 };
#else
enum { debug = 0 };
#endif

  /* sample point size; should be dynamic based on dpi or under user control */
#define BEZIERSUBDIVISION 20

  /* convert degrees to radians and vice versa */
#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif
#define D2R(d)    (M_PI*(d)/180.0)
#define R2D(r)    (180.0*(r)/M_PI)

static double currentmiterlimit = 10.0;

#define moveto(p,x,y) addto(p,x,y)
#define lineto(p,x,y) addto(p,x,y)

static void addto (stroke_t* p, double x, double y)
{
    pointf pt;

    p->vertices = gv_recalloc(p->vertices, p->nvertices, p->nvertices + 1,
                              sizeof(pointf));
    pt.x = x;
    pt.y = y;
    p->vertices[p->nvertices++] = pt;
}

/*
 * handle zeros
 */
static double myatan (double y, double x)
{
    double v;
    if (x == 0 && y == 0)
	return 0;
    v = atan2 (y, x);
    if (v >= 0) return v;
    return v + 2 * M_PI;
}

/*
 *  mod that accepts floats and makes negatives positive
 */
static double mymod (double original, double modulus)
{
    double v;
    if (original < 0 || original >= modulus) {
	v = -floor(original/modulus);
	return v * modulus + original;
    }
    return original;
}

typedef struct {
    double x;
    double y;
    double lengthsofar;
    char type;
    double dir;
    double lout;
    bool bevel;
    double dir2;
} pathpoint;

typedef LIST(pathpoint) vararr_t;

static void
insertArr (vararr_t* arr, pointf p, double l)
{
  pathpoint pt = {.x = p.x, .y = p.y, .lengthsofar = l};
  LIST_APPEND(arr, pt);
}

static void
printArr (vararr_t* arr, FILE* fp)
{
    fprintf(fp, "size %" PRISIZE_T "\n", LIST_SIZE(arr));
    for (size_t i = 0; i < LIST_SIZE(arr); i++) {
	pathpoint pt = LIST_GET(arr, i);
	fprintf(fp, "  [%" PRISIZE_T "] x %.02f y  %.02f d %.02f\n", i, pt.x, pt.y,
	        pt.lengthsofar);
    }
}

static double l2dist (pointf p0, pointf p1)
{
    double delx = p0.x - p1.x;
    double dely = p0.y - p1.y;
    return hypot(delx, dely);
}

/* analyze current path, creating pathpoints array
 * turn all curves into lines
 */
static vararr_t pathtolines(bezier *bez) {
    int step;
    double seglen, linelen = 0;
    vararr_t arr = {0};
    pointf p0, p1, V[4];
    const size_t n = bez->size;
    pointf* A = bez->list;

    insertArr(&arr, A[0], 0);
    V[3] = A[0];
    for (size_t i = 0; i + 3 < n; i += 3) {
	V[0] = V[3];
	for (size_t j = 1; j <= 3; j++)
	    V[j] = A[i + j];
	p0 = V[0];
	for (step = 1; step <= BEZIERSUBDIVISION; step++) {
	    p1 = Bezier(V, (double) step / BEZIERSUBDIVISION, NULL, NULL);
	    seglen = l2dist(p0, p1);
	    /* If initwid is large, this may never happen, so turn off. I assume this is to prevent
	     * too man points or too small a movement. Perhaps a better test can be made, but for now
	     * we turn it off. 
	     */
	    /* if (seglen > initwid/10) { */
		linelen += seglen;
		insertArr(&arr, p1, linelen);
	    /* } */
	    p0 = p1;
	}
    }
    if (debug) {
	printArr(&arr, stderr);
    }
    return arr;
}

static void drawbevel(double x, double lineout, bool forward, double dir,
                      double dir2, stroke_t *p) {
    double a2;

    if (forward) {
	a2 = dir2;
    } else {
	a2 = dir;
    }
    lineto (p, x + lineout*cos(a2), x + lineout*sin(a2));
}

typedef double (*radfunc_t) (double curlen, double totallen, double initwid);

/* taper:
 * Given a B-spline bez, returns a polygon that represents spline as a tapered
 * edge, starting with width initwid.
 * The radfunc determines the half-width along the curve. Typically, this will
 * decrease from initwid to 0 as the curlen goes from 0 to totallen.
 */
stroke_t taper(bezier *bez, radfunc_t radfunc, double initwid) {
    double direction=0, direction_2=0;
    vararr_t arr = pathtolines(bez);
    pathpoint cur_point, last_point, next_point;
    double x=0, y=0, dist;
    double nx, ny, ndir;
    double lx, ly, ldir;
    double lineout=0, linerad=0, linelen=0;
    double theta, phi;

    size_t pathcount;
    pathpoint *pathpoints;
    LIST_DETACH(&arr, &pathpoints, &pathcount);
    linelen = pathpoints[pathcount-1].lengthsofar;

    /* determine miter and bevel points and directions */
    for (size_t i = 0; i < pathcount; i++) {
	const size_t l = i == 0 ? pathcount - 1 : i - 1;
	const size_t n = (i + 1) % pathcount;

	cur_point = pathpoints[i];
	x = cur_point.x;
	y = cur_point.y;
	dist = cur_point.lengthsofar;

	next_point = pathpoints[n];
	nx = next_point.x;
	ny = next_point.y;
	ndir = myatan (ny-y, nx-x);

	last_point = pathpoints[l];
	lx = last_point.x;
	ly = last_point.y;
	ldir = myatan (ly-y, lx-x);

	bool bevel = false;
	direction_2 = 0;

	    /* effective line radius at this point */
	linerad = radfunc(dist, linelen, initwid);

 	if (i == 0 || i == pathcount-1) {
	    lineout = linerad;
	    if (i == 0) {
		direction = ndir + D2R(90);
	    } else {
		direction = ldir - D2R(90);
	    }
	    direction_2 = direction;
	} else {
	    theta = ndir-ldir;
	    if (theta < 0) {
		theta += D2R(360);
	    }
	    phi = D2R(90) - theta / 2;
		 /* actual distance to junction point */
	    if (cos(phi) == 0) {
		lineout = 0;
	    } else {
		lineout = linerad / cos(phi);
	    }
		 /* direction to junction point */
	    direction = ndir+D2R(90)+phi;
	    if (lineout > currentmiterlimit * linerad) {
		bevel = true;
		lineout = linerad;
		direction = mymod(ldir-D2R(90),D2R(360));
		direction_2 = mymod(ndir+D2R(90),D2R(360));
		if (i == pathcount-1) {
		    bevel = false;
		}
	    } else {
		direction_2 = direction;
	    }
	}
	pathpoints[i].x = x;
	pathpoints[i].y = y;
	pathpoints[i].lengthsofar = dist;
	pathpoints[i].type = 'l';
	pathpoints[i].dir = direction;
	pathpoints[i].lout = lineout;
	pathpoints[i].bevel = bevel;
	pathpoints[i].dir2 = direction_2;
    }

	 /* draw line */
    stroke_t p = {0};
	 /* side 1 */
    for (size_t i = 0; i < pathcount; i++) {
	cur_point = pathpoints[i];
	x = cur_point.x;
	y = cur_point.y;
	direction = cur_point.dir;
	lineout = cur_point.lout;
	bool bevel = cur_point.bevel;
	direction_2 = cur_point.dir2;
	if (i == 0) {
	    moveto(&p, x+cos(direction)*lineout, y+sin(direction)*lineout);
	} else {
	    lineto(&p, x+cos(direction)*lineout, y+sin(direction)*lineout);
	}
	if (bevel) {
	    drawbevel(x, lineout, true, direction, direction_2, &p);
	}
    }
	 /* end circle as needed */
    direction += D2R(180);
    lineto(&p, x+cos(direction)*lineout, y+sin(direction)*lineout);
	 /* side 2 */
    assert(pathcount > 0);
    for (size_t i = pathcount - 2; i != SIZE_MAX; i--) {
	cur_point = pathpoints[i];
	x = cur_point.x;
	y = cur_point.y;
	direction = cur_point.dir + D2R(180);
	lineout = cur_point.lout;
	bool bevel = cur_point.bevel;
	direction_2 = cur_point.dir2 + D2R(180);
	lineto(&p, x+cos(direction_2)*lineout, y+sin(direction_2)*lineout);
	if (bevel) { 
	    drawbevel(x, lineout, false, direction, direction_2, &p);
	}
    }
    free(pathpoints);
    return p;
}

#ifdef TEST
static double halffunc (double curlen, double totallen, double initwid)
{
  return (1 - curlen / totallen) * initwid / 2.0;
}

static pointf pts[] = {
  {100,100},
  {150,150},
  {200,100},
  {250,200},
};

main ()
{
    stroke_t* sp;
    bezier bez;

    bez.size = sizeof(pts)/sizeof(pointf);
    bez.list = pts;
    sp = taper(&bez, halffunc, 20.0);
    printf ("newpath\n");
    printf ("%.02f %.02f moveto\n", sp->vertices[0].x, sp->vertices[0].y);
    for (size_t i = 1; i < sp->nvertices; i++)
        printf ("%.02f %.02f lineto\n", sp->vertices[i].x, sp->vertices[i].y);
    printf ("fill showpage\n");
}
#endif
