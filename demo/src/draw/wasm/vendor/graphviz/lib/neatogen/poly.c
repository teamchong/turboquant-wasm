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
#include <assert.h>
#include <string.h>
#include <math.h>
#include <neatogen/poly.h>
#include <common/geom.h>
#include <neatogen/mem.h>
#include <stdbool.h>
#include <util/alloc.h>
#include <util/streq.h>

static const int BOX = 1;
static const int CIRCLE = 2;

static bool ISBOX(const Poly *p) { return p->kind & BOX; }
static bool ISCIRCLE(const Poly *p) { return p->kind & CIRCLE; }

static size_t maxcnt = 0;
static Point *tp1 = NULL;
static Point *tp2 = NULL;
static Point *tp3 = NULL;

void polyFree(void)
{
    maxcnt = 0;
    free(tp1);
    free(tp2);
    free(tp3);
    tp1 = NULL;
    tp2 = NULL;
    tp3 = NULL;
}

void breakPoly(Poly * pp)
{
    free(pp->verts);
}

static void bbox(Point *verts, size_t cnt, Point *o, Point *c) {
    double x_min, y_min, x_max, y_max;

    x_min = x_max = verts->x;
    y_min = y_max = verts->y;
    for (size_t i = 1; i < cnt; i++) {
	verts++;
	x_min = fmin(x_min, verts->x);
	y_min = fmin(y_min, verts->y);
	x_max = fmax(x_max, verts->x);
	y_max = fmax(y_max, verts->y);
    }
    o->x = x_min;
    o->y = y_min;
    c->x = x_max;
    c->y = y_max;
}

static void inflatePts(Point *verts, size_t cnt, double xmargin,
                       double ymargin) {
    Point *cur;

    cur = &verts[0];
    for (size_t i = 0; i < cnt; i++) {
	cur->x *= xmargin;
	cur->y *= ymargin;
	cur++;
    }
}

static int isBox(Point *verts, size_t cnt) {
    if (cnt != 4)
	return 0;

    if (verts[0].y == verts[1].y)
	return ((verts[2].y == verts[3].y) &&
		(verts[0].x == verts[3].x) && (verts[1].x == verts[2].x));
    else
	return ((verts[0].x == verts[1].x) &&
		(verts[2].x == verts[3].x) &&
		(verts[0].y == verts[3].y) && (verts[1].y == verts[2].y));
}

static Point makeScaledTransPoint(double x, double y, double dx, double dy) {
    Point rv;
    rv.x = PS2INCH(x) + dx;
    rv.y = PS2INCH(y) + dy;
    return rv;
}

static Point makeScaledPoint(double x, double y)
{
    Point rv;
    rv.x = PS2INCH(x);
    rv.y = PS2INCH(y);
    return rv;
}

static Point *genRound(Agnode_t *n, size_t *sidep, double xm, double ym) {
    size_t sides = 0;
    char *p = agget(n, "samplepoints");

    int isides = 0;
    if (p)
	isides = atoi(p);
    sides = isides < 3 ? DFLT_SAMPLE : (size_t)isides;
    Point *verts = gv_calloc(sides, sizeof(Point));
    for (size_t i = 0; i < sides; i++) {
	verts[i].x =
	    (ND_width(n) / 2.0 + xm) * cos((double)i / (double)sides * M_PI * 2.0);
	verts[i].y =
	    (ND_height(n) / 2.0 + ym) * sin((double)i / (double)sides * M_PI * 2.0);
    }
    *sidep = sides;
    return verts;
}

#define PUTPT(P,X,Y) ((P).x=(X),(P).y=(Y))

int makeAddPoly(Poly *pp, Agnode_t *n, double xmargin, double ymargin) {
    size_t sides;
    Point *verts;
    polygon_t *poly;

    if (ND_clust(n)) {
	Point b;
	sides = 4;
	b.x = ND_width(n) / 2.0 + xmargin;
	b.y = ND_height(n) / 2.0 + ymargin;
	pp->kind = BOX;
	verts = gv_calloc(sides, sizeof(Point));
	PUTPT(verts[0], b.x, b.y);
	PUTPT(verts[1], -b.x, b.y);
	PUTPT(verts[2], -b.x, -b.y);
	PUTPT(verts[3], b.x, -b.y);
    } else
	switch (shapeOf(n)) {
	case SH_POLY:
	    poly = ND_shape_info(n);
	    sides = poly->sides;

	    if (streq(ND_shape(n)->name, "box"))
		pp->kind = BOX;
	    else if (streq(ND_shape(n)->name, "polygon")
		     && isBox(poly->vertices, sides))
		pp->kind = BOX;
	    else if ((poly->sides < 3) && poly->regular)
		pp->kind = CIRCLE;
	    else
		pp->kind = 0;

	    if (sides >= 3) {	/* real polygon */
		verts = gv_calloc(sides, sizeof(Point));
		if (pp->kind == BOX) {
			/* To do an additive margin, we rely on knowing that
			 * the vertices are CCW starting from the UR
			 */
		    verts[0].x = PS2INCH(poly->vertices[0].x) + xmargin;
		    verts[0].y = PS2INCH(poly->vertices[0].y) + ymargin;
		    verts[1].x = PS2INCH(poly->vertices[1].x) - xmargin;
		    verts[1].y = PS2INCH(poly->vertices[1].y) + ymargin;
		    verts[2].x = PS2INCH(poly->vertices[2].x) - xmargin;
		    verts[2].y = PS2INCH(poly->vertices[2].y) - ymargin;
		    verts[3].x = PS2INCH(poly->vertices[3].x) + xmargin;
		    verts[3].y = PS2INCH(poly->vertices[3].y) - ymargin;
 
		}
		else {
		    for (size_t i = 0; i < sides; i++) {
		        const double h = hypot(poly->vertices[i].x, poly->vertices[i].y);
		        verts[i].x = poly->vertices[i].x * (1.0 + xmargin/h);
		        verts[i].y = poly->vertices[i].y * (1.0 + ymargin/h);
		        verts[i].x = PS2INCH(verts[i].x);
		        verts[i].y = PS2INCH(verts[i].y);
		    }
		}
	    } else
		verts = genRound(n, &sides, xmargin, ymargin);
	    break;
	case SH_RECORD: {
	    sides = 4;
	    verts = gv_calloc(sides, sizeof(Point));
	    boxf b = ((field_t*)ND_shape_info(n))->b;
	    verts[0] = makeScaledTransPoint(b.LL.x, b.LL.y, -xmargin, -ymargin);
	    verts[1] = makeScaledTransPoint(b.UR.x, b.LL.y, xmargin, -ymargin);
	    verts[2] = makeScaledTransPoint(b.UR.x, b.UR.y, xmargin, ymargin);
	    verts[3] = makeScaledTransPoint(b.LL.x, b.UR.y, -xmargin, ymargin);
	    pp->kind = BOX;
	    break;
	}
	case SH_POINT:
	    pp->kind = CIRCLE;
	    verts = genRound(n, &sides, xmargin, ymargin);
	    break;
	default:
	    agerrorf("makeAddPoly: unknown shape type %s\n",
		  ND_shape(n)->name);
	    return 1;
	}

    pp->verts = verts;
    pp->nverts = (int)sides;
    bbox(verts, sides, &pp->origin, &pp->corner);

    if (sides > maxcnt)
	maxcnt = sides;
    return 0;
}

int makePoly(Poly *pp, Agnode_t *n, double xmargin, double ymargin) {
    size_t sides;
    Point *verts;
    polygon_t *poly;

    if (ND_clust(n)) {
	Point b;
	sides = 4;
	b.x = ND_width(n) / 2.0;
	b.y = ND_height(n) / 2.0;
	pp->kind = BOX;
	verts = gv_calloc(sides, sizeof(Point));
	PUTPT(verts[0], b.x, b.y);
	PUTPT(verts[1], -b.x, b.y);
	PUTPT(verts[2], -b.x, -b.y);
	PUTPT(verts[3], b.x, -b.y);
    } else
	switch (shapeOf(n)) {
	case SH_POLY:
	    poly = ND_shape_info(n);
	    sides = poly->sides;
	    if (sides >= 3) {	/* real polygon */
		verts = gv_calloc(sides, sizeof(Point));
		for (size_t i = 0; i < sides; i++) {
		    verts[i].x = PS2INCH(poly->vertices[i].x);
		    verts[i].y = PS2INCH(poly->vertices[i].y);
		}
	    } else
		verts = genRound(n, &sides, 0, 0);

	    if (streq(ND_shape(n)->name, "box"))
		pp->kind = BOX;
	    else if (streq(ND_shape(n)->name, "polygon")
		     && isBox(verts, sides))
		pp->kind = BOX;
	    else if ((poly->sides < 3) && poly->regular)
		pp->kind = CIRCLE;
	    else
		pp->kind = 0;

	    break;
	case SH_RECORD: {
	    sides = 4;
	    verts = gv_calloc(sides, sizeof(Point));
	    boxf b = ((field_t *) ND_shape_info(n))->b;
	    verts[0] = makeScaledPoint(b.LL.x, b.LL.y);
	    verts[1] = makeScaledPoint(b.UR.x, b.LL.y);
	    verts[2] = makeScaledPoint(b.UR.x, b.UR.y);
	    verts[3] = makeScaledPoint(b.LL.x, b.UR.y);
	    pp->kind = BOX;
	    break;
	}
	case SH_POINT:
	    pp->kind = CIRCLE;
	    verts = genRound(n, &sides, 0, 0);
	    break;
	default:
	    agerrorf("makePoly: unknown shape type %s\n",
		  ND_shape(n)->name);
	    return 1;
	}

    if ((xmargin != 1.0) || (ymargin != 1.0))
	inflatePts(verts, sides, xmargin, ymargin);

    pp->verts = verts;
    pp->nverts = (int)sides;
    bbox(verts, sides, &pp->origin, &pp->corner);

    if (sides > maxcnt)
	maxcnt = sides;
    return 0;
}

static int
pintersect(Point originp, Point cornerp, Point originq, Point cornerq)
{
    return ((originp.x <= cornerq.x) && (originq.x <= cornerp.x) &&
	    (originp.y <= cornerq.y) && (originq.y <= cornerp.y));
}

#define Pin 1
#define Qin 2
#define Unknown 0

#define advance(A,B,N) (B++, A = (A+1)%N)

static int edgesIntersect(Point * P, Point * Q, int n, int m)
{
    int a = 0;
    int b = 0;
    int aa = 0;
    int ba = 0;
    int a1, b1;
    Point A, B;
    double cross;
    int bHA, aHB;
    Point p;
    int inflag = Unknown;
    /* int      i = 0; */
    /* int      Reset = 0; */

    do {
	a1 = (a + n - 1) % n;
	b1 = (b + m - 1) % m;

	subpt(&A, P[a], P[a1]);
	subpt(&B, Q[b], Q[b1]);

	cross = area_2((Point){0}, A, B);
	bHA = leftOf(P[a1], P[a], Q[b]);
	aHB = leftOf(Q[b1], Q[b], P[a]);

	/* If A & B intersect, update inflag. */
	if (intersection(P[a1], P[a], Q[b1], Q[b], &p))
	    return 1;

	/* Advance rules. */
	if ((cross == 0) && !bHA && !aHB) {
	    if (inflag == Pin)
		advance(b, ba, m);
	    else
		advance(a, aa, n);
	} else if (cross >= 0)
	    if (bHA)
		advance(a, aa, n);
	    else {
		advance(b, ba, m);
	} else {		/* if ( cross < 0 ) */

	    if (aHB)
		advance(b, ba, m);
	    else
		advance(a, aa, n);
	}

    } while (((aa < n) || (ba < m)) && (aa < 2 * n) && (ba < 2 * m));

    return 0;

}

/* inPoly:
 * Return 1 if q is inside polygon vertex[]
 * Assume points are in CCW order
 */
static int inPoly(Point vertex[], int n, Point q)
{
    int i, i1;			/* point index; i1 = i-1 mod n */
    double x;			/* x intersection of e with ray */
    double crossings = 0;	/* number of edge/ray crossings */

    if (tp3 == NULL)
	tp3 = gv_calloc(maxcnt, sizeof(Point));

    /* Shift so that q is the origin. */
    for (i = 0; i < n; i++) {
	tp3[i].x = vertex[i].x - q.x;
	tp3[i].y = vertex[i].y - q.y;
    }

    /* For each edge e=(i-1,i), see if crosses ray. */
    for (i = 0; i < n; i++) {
	i1 = (i + n - 1) % n;

	/* if edge is horizontal, test to see if the point is on it */
	if (tp3[i].y == 0 && tp3[i1].y == 0) {
	    if (tp3[i].x * tp3[i1].x < 0) {
		return 1;
	    } else {
		continue;
	    }
	}

	/* if e straddles the x-axis... */
	if ((tp3[i].y >= 0 && tp3[i1].y <= 0) ||
	    (tp3[i1].y >= 0 && tp3[i].y <= 0)) {
	    /* e straddles ray, so compute intersection with ray. */
	    x = (tp3[i].x * tp3[i1].y - tp3[i1].x * tp3[i].y)
		/ (double) (tp3[i1].y - tp3[i].y);

	    /* if intersect at origin, we've found intersection */
	    if (x == 0)
		return 1;;

	    /* crosses ray if strictly positive intersection. */
	    if (x > 0) {
		if (tp3[i].y == 0 || tp3[i1].y == 0) {
		    crossings += .5;	/* goes through vertex */
		} else {
		    crossings += 1.0;
		}
	    }
	}
    }

    /* q inside if an odd number of crossings. */
    if ((int)crossings % 2 == 1)
	return 1;
    else
	return 0;
}

static bool inBox(Point p, Point origin_point, Point corner) {
    return p.x <= corner.x && p.x >= origin_point.x && p.y <= corner.y &&
           p.y >= origin_point.y;
}

static void transCopy(Point * inp, int cnt, Point off, Point * outp)
{
    int i;

    for (i = 0; i < cnt; i++) {
	outp->x = inp->x + off.x;
	outp->y = inp->y + off.y;
	inp++;
	outp++;
    }
}

int polyOverlap(Point p, Poly * pp, Point q, Poly * qp)
{
    Point op, cp;
    Point oq, cq;

    /* translate bounding boxes */
    addpt(&op, p, pp->origin);
    addpt(&cp, p, pp->corner);
    addpt(&oq, q, qp->origin);
    addpt(&cq, q, qp->corner);

    /* If bounding boxes don't overlap, done */
    if (!pintersect(op, cp, oq, cq))
	return 0;

    if (ISBOX(pp) && ISBOX(qp))
	return 1;
    if (ISCIRCLE(pp) && ISCIRCLE(qp)) {
	double d =
	    (pp->corner.x - pp->origin.x + qp->corner.x - qp->origin.x);
	double dx = p.x - q.x;
	double dy = p.y - q.y;
	if ((dx * dx + dy * dy) > (d * d) / 4.0)
	    return 0;
	else
	    return 1;
    }

    if (tp1 == NULL) {
	tp1 = gv_calloc(maxcnt, sizeof(Point));
	tp2 = gv_calloc(maxcnt, sizeof(Point));
    }

    transCopy(pp->verts, pp->nverts, p, tp1);
    transCopy(qp->verts, qp->nverts, q, tp2);
    return (edgesIntersect(tp1, tp2, pp->nverts, qp->nverts) ||
	    (inBox(*tp1, oq, cq) && inPoly(tp2, qp->nverts, *tp1)) ||
	    (inBox(*tp2, op, cp) && inPoly(tp1, pp->nverts, *tp2)));
}
