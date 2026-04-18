/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include <neatogen/neato.h>
#include <pathplan/pathutil.h>
#include <stddef.h>
#include <util/alloc.h>
#include <util/exit.h>

#define SLOPE(p,q) ( ( ( p.y ) - ( q.y ) ) / ( ( p.x ) - ( q.x ) ) )

#define EQ_PT(v,w) (((v).x == (w).x) && ((v).y == (w).y))

#define after(v) (((v)==((v)->poly->finish))?((v)->poly->start):((v)+1))
#define prior(v) (((v)==((v)->poly->start))?((v)->poly->finish):((v)-1))

typedef struct active_edge active_edge;
typedef struct polygon polygon;

    typedef struct {
	pointf pos;
	polygon *poly;
	active_edge *active;
    } vertex ;

    struct polygon {
	vertex *start, *finish;
	boxf bb;
    };

    struct active_edge {
	vertex *name;
	struct active_edge *next, *last;
    };
    typedef struct active_edge_list {
	active_edge *first, *final;
	int number;
    } active_edge_list ;
    typedef struct {
	int nvertices, ninters;
    } data ;

static int sign(double v) {
  if (v < 0)
    return -1;
  if (v > 0)
    return 1;
  return 0;
}

/* find the sign of the area of each of the triangles
    formed by adding a vertex of m to l  
    also find the sign of their product  */
static void sgnarea(vertex *l, vertex *m, int i[])
{
    double a, b, c, d, e, f, g, h, t;
    a = l->pos.x;
    b = l->pos.y;
    c = after(l)->pos.x - a;
    d = after(l)->pos.y - b;
    e = m->pos.x - a;
    f = m->pos.y - b;
    g = after(m)->pos.x - a;
    h = after(m)->pos.y - b;
    t = c * f - d * e;
    i[0] = sign(t);
    t = c * h - d * g;
    i[1] = sign(t);
    i[2] = i[0] * i[1];
}

/** where is `g` relative to the interval delimited by `f` and `h`?
 *
 * The order of `f` and `h` is not assumed. That is, the interval defined may be
 * `(f, h)` or `(h, f)` depending on whether `f` is less than or greater than
 * `h`.
 *
 * \param f First boundary of the interval
 * \param g Value to test
 * \param h Second boundary of the interval
 * \return -1 if g is not in the interval, 1 if g is in the interval, 0 if g is
 *   on the boundary (that is, equal to f or equal to h)
 */
static int between(double f, double g, double h) {
  if (f < g) {
    if (g < h) {
      return 1;
    }
    if (g > h) {
      return -1;
    }
    return 0;
  }
  if (f > g) {
    if (g > h) {
      return 1;
    }
    if (g < h) {
      return -1;
    }
    return 0;
  }
  return 0;
}

/* determine if vertex i of line m is on line l     */
static int online(vertex *l, vertex *m, int i)
{
    pointf a, b, c;
    a = l->pos;
    b = after(l)->pos;
    c = i == 0 ? m->pos : after(m)->pos;
    return a.x == b.x
      ? (a.x == c.x && -1 != between(a.y, c.y, b.y))
      : between(a.x, c.x, b.x);
}

/* determine point of detected intersections  */
static int intpoint(vertex *l, vertex *m, double *x, double *y, int cond)
{
    pointf ls, le, ms, me, pt1, pt2;
    double m1, m2, c1, c2;

    if (cond <= 0)
	return 0;
    ls = l->pos;
    le = after(l)->pos;
    ms = m->pos;
    me = after(m)->pos;

    switch (cond) {

    case 3:			/* a simple intersection        */
	if (ls.x == le.x) {
	    *x = ls.x;
	    *y = me.y + SLOPE(ms, me) * (*x - me.x);
	} else if (ms.x == me.x) {
	    *x = ms.x;
	    *y = le.y + SLOPE(ls, le) * (*x - le.x);
	} else {
	    m1 = SLOPE(ms, me);
	    m2 = SLOPE(ls, le);
	    c1 = ms.y - m1 * ms.x;
	    c2 = ls.y - m2 * ls.x;
	    *x = (c2 - c1) / (m1 - m2);
	    *y = (m1 * c2 - c1 * m2) / (m1 - m2);
	}
	break;

    case 2:			/*     the two lines  have a common segment  */
	if (online(l, m, 0) == -1) {	/* ms between ls and le */
	    pt1 = ms;
	    pt2 = online(m, l, 1) == -1
	      ? (online(m, l, 0) == -1 ? le : ls) : me;
	} else if (online(l, m, 1) == -1) {	/* me between ls and le */
	    pt1 = me;
	    pt2 = online(l, m, 0) == -1
	      ? (online(m, l, 0) == -1 ? le : ls) : ms;
	} else {
	    /* may be degenerate? */
	    if (online(m, l, 0) != -1)
		return 0;
	    pt1 = ls;
	    pt2 = le;
	}

	*x = (pt1.x + pt2.x) / 2;
	*y = (pt1.y + pt2.y) / 2;
	break;

    case 1:			/* a vertex of line m is on line l */
	if ((ls.x - le.x) * (ms.y - ls.y) == (ls.y - le.y) * (ms.x - ls.x)) {
	    *x = ms.x;
	    *y = ms.y;
	} else {
	    *x = me.x;
	    *y = me.y;
	}
    }				/* end switch  */
    return 1;
}

static void
putSeg (int i, vertex* v)
{
    fprintf(stderr, "seg#%d : (%.3f, %.3f) (%.3f, %.3f)\n",
	i, v->pos.x, v->pos.y, after(v)->pos.x, after(v)->pos.y);
}

/* realIntersect:
 * Return 1 if a real intersection has been found
 */
static int
realIntersect (vertex *firstv, vertex *secondv, pointf p)
{
    pointf vft, vsd, avft, avsd;

    vft = firstv->pos;
    avft = after(firstv)->pos;
    vsd = secondv->pos;
    avsd = after(secondv)->pos;

    if ((vft.x != avft.x && vsd.x != avsd.x) ||
	(vft.x == avft.x && !EQ_PT(vft, p) && !EQ_PT(avft, p)) ||
	(vsd.x == avsd.x && !EQ_PT(vsd, p) && !EQ_PT(avsd, p))) 
    {
	if (Verbose > 1) {
		fprintf(stderr, "\nintersection at %.3f %.3f\n",
			p.x, p.y);
		putSeg (1, firstv);
		putSeg (2, secondv);
	}
	return 1;
    }
    else return 0;
}

/* find_intersection:
 * detect whether segments l and m intersect      
 * Return 1 if found; 0 otherwise;
 */
static int find_intersection(vertex *l, vertex *m) {
    double x, y;
    pointf p;
	int i[3];
    sgnarea(l, m, i);

    if (i[2] > 0)
	return 0;

    if (i[2] < 0) {
	sgnarea(m, l, i);
	if (i[2] > 0)
	    return 0;
	if (!intpoint(l, m, &x, &y, i[2] < 0 ? 3 : online(m, l, abs(i[0]))))
	    return 0;
    }

    else if (!intpoint(l, m, &x, &y, i[0] == i[1] ?
		       2 * MAX(online(l, m, 0),
			       online(l, m, 1)) : online(l, m, abs(i[0]))))
	return 0;

    p.x = x;
    p.y = y;
    return realIntersect(l, m, p);
}

static int gt(const void *a, const void *b) {
    const vertex *const *i = a;
    const vertex *const *j = b;
    if ((*i)->pos.x > (*j)->pos.x) {
      return 1;
    }
    if ((*i)->pos.x < (*j)->pos.x) {
      return -1;
    }
    if ((*i)->pos.y > (*j)->pos.y) {
      return 1;
    }
    if ((*i)->pos.y < (*j)->pos.y) {
      return -1;
    }
    return 0;
}

/* find_ints:
 * Check for pairwise intersection of polygon sides
 * Return 1 if intersection found, 0 for not found, -1 for error.
 */
static int find_ints(vertex vertex_list[], size_t nvertices) {
    int j, k, found = 0;
    active_edge_list all;
    active_edge *new, *tempa;
    vertex *pt1, *pt2, *templ;

    all.first = all.final = 0;
    all.number = 0;

    vertex **pvertex = gv_calloc(nvertices, sizeof(vertex*));

    for (size_t i = 0; i < nvertices; i++)
	pvertex[i] = vertex_list + i;

/* sort vertices by x coordinate	*/
    qsort(pvertex, nvertices, sizeof(vertex *), gt);

/* walk through the vertices in order of increasing x coordinate	*/
    for (size_t i = 0; i < nvertices; i++) {
	pt1 = pvertex[i];
	templ = pt2 = prior(pvertex[i]);
	for (k = 0; k < 2; k++) {	/* each vertex has 2 edges */
	    switch (gt(&pt1, &pt2)) {

	    case -1:		/* forward edge, test and insert      */

                 /* test */
		for (tempa = all.first, j = 0; j < all.number;
		     j++, tempa = tempa->next) {
		    found = find_intersection(tempa->name, templ);
		    if (found)
			goto finish;
		}

		new = gv_alloc(sizeof(active_edge));
		if (all.number == 0) {
		    all.first = new;
		    new->last = 0;
		} /* insert */
		else {
		    all.final->next = new;
		    new->last = all.final;
		}

		new->name = templ;
		new->next = 0;
		templ->active = new;
		all.final = new;
		all.number++;

		break;		/* end of case -1       */

	    case 1:		/* backward edge, delete        */

		if ((tempa = templ->active) == 0) {
		    agerrorf("trying to delete a non-line\n");
		    return -1;
		}
		if (all.number == 1)
		    all.final = all.first = 0;	/* delete the line */
		else if (tempa == all.first) {
		    all.first = all.first->next;
		    all.first->last = 0;
		} else if (tempa == all.final) {
		    all.final = all.final->last;
		    all.final->next = 0;
		} else {
		    tempa->last->next = tempa->next;
		    tempa->next->last = tempa->last;
		}
		free(tempa);
		all.number--;
		templ->active = 0;
		break;		/* end of case 1        */

	    default:
		break; // same point; do nothing

	    }			/* end switch   */

	    pt2 = after(pvertex[i]);
	    templ = pvertex[i];	/*second neighbor */
	}			/* end k for loop       */
    }				/* end i for loop       */

finish :
    for (tempa = all.first, j = 0; j < all.number;
		     j++, tempa = new) {
	new = tempa->next;
	free (tempa);
    }
    free (pvertex);
    return found;
}

#define INBOX(p,bb) ((p.x <= bb.UR.x) && (p.x >= bb.LL.x) && (p.y <= bb.UR.y) && (p.y >= bb.LL.y))
#define NESTED(a,b) (INBOX(a.LL,b) && INBOX(a.UR,b))

/* findInside:
 * Check if one polygon is inside another. We know that each
 * pair is either disjoint or one is inside the other.
 * Return 1 if an intersection is found, 0 otherwise.
 */
static int
findInside(Ppoly_t ** polys, int n_polys, polygon* polygon_list)
{
    int i, j;
    pointf pt;
    Ppoly_t* p1;
    Ppoly_t* p2;

    for (i = 0; i < n_polys; i++) {
	p1 = polys[i];
        pt = p1->ps[0];
	for (j = i+1; j < n_polys; j++) {
	    p2 = polys[j];
	    if (NESTED(polygon_list[i].bb,polygon_list[j].bb)) {
		if (in_poly(*p2, pt)) return 1;
	    }
	    else if (NESTED(polygon_list[j].bb,polygon_list[i].bb)) {
		if (in_poly(*p1, p2->ps[0])) return 1;
	    }
	}
    }
    return 0;
}

/* Plegal_arrangement:
 * Check that none of the polygons overlap.
 * Return 1 if okay; 0 otherwise.
 */
int Plegal_arrangement(Ppoly_t ** polys, int n_polys)
{
    int i, vno, found;
    boxf bb;
    double x, y;

    polygon *polygon_list = gv_calloc(n_polys, sizeof(polygon));

    size_t nverts;
    for (nverts = 0, i = 0; i < n_polys; i++) {
	nverts += polys[i]->pn;
    }

    vertex *vertex_list = gv_calloc(nverts, sizeof(vertex));

    for (i = vno = 0; i < n_polys; i++) {
	polygon_list[i].start = &vertex_list[vno];
	bb.LL.x = bb.LL.y = DBL_MAX;
	bb.UR.x = bb.UR.y = -DBL_MAX;
	for (size_t j = 0; j < polys[i]->pn; j++) {
	    x = polys[i]->ps[j].x;
	    y = polys[i]->ps[j].y;
	    bb.LL.x = fmin(bb.LL.x,x);
	    bb.LL.y = fmin(bb.LL.y,y);
	    bb.UR.x = fmax(bb.UR.x,x);
	    bb.UR.y = fmax(bb.UR.y,y);
	    vertex_list[vno].pos.x = x;
	    vertex_list[vno].pos.y = y;
	    vertex_list[vno].poly = &polygon_list[i];
	    vertex_list[vno].active = 0;
	    vno++;
	}
	polygon_list[i].finish = &vertex_list[vno - 1];
	polygon_list[i].bb = bb;
    }

    found = find_ints(vertex_list, nverts);
    if (found < 0) {
	free(polygon_list);
	free(vertex_list);
	return 0;
    }

    if (!found) {
	found = findInside(polys, n_polys, polygon_list);
    }
    free(polygon_list);
    free(vertex_list);

    return !found;
}
