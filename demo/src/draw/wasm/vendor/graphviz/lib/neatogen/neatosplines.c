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
#include "config.h"
#include <limits.h>
#include <math.h>
#include <neatogen/neato.h>
#include <neatogen/adjust.h>
#include <pathplan/pathplan.h>
#include <pathplan/vispath.h>
#include <neatogen/multispline.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/alloc.h>
#include <util/unreachable.h>

#ifdef ORTHO
#include <ortho/ortho.h>
#endif


static bool spline_merge(node_t * n)
{
    (void)n;
    return false;
}

static bool swap_ends_p(edge_t * e)
{
    (void)e;
    return false;
}

static splineInfo sinfo = {.swapEnds = swap_ends_p,
                           .splineMerge = spline_merge};

static void make_barriers(Ppoly_t **poly, int npoly, int pp, int qp,
                          Pedge_t **barriers, size_t *n_barriers) {
    int i, j, k;

    size_t n = 0;
    for (i = 0; i < npoly; i++) {
	if (i == pp)
	    continue;
	if (i == qp)
	    continue;
	n += poly[i]->pn;
    }
    Pedge_t *bar = gv_calloc(n, sizeof(Pedge_t));
    size_t b = 0;
    for (i = 0; i < npoly; i++) {
	if (i == pp)
	    continue;
	if (i == qp)
	    continue;
	for (j = 0; j < (int)poly[i]->pn; j++) {
	    k = j + 1;
	    if (k >= (int)poly[i]->pn)
		k = 0;
	    bar[b].a = poly[i]->ps[j];
	    bar[b].b = poly[i]->ps[k];
	    b++;
	}
    }
    assert(b == n);
    *barriers = bar;
    *n_barriers = n;
}

static Ppoint_t genPt(double x, double y, pointf c)
{
    Ppoint_t p;

    p.x = x + c.x;
    p.y = y + c.y;
    return p;
}

static Ppoint_t recPt(double x, double y, pointf c, expand_t* m)
{
    Ppoint_t p;

    p.x = x * m->x + c.x;
    p.y = y * m->y + c.y;
    return p;
}

typedef struct {
    node_t *n1;
    pointf p1;
    node_t *n2;
    pointf p2;
} edgeinfo;
typedef struct {
    Dtlink_t link;
    edgeinfo id;
    edge_t *e;
} edgeitem;

static void *newitem(void *p, Dtdisc_t *disc) {
    edgeitem *obj = p;
    edgeitem *newp;

    (void)disc;
    newp = gv_alloc(sizeof(edgeitem));
    newp->id = obj->id;
    newp->e = obj->e;
    ED_count(newp->e) = 1;

    return newp;
}

static int cmpitems(void *k1, void *k2) {
    const edgeinfo *key1 = k1;
    const edgeinfo *key2 = k2;
    if (key1->n1 > key2->n1)
	return 1;
    if (key1->n1 < key2->n1)
	return -1;
    if (key1->n2 > key2->n2)
	return 1;
    if (key1->n2 < key2->n2)
	return -1;

    if (key1->p1.x > key2->p1.x)
	return 1;
    if (key1->p1.x < key2->p1.x)
	return -1;
    if (key1->p1.y > key2->p1.y)
	return 1;
    if (key1->p1.y < key2->p1.y)
	return -1;
    if (key1->p2.x > key2->p2.x)
	return 1;
    if (key1->p2.x < key2->p2.x)
	return -1;
    if (key1->p2.y > key2->p2.y)
	return 1;
    if (key1->p2.y < key2->p2.y)
	return -1;
    return 0;
}

Dtdisc_t edgeItemDisc = {
    offsetof(edgeitem, id),
    sizeof(edgeinfo),
    offsetof(edgeitem, link),
    newitem,
    free,
    cmpitems,
};

/* equivEdge:
 * See if we have already encountered an edge between the same
 * node:port pairs. If so, return the earlier edge. If not, 
 * this edge is added to map and returned.
 * We first have to canonicalize the key fields using a lexicographic
 * ordering.
 */
static edge_t *equivEdge(Dt_t * map, edge_t * e)
{
    edgeinfo test;
    edgeitem dummy;
    edgeitem *ip;

    if (agtail(e) < aghead(e)) {
	test.n1 = agtail(e);
	test.p1 = ED_tail_port(e).p;
	test.n2 = aghead(e);
	test.p2 = ED_head_port(e).p;
    } else if (agtail(e) > aghead(e)) {
	test.n2 = agtail(e);
	test.p2 = ED_tail_port(e).p;
	test.n1 = aghead(e);
	test.p1 = ED_head_port(e).p;
    } else {
	pointf hp = ED_head_port(e).p;
	pointf tp = ED_tail_port(e).p;
	if (tp.x < hp.x) {
	    test.p1 = tp;
	    test.p2 = hp;
	} else if (tp.x > hp.x) {
	    test.p1 = hp;
	    test.p2 = tp;
	} else if (tp.y < hp.y) {
	    test.p1 = tp;
	    test.p2 = hp;
	} else if (tp.y > hp.y) {
	    test.p1 = hp;
	    test.p2 = tp;
	} else {
	    test.p1 = test.p2 = tp;
	}
	test.n2 = test.n1 = agtail(e);
    }
    dummy.id = test;
    dummy.e = e;
    ip = dtinsert(map, &dummy);
    return ip->e;
}


/* makeSelfArcs:
 * Generate loops. We use the library routine makeSelfEdge
 * which also places the labels.
 * We have to handle port labels here.
 * as well as update the bbox from edge labels.
 */
void makeSelfArcs(edge_t * e, int stepx)
{
    assert(ED_count(e) >= 0);
    const size_t cnt = (size_t)ED_count(e);

    if (cnt == 1 || Concentrate) {
	edge_t *edges1[1];
	edges1[0] = e;
	makeSelfEdge(edges1, 1, stepx, stepx, &sinfo);
	if (ED_label(e))
	    updateBB(agraphof(agtail(e)), ED_label(e));
	makePortLabels(e);
    } else if (cnt > 1) {
	edge_t **edges = gv_calloc(cnt, sizeof(edge_t*));
	for (size_t i = 0; i < cnt; i++) {
	    edges[i] = e;
	    e = ED_to_virt(e);
	}
	makeSelfEdge(edges, cnt, stepx, stepx, &sinfo);
	for (size_t i = 0; i < cnt; i++) {
	    e = edges[i];
	    if (ED_label(e))
		updateBB(agraphof(agtail(e)), ED_label(e));
	    makePortLabels(e);
	}
	free(edges);
    }
}

/** calculate the slope of the tangent of an ellipse
 *
 * The equation for the slope `m` of the tangent of an ellipse as a function of
 * `x` * is given by:
 *
 *           bx
 * m = ± ――――――――――
 *          _______
 *       a √ a²- x²
 *
 * or
 *
 * m = ± (b * x) / (a * sqrt(a * a - x * x))
 *
 * We know that the slope is negative in the first and third quadrant, i.e.,
 * when the signs of x and y are the same, so we use that to select the correct
 * slope.
 *
 * @param a Half the width of the ellipse, i.e., the maximum x value
 * @param b Half the height of the ellipse, i.e., the maximum y value
 * @param p A point on the ellipse periphery in which to calculate the slope of
 * the tangent
 * @return The slope of the tangent in point `p`
 */
static double ellipse_tangent_slope(double a, double b, pointf p) {
  assert(p.x != a &&
         "cannot handle ellipse tangent slope in horizontal extreme point");
  const double sign_y = p.y >= 0 ? 1 : -1;
  const double m = -sign_y * (b * p.x) / (a * sqrt(a * a - p.x * p.x));
  assert(isfinite(m) && "ellipse tangent slope is infinite");
  return m;
}

/** calculate the intersection of two lines
 *
 * @param l0 First line
 * @param l1 Second line
 * @return Intersection of the two lines
 */
static pointf line_intersection(linef l0, linef l1) {
  const double x =
      (l0.m * l0.p.x - l0.p.y - l1.m * l1.p.x + l1.p.y) / (l0.m - l1.m);
  const double y = l0.p.y + l0.m * (x - l0.p.x);
  return (pointf){x, y};
}

/** calculate a corner of a polygon circumscribed about an ellipse
 *
 * @param a Half the width of the ellipse, i.e., the maximum x value
 * @param b Half the height of the ellipse, i.e., the maximum y value
 * @param i Index of the polygon corner
 * @param nsides Number of sides of the polygon
 * @return Polygon corner at index `i`
 */
static pointf circumscribed_polygon_corner_about_ellipse(double a, double b,
                                                         size_t i,
                                                         size_t nsides) {
  const double angle0 = 2.0 * M_PI * ((double)i - 0.5) / (double)nsides;
  const double angle1 = 2.0 * M_PI * ((double)i + 0.5) / (double)nsides;
  const pointf p0 = {a * cos(angle0), b * sin(angle0)};
  const pointf p1 = {a * cos(angle1), b * sin(angle1)};
  const double m0 = ellipse_tangent_slope(a, b, p0);
  const double m1 = ellipse_tangent_slope(a, b, p1);
  const linef line0 = {{p0.x, p0.y}, m0};
  const linef line1 = {{p1.x, p1.y}, m1};
  return line_intersection(line0, line1);
}

/* makeObstacle:
 * Given a node, return an obstacle reflecting the
 * node's geometry. pmargin specifies how much space to allow
 * around the polygon. 
 * Returns the constructed polygon on success, NULL on failure.
 * Failure means the node shape is not supported. 
 *
 * If isOrtho is true, we have to use the bounding box of each node.
 *
 * The polygon has its vertices in CW order.
 * 
 */
Ppoly_t *makeObstacle(node_t * n, expand_t* pmargin, bool isOrtho)
{
    Ppoly_t *obs;
    polygon_t *poly;
    size_t sides;
    pointf polyp;
    boxf b;
    pointf pt;
    field_t *fld;
    bool isPoly;
    pointf* verts = NULL;
    pointf vs[4];
    pointf p;
    pointf margin = {0};

    switch (shapeOf(n)) {
    case SH_POLY:
    case SH_POINT:
	obs = gv_alloc(sizeof(Ppoly_t));
	poly = ND_shape_info(n);
	if (isOrtho) {
	    isPoly = true;
	    sides = 4;
	    verts = vs;
		/* For fixedshape, we can't use the width and height, as this includes
		 * the label. We only want to use the actual node shape.
		 */
	    if (poly->option.fixedshape) {
		b = polyBB (poly);
		vs[0] = b.LL;
		vs[1].x = b.UR.x;
		vs[1].y = b.LL.y;
		vs[2] = b.UR;
		vs[3].x = b.LL.x;
		vs[3].y = b.UR.y;
	    } else {
		const double width = ND_lw(n) + ND_rw(n);
		const double outline_width = INCH2PS(ND_outline_width(n));
		// scale lw and rw proportionally to sum to outline width
		const double outline_lw = ND_lw(n) * outline_width / width;
		const double outline_ht = INCH2PS(ND_outline_height(n));
		p.x = -outline_lw;
		p.y = -outline_ht / 2.0;
		vs[0] = p;
		p.x = outline_lw;
		vs[1] = p;
		p.y = outline_ht / 2.0;
		vs[2] = p;
		p.x = -outline_lw;
		vs[3] = p;
	    }
	}
	else if (poly->sides >= 3) {
	    isPoly = true;
	    sides = poly->sides;
            const double penwidth = late_double(n, N_penwidth, 1.0, 0.0);
            // possibly use extra vertices representing the outline, i.e., the
            // outermost periphery with penwidth taken into account
            const size_t extra_peripheries = poly->peripheries >= 1 && penwidth > 0.0 ? 1 : 0;
            const size_t outline_periphery = poly->peripheries + extra_peripheries;
            const size_t vertices_offset = outline_periphery >= 1 ? (outline_periphery - 1) * sides : 0;
            verts = poly->vertices + vertices_offset;
            margin.x = pmargin->x;
            margin.y = pmargin->y;
	} else {		/* ellipse */
	    isPoly = false;
	    sides = 8;
	}
	obs->pn = sides;
	obs->ps = gv_calloc(sides, sizeof(Ppoint_t));
	/* assuming polys are in CCW order, and pathplan needs CW */
	for (size_t j = 0; j < sides; j++) {
	    double xmargin = 0.0, ymargin = 0.0;
	    if (isPoly) {
		if (pmargin->doAdd) {
		    if (sides == 4) {
			switch (j) {
			case 0 :
			    xmargin = margin.x;
			    ymargin = margin.y;
			    break;
			case 1 :
			    xmargin = -margin.x;
			    ymargin = margin.y;
			    break;
			case 2 :
			    xmargin = -margin.x;
			    ymargin = -margin.y;
			    break;
			case 3 :
			    xmargin = margin.x;
			    ymargin = -margin.y;
			    break;
			default:
			    UNREACHABLE();
			}
			polyp.x = verts[j].x + xmargin;
			polyp.y = verts[j].y + ymargin;
		    }
		    else {
			const double h = hypot(verts[j].x, verts[j].y);
			polyp.x = verts[j].x * (1.0 + margin.x/h);
			polyp.y = verts[j].y * (1.0 + margin.y/h);
		    }
		}
		else {
		    polyp.x = verts[j].x * margin.x;
		    polyp.y = verts[j].y * margin.y;
		}
	    } else {
		const double width = INCH2PS(ND_outline_width(n));
		const double height = INCH2PS(ND_outline_height(n));
		margin = pmargin->doAdd ? (pointf) {pmargin->x, pmargin->y} : (pointf) {0.0, 0.0};
		const double ellipse_a = (width + margin.x) / 2.0;
		const double ellipse_b = (height + margin.y) / 2.0;
		polyp = circumscribed_polygon_corner_about_ellipse(ellipse_a, ellipse_b, j, sides);
	    }
	    obs->ps[sides - j - 1].x = polyp.x + ND_coord(n).x;
	    obs->ps[sides - j - 1].y = polyp.y + ND_coord(n).y;
	}
	break;
    case SH_RECORD:
	fld = ND_shape_info(n);
	b = fld->b;
	obs = gv_alloc(sizeof(Ppoly_t));
	obs->pn = 4;
	obs->ps = gv_calloc(4, sizeof(Ppoint_t));
	/* CW order */
	pt = ND_coord(n);
	if (pmargin->doAdd) {
	    obs->ps[0] = genPt(b.LL.x-pmargin->x, b.LL.y-pmargin->y, pt);
	    obs->ps[1] = genPt(b.LL.x-pmargin->x, b.UR.y+pmargin->y, pt);
	    obs->ps[2] = genPt(b.UR.x+pmargin->x, b.UR.y+pmargin->y, pt);
	    obs->ps[3] = genPt(b.UR.x+pmargin->x, b.LL.y-pmargin->y, pt);
	}
	else {
	    obs->ps[0] = recPt(b.LL.x, b.LL.y, pt, pmargin);
	    obs->ps[1] = recPt(b.LL.x, b.UR.y, pt, pmargin);
	    obs->ps[2] = recPt(b.UR.x, b.UR.y, pt, pmargin);
	    obs->ps[3] = recPt(b.UR.x, b.LL.y, pt, pmargin);
	}
	break;
    case SH_EPSF:
	obs = gv_alloc(sizeof(Ppoly_t));
	obs->pn = 4;
	obs->ps = gv_calloc(4, sizeof(Ppoint_t));
	/* CW order */
	pt = ND_coord(n);
	if (pmargin->doAdd) {
	    obs->ps[0] = genPt(-ND_lw(n)-pmargin->x, -ND_ht(n)-pmargin->y,pt);
	    obs->ps[1] = genPt(-ND_lw(n)-pmargin->x, ND_ht(n)+pmargin->y,pt);
	    obs->ps[2] = genPt(ND_rw(n)+pmargin->x, ND_ht(n)+pmargin->y,pt);
	    obs->ps[3] = genPt(ND_rw(n)+pmargin->x, -ND_ht(n)-pmargin->y,pt);
	}
	else {
	    obs->ps[0] = recPt(-ND_lw(n), -ND_ht(n), pt, pmargin);
	    obs->ps[1] = recPt(-ND_lw(n), ND_ht(n), pt, pmargin);
	    obs->ps[2] = recPt(ND_rw(n), ND_ht(n), pt, pmargin);
	    obs->ps[3] = recPt(ND_rw(n), -ND_ht(n), pt, pmargin);
	}
	break;
    default:
	obs = NULL;
	break;
    }
    return obs;
}

/* getPath
 * Construct the shortest path from one endpoint of e to the other.
 * vconfig is a precomputed data structure to help in the computation.
 * If chkPts is true, the function finds the polygons, if any, containing
 * the endpoints and tells the shortest path computation to ignore them. 
 * Assumes this info is set in ND_lim, usually from _spline_edges.
 * Returns the shortest path.
 */
Ppolyline_t getPath(edge_t *e, vconfig_t *vconfig, bool chkPts) {
    Ppolyline_t line;
    int pp, qp;
    Ppoint_t p, q;

    p = add_pointf(ND_coord(agtail(e)), ED_tail_port(e).p);
    q = add_pointf(ND_coord(aghead(e)), ED_head_port(e).p);

    /* determine the polygons (if any) that contain the endpoints */
    pp = qp = POLYID_NONE;
    if (chkPts) {
	pp = ND_lim(agtail(e));
	qp = ND_lim(aghead(e));
    }
    Pobspath(vconfig, p, pp, q, qp, &line);
    return line;
}

static void makePolyline(edge_t * e) {
    Ppolyline_t spl, line = ED_path(e);

    make_polyline (line, &spl);
    if (Verbose > 1)
	fprintf(stderr, "polyline %s %s\n", agnameof(agtail(e)), agnameof(aghead(e)));
    clip_and_install(e, aghead(e), spl.ps, spl.pn, &sinfo);
    addEdgeLabels(e);
}

/* makeSpline:
 * Construct a spline connecting the endpoints of e, avoiding the npoly
 * obstacles obs.
 * The resultant spline is attached to the edge, the positions of any 
 * edge labels are computed, and the graph's bounding box is recomputed.
 * 
 * If chkPts is true, the function checks if one or both of the endpoints 
 * is on or inside one of the obstacles and, if so, tells the shortest path
 * computation to ignore them. 
 */
void makeSpline(edge_t *e, Ppoly_t **obs, int npoly, bool chkPts) {
    Ppolyline_t line, spline;
    Pvector_t slopes[2];
    int i;
    int pp, qp;
    Ppoint_t p, q;
    Pedge_t *barriers;

    line = ED_path(e);
    p = line.ps[0];
    q = line.ps[line.pn - 1];
    /* determine the polygons (if any) that contain the endpoints */
    pp = qp = POLYID_NONE;
    if (chkPts)
	for (i = 0; i < npoly; i++) {
	    if (pp == POLYID_NONE && in_poly(*obs[i], p))
		pp = i;
	    if (qp == POLYID_NONE && in_poly(*obs[i], q))
		qp = i;
	}

    size_t n_barriers;
    make_barriers(obs, npoly, pp, qp, &barriers, &n_barriers);
    slopes[0].x = slopes[0].y = 0.0;
    slopes[1].x = slopes[1].y = 0.0;
    if (Proutespline(barriers, n_barriers, line, slopes, &spline) < 0) {
	agerrorf("makeSpline: failed to make spline edge (%s,%s)\n", agnameof(agtail(e)), agnameof(aghead(e)));
	return;
    }

    /* north why did you ever use int coords */
    if (Verbose > 1)
	fprintf(stderr, "spline %s %s\n", agnameof(agtail(e)), agnameof(aghead(e)));
    clip_and_install(e, aghead(e), spline.ps, spline.pn, &sinfo);
    free(barriers);
    addEdgeLabels(e);
}

  /* True if either head or tail has a port on its boundary */
#define BOUNDARY_PORT(e) ((ED_tail_port(e).side)||(ED_head_port(e).side))

/* Basic default routine for creating edges.
 * If splines are requested, we construct the obstacles.
 * If not, or nodes overlap, the function reverts to line segments.
 * NOTE: intra-cluster edges are not constrained to
 * remain in the cluster's bounding box and, conversely, a cluster's box
 * is not altered to reflect intra-cluster edges.
 * If Nop > 1 and the spline exists, it is just copied.
 * NOTE: if edgetype = EDGETYPE_NONE, we shouldn't be here.
 */
static int spline_edges_(graph_t *g, expand_t *pmargin, int edgetype) {
    node_t *n;
    edge_t *e;
    edge_t *e0;
    Ppoly_t **obs = 0;
    Ppoly_t *obp;
    int cnt, i = 0, npoly;
    vconfig_t *vconfig = 0;
    int useEdges = Nop > 1;
    int legal = 0;

#ifdef HAVE_GTS
    router_t* rtr = 0;
#endif
    
    /* build configuration */
    if (edgetype >= EDGETYPE_PLINE) {
	obs = gv_calloc(agnnodes(g), sizeof(Ppoly_t*));
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    obp = makeObstacle(n, pmargin, edgetype == EDGETYPE_ORTHO);
	    if (obp) {
		ND_lim(n) = i; 
		obs[i++] = obp;
	    }
	    else
		ND_lim(n) = POLYID_NONE; 
	}
    } else {
	obs = 0;
    }
    npoly = i;
    if (obs) {
	if ((legal = Plegal_arrangement(obs, npoly))) {
	    if (edgetype != EDGETYPE_ORTHO) vconfig = Pobsopen(obs, npoly);
	}
	else {
	    if (edgetype == EDGETYPE_ORTHO)
		agwarningf("the bounding boxes of some nodes touch - falling back to straight line edges\n");
	    else 
		agwarningf("some nodes with margin (%.02f,%.02f) touch - falling back to straight line edges\n", pmargin->x, pmargin->y);
	}
    }

    /* route edges  */
    if (Verbose)
	fprintf(stderr, "Creating edges using %s\n",
	    (legal && edgetype == EDGETYPE_ORTHO) ? "orthogonal lines" :
	    (vconfig ? (edgetype == EDGETYPE_SPLINE ? "splines" : "polylines") :
		"line segments"));
    if (vconfig) {
	/* path-finding pass */
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
		ED_path(e) = getPath(e, vconfig, true);
	    }
	}
    }
#ifdef ORTHO
    else if (legal && edgetype == EDGETYPE_ORTHO) {
	orthoEdges(g, false);
	useEdges = 1;
    }
#endif

    /* spline-drawing pass */
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
/* fprintf (stderr, "%s -- %s %d\n", agnameof(agtail(e)), agnameof(aghead(e)), ED_count(e)); */
	    node_t *head = aghead(e);
	    if (useEdges && ED_spl(e)) {
		(void)add_pointf(ND_coord(n), ED_tail_port(e).p);
		(void)add_pointf(ND_coord(head), ED_head_port(e).p);
		addEdgeLabels(e);
	    } 
	    else if (ED_count(e) == 0) continue;  /* only do representative */
	    else if (n == head) {    /* self arc */
		makeSelfArcs(e, GD_nodesep(g->root));
	    } else if (vconfig) { /* EDGETYPE_SPLINE or EDGETYPE_PLINE */
#ifdef HAVE_GTS
		if (ED_count(e) > 1 || BOUNDARY_PORT(e)) {
		    int fail = 0;
		    if (ED_path(e).pn == 2 && !BOUNDARY_PORT(e))
			     /* if a straight line can connect the ends */
			makeStraightEdge(g, e, edgetype, &sinfo);
		    else { 
			if (!rtr) rtr = mkRouter (obs, npoly);
			fail = makeMultiSpline(e, rtr, edgetype == EDGETYPE_PLINE);
		    } 
		    if (!fail) continue;
		}
		/* We can probably remove this branch and just use
		 * makeMultiSpline. It can also catch the makeStraightEdge
		 * case. We could then eliminate all of the vconfig stuff.
		 */
#endif
		cnt = ED_count(e);
		if (Concentrate) cnt = 1; /* only do representative */
		e0 = e;
		for (i = 0; i < cnt; i++) {
		    if (edgetype == EDGETYPE_SPLINE)
			makeSpline(e0, obs, npoly, true);
		    else
			makePolyline(e0);
		    e0 = ED_to_virt(e0);
		}
	    } else {
		makeStraightEdge(g, e, edgetype, &sinfo);
	    }
	}
    }

#ifdef HAVE_GTS
    if (rtr)
	freeRouter (rtr);
#endif

    if (vconfig)
	Pobsclose (vconfig);
    if (obs) {
	for (i=0; i < npoly; i++) {
	    free (obs[i]->ps);
	    free (obs[i]);
	}
	free (obs);
    }
    return 0;
}

/* splineEdges:
 * Main wrapper code for generating edges.
 * Sets desired separation. 
 * Coalesces equivalent edges (edges * with the same endpoints). 
 * It then calls the edge generating function, and marks the
 * spline phase complete.
 * Returns 0 on success.
 *
 * The edge function is given the graph, the separation to be added
 * around obstacles, and the type of edge. It must guarantee 
 * that all bounding boxes are current; in particular, the bounding box of 
 * g must reflect the addition of the edges.
 */
int
splineEdges(graph_t * g, int (*edgefn) (graph_t *, expand_t*, int),
	    int edgetype)
{
    node_t *n;
    edge_t *e;
    expand_t margin;
    Dt_t *map;

    margin = esepFactor (g);

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    resolvePorts (e);
	}
    }

    /* find equivalent edges */
    map = dtopen(&edgeItemDisc, Dtoset);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    if (Nop > 1 && ED_spl(e)) {
		/* If Nop > 1 (use given edges) and e has a spline, it
 		 * should have its own equivalence class.
		 */
		    ED_count(e)++;
	    } else {
		edge_t *leader = equivEdge(map, e);
		if (leader != e) {
		    ED_count(leader)++;
		    ED_to_virt(e) = ED_to_virt(leader);
		    ED_to_virt(leader) = e;
		}
	    }
	}
    }
    dtclose(map);

    if (edgefn(g, &margin, edgetype))
	return 1;

    State = GVSPLINES;
    return 0;
}

/* spline_edges1:
 * Construct edges using default algorithm and given splines value.
 * Return 0 on success.
 */
int spline_edges1(graph_t * g, int edgetype)
{
  return splineEdges(g, spline_edges_, edgetype);
}

/* spline_edges0:
 * Sets the graph's aspect ratio.
 * Check splines attribute and construct edges using default algorithm.
 * If the splines attribute is defined but equal to "", skip edge routing.
 * 
 * Assumes u.bb for has been computed for g and all clusters
 * (not just top-level clusters), and  that GD_bb(g).LL is at the origin.
 *
 * This last criterion is, I believe, mainly to simplify the code
 * in neato_set_aspect. It would be good to remove this constraint,
 * as this would allow nodes pinned on input to have the same coordinates
 * when output in dot or plain format.
 *
 */
void spline_edges0(graph_t *g, bool set_aspect) {
    int et = EDGE_TYPE (g);
    if (set_aspect) neato_set_aspect(g);
    if (et == EDGETYPE_NONE) return;
#ifndef ORTHO
    if (et == EDGETYPE_ORTHO) {
	agwarningf("Orthogonal edges not yet supported\n");
	et = EDGETYPE_PLINE;
	GD_flags(g->root) &= ~EDGETYPE_ORTHO;
	GD_flags(g->root) |= EDGETYPE_PLINE;
    }
#endif
    spline_edges1(g, et);
}

static void
shiftClusters (graph_t * g, pointf offset)
{
    int i;

    for (i = 1; i <= GD_n_cluster(g); i++) {
	shiftClusters (GD_clust(g)[i], offset);
    }

    GD_bb(g).UR.x -= offset.x;
    GD_bb(g).UR.y -= offset.y;
    GD_bb(g).LL.x -= offset.x;
    GD_bb(g).LL.y -= offset.y;
}

/* spline_edges:
 * Compute bounding box, translate graph to origin,
 * then construct all edges.
 */
void spline_edges(graph_t * g)
{
    node_t *n;
    pointf offset;

    compute_bb(g);
    offset.x = PS2INCH(GD_bb(g).LL.x);
    offset.y = PS2INCH(GD_bb(g).LL.y);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	ND_pos(n)[0] -= offset.x;
	ND_pos(n)[1] -= offset.y;
    }
	
    shiftClusters (g, GD_bb(g).LL);
    spline_edges0(g, true);
}

/* scaleEdge:
 * Scale edge by given factor.
 * Assume ED_spl != NULL.
 */
static void scaleEdge(edge_t * e, double xf, double yf)
{
    pointf *pt;
    bezier *bez;
    pointf delh, delt;

    delh.x = POINTS_PER_INCH * (ND_pos(aghead(e))[0] * (xf - 1.0));
    delh.y = POINTS_PER_INCH * (ND_pos(aghead(e))[1] * (yf - 1.0));
    delt.x = POINTS_PER_INCH * (ND_pos(agtail(e))[0] * (xf - 1.0));
    delt.y = POINTS_PER_INCH * (ND_pos(agtail(e))[1] * (yf - 1.0));

    bez = ED_spl(e)->list;
    for (size_t i = 0; i < ED_spl(e)->size; i++) {
	pt = bez->list;
	for (size_t j = 0; j < bez->size; j++) {
	    if (i == 0 && j == 0) {
		pt->x += delt.x;
		pt->y += delt.y;
	    }
	    else if (i == ED_spl(e)->size-1 && j == bez->size-1) {
		pt->x += delh.x;
		pt->y += delh.y;
	    }
	    else {
		pt->x *= xf;
		pt->y *= yf;
	    }
	    pt++;
	}
	if (bez->sflag) {
	    bez->sp.x += delt.x;
	    bez->sp.y += delt.y;
	}
	if (bez->eflag) {
	    bez->ep.x += delh.x;
	    bez->ep.y += delh.y;
	}
	bez++;
    }

    if (ED_label(e) && ED_label(e)->set) {
	ED_label(e)->pos.x *= xf;
	ED_label(e)->pos.y *= yf;
    }
    if (ED_head_label(e) && ED_head_label(e)->set) {
	ED_head_label(e)->pos.x += delh.x;
	ED_head_label(e)->pos.y += delh.y;
    }
    if (ED_tail_label(e) && ED_tail_label(e)->set) {
	ED_tail_label(e)->pos.x += delt.x;
	ED_tail_label(e)->pos.y += delt.y;
    }
}

/* scaleBB:
 * Scale bounding box of clusters of g by given factors.
 */
static void scaleBB(graph_t * g, double xf, double yf)
{
    int i;

    GD_bb(g).UR.x *= xf;
    GD_bb(g).UR.y *= yf;
    GD_bb(g).LL.x *= xf;
    GD_bb(g).LL.y *= yf;

    if (GD_label(g) && GD_label(g)->set) {
	GD_label(g)->pos.x *= xf;
	GD_label(g)->pos.y *= yf;
    }

    for (i = 1; i <= GD_n_cluster(g); i++)
	scaleBB(GD_clust(g)[i], xf, yf);
}

/* translateE:
 * Translate edge by offset.
 * Assume ED_spl(e) != NULL
 */
static void translateE(edge_t * e, pointf offset)
{
    pointf *pt;
    bezier *bez;

    bez = ED_spl(e)->list;
    for (size_t i = 0; i < ED_spl(e)->size; i++) {
	pt = bez->list;
	for (size_t j = 0; j < bez->size; j++) {
	    pt->x -= offset.x;
	    pt->y -= offset.y;
	    pt++;
	}
	if (bez->sflag) {
	    bez->sp.x -= offset.x;
	    bez->sp.y -= offset.y;
	}
	if (bez->eflag) {
	    bez->ep.x -= offset.x;
	    bez->ep.y -= offset.y;
	}
	bez++;
    }

    if (ED_label(e) && ED_label(e)->set) {
	ED_label(e)->pos.x -= offset.x;
	ED_label(e)->pos.y -= offset.y;
    }
    if (ED_xlabel(e) && ED_xlabel(e)->set) {
	ED_xlabel(e)->pos.x -= offset.x;
	ED_xlabel(e)->pos.y -= offset.y;
    }
    if (ED_head_label(e) && ED_head_label(e)->set) {
	ED_head_label(e)->pos.x -= offset.x;
	ED_head_label(e)->pos.y -= offset.y;
    }
    if (ED_tail_label(e) && ED_tail_label(e)->set) {
	ED_tail_label(e)->pos.x -= offset.x;
	ED_tail_label(e)->pos.y -= offset.y;
    }
}

static void translateG(Agraph_t * g, pointf offset)
{
    int i;

    GD_bb(g).UR.x -= offset.x;
    GD_bb(g).UR.y -= offset.y;
    GD_bb(g).LL.x -= offset.x;
    GD_bb(g).LL.y -= offset.y;

    if (GD_label(g) && GD_label(g)->set) {
	GD_label(g)->pos.x -= offset.x;
	GD_label(g)->pos.y -= offset.y;
    }

    for (i = 1; i <= GD_n_cluster(g); i++)
	translateG(GD_clust(g)[i], offset);
}

void neato_translate(Agraph_t * g)
{
    node_t *n;
    edge_t *e;
    pointf offset;
    pointf ll;

    ll = GD_bb(g).LL;

    offset.x = PS2INCH(ll.x);
    offset.y = PS2INCH(ll.y);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	ND_pos(n)[0] -= offset.x;
	ND_pos(n)[1] -= offset.y;
	if (ND_xlabel(n) && ND_xlabel(n)->set) {
	    ND_xlabel(n)->pos.x -= ll.x;
	    ND_xlabel(n)->pos.y -= ll.y;
	}
    }
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e))
	    if (ED_spl(e))
		translateE(e, ll);
    }
    translateG(g, ll);
}

/* _neato_set_aspect;
 * Assume all bounding boxes are correct.
 * Return false if no transform is performed. This includes
 * the possibility that a translation was done.
 */
static bool _neato_set_aspect(graph_t * g)
{
    double xf, yf, actual, desired;
    node_t *n;
    bool translated = false;

    if (g->root != g)
	return false;

    if (GD_drawing(g)->ratio_kind) {
	if (GD_bb(g).LL.x || GD_bb(g).LL.y) {
	    translated = true;
	    neato_translate (g);
	}
	/* normalize */
	if (GD_flip(g)) {
	    GD_bb(g).UR = exch_xyf(GD_bb(g).UR);
	}
	if (GD_drawing(g)->ratio_kind == R_FILL) {
	    /* fill is weird because both X and Y can stretch */
	    if (GD_drawing(g)->size.x <= 0)
		return translated;
	    xf = GD_drawing(g)->size.x / GD_bb(g).UR.x;
	    yf = GD_drawing(g)->size.y / GD_bb(g).UR.y;
	    /* handle case where one or more dimensions is too big */
	    if (xf < 1.0 || yf < 1.0) {
		if (xf < yf) {
		    yf /= xf;
		    xf = 1.0;
		} else {
		    xf /= yf;
		    yf = 1.0;
		}
	    }
	} else if (GD_drawing(g)->ratio_kind == R_EXPAND) {
	    if (GD_drawing(g)->size.x <= 0)
		return translated;
	    xf = GD_drawing(g)->size.x / GD_bb(g).UR.x;
	    yf = GD_drawing(g)->size.y / GD_bb(g).UR.y;
	    if (xf > 1.0 && yf > 1.0) {
		double scale = fmin(xf, yf);
		xf = yf = scale;
	    } else
		return translated;
	} else if (GD_drawing(g)->ratio_kind == R_VALUE) {
	    desired = GD_drawing(g)->ratio;
	    actual = GD_bb(g).UR.y / GD_bb(g).UR.x;
	    if (actual < desired) {
		yf = desired / actual;
		xf = 1.0;
	    } else {
		xf = actual / desired;
		yf = 1.0;
	    }
	} else
	    return translated;
	if (GD_flip(g)) {
	    double t = xf;
	    xf = yf;
	    yf = t;
	}

	if (Nop > 1) {
	    edge_t *e;
	    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
		for (e = agfstout(g, n); e; e = agnxtout(g, e))
		    if (ED_spl(e))
			scaleEdge(e, xf, yf);
	    }
	}
	/* Not relying on neato_nlist here allows us not to have to 
	 * allocate it in the root graph and the connected components. 
	 */
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    ND_pos(n)[0] *= xf;
	    ND_pos(n)[1] *= yf;
	}
	scaleBB(g, xf, yf);
        return true;
    }
    else
	return false;
}

/* neato_set_aspect:
 * Sets aspect ratio if necessary; real work done in _neato_set_aspect;
 * This also copies the internal layout coordinates (ND_pos) to the 
 * external ones (ND_coord).
 *
 * Return true if some node moved.
 */
bool neato_set_aspect(graph_t * g)
{
    node_t *n;
    bool moved = false;

	/* setting aspect ratio only makes sense on root graph */
    moved = _neato_set_aspect(g);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	ND_coord(n).x = POINTS_PER_INCH * ND_pos(n)[0];
	ND_coord(n).y = POINTS_PER_INCH * ND_pos(n)[1];
    }
    return moved;
}

