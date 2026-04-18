/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/* adjust.c
 * Routines for repositioning nodes after initial layout in
 * order to reduce/remove node overlaps.
 */

#include <assert.h>
#include <cgraph/gv_ctype.h>
#include <neatogen/neato.h>
#include <common/utils.h>
#include <float.h>
#include <math.h>
#include <neatogen/voronoi.h>
#include <neatogen/info.h>
#include <neatogen/edges.h>
#include <neatogen/site.h>
#include <neatogen/hedges.h>
#include <neatogen/digcola.h>
#if ((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))
#include <neatogen/overlap.h>
#endif
#include <stdbool.h>
#ifdef IPSEPCOLA
#include <vpsc/csolve_VPSC.h>
#include <neatogen/quad_prog_vpsc.h>
#endif
#include <stddef.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/startswith.h>
#include <util/strcasecmp.h>

#define SEPFACT         0.8 // default esep/sep

static const double incr = 0.05;	/* Increase bounding box by adding
				 * incr * dimension around box.
				 */

typedef struct {
  Site **sites; ///< array of pointers to sites; used in qsort
  Site **endSite; ///< sentinel on sites array
  Point nw, ne, sw, se; ///< corners of clipping window
  Site **nextSite;
} state_t;

static void setBoundBox(state_t *st, Point *ll, Point *ur) {
    pxmin = ll->x;
    pxmax = ur->x;
    pymin = ll->y;
    pymax = ur->y;
    st->nw.x = st->sw.x = pxmin;
    st->ne.x = st->se.x = pxmax;
    st->nw.y = st->ne.y = pymax;
    st->sw.y = st->se.y = pymin;
}

/// Free node resources.
static void freeNodes(void)
{
    for (size_t i = 0; i < nsites; i++) {
	breakPoly(&nodeInfo[i].poly);
    }
    polyFree();
    if (nodeInfo != NULL) {
	free(nodeInfo->verts); // Free vertices
    }
    free(nodeInfo);
}

/*   Compute extremes of graph, then set up bounding box.
 *   If user supplied a bounding box, use that;
 *   else if "window" is a graph attribute, use that; 
 *   otherwise, define bounding box as a percentage expansion of
 *   graph extremes.
 *   In the first two cases, check that graph fits in bounding box.
 */
static void chkBoundBox(state_t *st, Agraph_t *graph) {
    Point ll, ur;

    double x_min = DBL_MAX;
    double y_min = DBL_MAX;
    double x_max = -DBL_MAX;
    double y_max = -DBL_MAX;
    assert(nsites > 0);
    for (size_t i = 0; i < nsites; ++i) {
	Info_t *ip = &nodeInfo[i];
	Poly *pp = &ip->poly;
	double x = ip->site.coord.x;
	double y = ip->site.coord.y;
	x_min = fmin(x_min, pp->origin.x + x);
	y_min = fmin(y_min, pp->origin.y + y);
	x_max = fmax(x_max, pp->corner.x + x);
	y_max = fmax(y_max, pp->corner.y + y);
    }

    // create initial bounding box by adding margin × dimension around box
    // enclosing nodes
    char *marg = agget(graph, "voro_margin");
    const double margin = (marg && *marg != '\0') ? atof(marg) : 0.05;
    double ydelta = margin * (y_max - y_min);
    double xdelta = margin * (x_max - x_min);
    ll.x = x_min - xdelta;
    ll.y = y_min - ydelta;
    ur.x = x_max + xdelta;
    ur.y = y_max + ydelta;

    setBoundBox(st, &ll, &ur);
}

 /// For each node in the graph, create a Info data structure 
static int makeInfo(Agraph_t * graph)
{
    int (*polyf)(Poly *, Agnode_t *, double, double);

    assert(agnnodes(graph) >= 0);
    nsites = (size_t)agnnodes(graph);
    geominit();

    nodeInfo = gv_calloc(nsites, sizeof(Info_t));

    Agnode_t *node = agfstnode(graph);

    expand_t pmargin = sepFactor (graph);

    if (pmargin.doAdd) {
	polyf = makeAddPoly;
	/* we need inches for makeAddPoly */
	pmargin.x = PS2INCH(pmargin.x);
	pmargin.y = PS2INCH(pmargin.y);
    }
	
    else polyf = makePoly;
    for (size_t i = 0; i < nsites; i++) {
	Info_t *ip = &nodeInfo[i];
	ip->site.coord.x = ND_pos(node)[0];
	ip->site.coord.y = ND_pos(node)[1];

	if (polyf(&ip->poly, node, pmargin.x, pmargin.y)) {
	    free (nodeInfo);
	    nodeInfo = NULL;
	    return 1;
        }

	ip->site.sitenbr = i;
	ip->site.refcnt = 1;
	ip->node = node;
	ip->verts = NULL;
	ip->n_verts = 0;
	node = agnxtnode(graph, node);
    }
    return 0;
}

/* sort sites on y, then x, coord */
static int scomp(const void *S1, const void *S2)
{
    const Site *s1 = *(Site *const *)S1;
    const Site *s2 = *(Site *const *)S2;
    if (s1->coord.y < s2->coord.y)
	return -1;
    if (s1->coord.y > s2->coord.y)
	return 1;
    if (s1->coord.x < s2->coord.x)
	return -1;
    if (s1->coord.x > s2->coord.x)
	return 1;
    return 0;
}

 /// Fill array of pointer to sites and sort the sites using scomp
static void sortSites(state_t *st) {
    if (st->sites == NULL) {
	st->sites = gv_calloc(nsites, sizeof(Site*));
	st->endSite = st->sites + nsites;
    }

    for (size_t i = 0; i < nsites; i++) {
	Info_t *ip = &nodeInfo[i];
	st->sites[i] = &ip->site;
	ip->verts = NULL;
	ip->n_verts = 0;
	ip->site.refcnt = 1;
    }

    qsort(st->sites, nsites, sizeof(Site *), scomp);

    /* Reset site index for nextOne */
    st->nextSite = st->sites;

}

static void geomUpdate(state_t *st, int doSort) {
    if (doSort)
	sortSites(st);

    /* compute ranges */
    xmin = DBL_MAX;
    xmax = -DBL_MAX;
    assert(nsites > 0);
    for (size_t i = 0; i < nsites; ++i) {
	xmin = fmin(xmin, st->sites[i]->coord.x);
	xmax = fmax(xmax, st->sites[i]->coord.x);
    }
    ymin = st->sites[0]->coord.y;
    ymax = st->sites[nsites - 1]->coord.y;

    deltax = xmax - xmin;
}

static Site *nextOne(void *state) {
    state_t *st = state;
    if (st->nextSite < st->endSite) {
	return *st->nextSite++;
    } else
	return NULL;
}

/// Check for nodes with identical positions and tweak the positions.
static void rmEquality(state_t *st) {
    sortSites(st);

    for (Site **ip = st->sites; ip < st->endSite; ) {
	Site **jp = ip + 1;
	if (jp >= st->endSite ||
	    (*jp)->coord.x != (*ip)->coord.x ||
	    (*jp)->coord.y != (*ip)->coord.y) {
	    ip = jp;
	    continue;
	}

	/* Find first node kp with position different from ip */
	int cnt = 2;
	Site **kp = jp + 1;
	while (kp < st->endSite &&
	       (*kp)->coord.x == (*ip)->coord.x &&
	       (*kp)->coord.y == (*ip)->coord.y) {
	    cnt++;
	    jp = kp;
	    kp = jp + 1;
	}

	/* If next node exists and is on the same line */
	if (kp < st->endSite && (*kp)->coord.y == (*ip)->coord.y) {
	    const double xdel = ((*kp)->coord.x - (*ip)->coord.x) / cnt;
	    int i = 1;
	    for (jp = ip + 1; jp < kp; jp++) {
		(*jp)->coord.x += i * xdel;
		i++;
	    }
	} else {		/* nothing is to the right */
	    Info_t *info;
	    for (jp = ip + 1; jp < kp; ip++, jp++) {
		info = nodeInfo + (*ip)->sitenbr;
		double xdel = info->poly.corner.x - info->poly.origin.x;
		info = nodeInfo + (*jp)->sitenbr;
		xdel += info->poly.corner.x - info->poly.origin.x;
		(*jp)->coord.x = (*ip)->coord.x + xdel / 2;
	    }
	}
	ip = kp;
    }
}

/// Count number of node-node overlaps at iteration iter.
static unsigned countOverlap(unsigned iter) {
    unsigned count = 0;

    for (size_t i = 0; i < nsites; i++)
	nodeInfo[i].overlaps = false;

    for (size_t i = 0; i < nsites - 1; i++) {
	Info_t *ip = &nodeInfo[i];
	for (size_t j = i + 1; j < nsites; j++) {
	    Info_t *jp = &nodeInfo[j];
	    if (polyOverlap(ip->site.coord, &ip->poly, jp->site.coord, &jp->poly)) {
		count++;
		ip->overlaps = true;
		jp->overlaps = true;
	    }
	}
    }

    if (Verbose > 1)
	fprintf(stderr, "overlap [%u] : %u\n", iter, count);
    return count;
}

static void increaseBoundBox(state_t *st) {
    Point ur = {.x = pxmax, .y = pymax};
    Point ll = {.x = pxmin, .y = pymin};

    const double ydelta = incr * (ur.y - ll.y);
    const double xdelta = incr * (ur.x - ll.x);

    ur.x += xdelta;
    ur.y += ydelta;
    ll.x -= xdelta;
    ll.y -= ydelta;

    setBoundBox(st, &ll, &ur);
}

/// Area of triangle whose vertices are a,b,c
static double areaOf(Point a, Point b, Point c)
{
    return fabs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2;
}

/* Compute centroid of triangle with vertices a, b, c.
 * Return coordinates in x and y.
 */
static void centroidOf(Point a, Point b, Point c, double *x, double *y)
{
    *x = (a.x + b.x + c.x) / 3;
    *y = (a.y + b.y + c.y) / 3;
}

/* The new position is the centroid of the voronoi polygon. This is the weighted
 * sum of the centroids of a triangulation, normalized to the total area.
 */
static void newpos(Info_t * ip)
{
    const Point anchor = ip->verts[0];
    double totalArea = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double x;
    double y;

    for (size_t i = 1; i + 1 < ip->n_verts; ++i) {
	const Point p = ip->verts[i];
	const Point q = ip->verts[i + 1];
	const double area = areaOf(anchor, p, q);
	centroidOf(anchor, p, q, &x, &y);
	cx += area * x;
	cy += area * y;
	totalArea += area;
    }

    ip->site.coord.x = cx / totalArea;
    ip->site.coord.y = cy / totalArea;
}

 /* Add corners of clipping window to appropriate sites.
  * A site gets a corner if it is the closest site to that corner.
  */
static void addCorners(const state_t *st) {
    Info_t *ip = nodeInfo;
    Info_t *sws = ip;
    Info_t *nws = ip;
    Info_t *ses = ip;
    Info_t *nes = ip;
    double swd = dist_2(ip->site.coord, st->sw);
    double nwd = dist_2(ip->site.coord, st->nw);
    double sed = dist_2(ip->site.coord, st->se);
    double ned = dist_2(ip->site.coord, st->ne);

    for (size_t i = 1; i < nsites; i++) {
	ip = &nodeInfo[i];
	double d = dist_2(ip->site.coord, st->sw);
	if (d < swd) {
	    swd = d;
	    sws = ip;
	}
	d = dist_2(ip->site.coord, st->se);
	if (d < sed) {
	    sed = d;
	    ses = ip;
	}
	d = dist_2(ip->site.coord, st->nw);
	if (d < nwd) {
	    nwd = d;
	    nws = ip;
	}
	d = dist_2(ip->site.coord, st->ne);
	if (d < ned) {
	    ned = d;
	    nes = ip;
	}
    }

    addVertex(&sws->site, st->sw.x, st->sw.y);
    addVertex(&ses->site, st->se.x, st->se.y);
    addVertex(&nws->site, st->nw.x, st->nw.y);
    addVertex(&nes->site, st->ne.x, st->ne.y);
}

 /* Calculate the new position of a site as the centroid
  * of its voronoi polygon, if it overlaps other nodes.
  * The polygons are finite by being clipped to the clipping
  * window.
  * We first add the corner of the clipping windows to the
  * vertex lists of the appropriate sites.
  *
  * @param st Algorithm state
  * @param doAll Move all nodes, regardless of overlap
  */
static void newPos(const state_t *st, bool doAll) {
    addCorners(st);
    for (size_t i = 0; i < nsites; i++) {
	Info_t *ip = &nodeInfo[i];
	if (doAll || ip->overlaps)
	    newpos(ip);
    }
}

/* Cleanup voronoi memory.
 * Note that ELcleanup relies on the number
 * of sites, so should at least be reset every time we use vAdjust.
 * This could be optimized, over multiple components or
 * even multiple graphs, but probably not worth it.
 */
static void cleanup(void)
{
    ELcleanup();
    siteinit();			/* free memory */
    edgeinit();			/* free memory */
}

static int vAdjust(state_t *st) {
    unsigned iterCnt = 0;
    unsigned badLevel = 0;
    unsigned increaseCnt = 0;

    unsigned overlapCnt = countOverlap(iterCnt);

    if (overlapCnt == 0)
	return 0;

    rmEquality(st);
    geomUpdate(st, 0);
    voronoi(nextOne, st);
    for (bool doAll = false;;) {
	newPos(st, doAll);
	iterCnt++;

	const unsigned cnt = countOverlap(iterCnt);
	if (cnt == 0)
	    break;
	if (cnt >= overlapCnt)
	    badLevel++;
	else
	    badLevel = 0;
	overlapCnt = cnt;

	switch (badLevel) {
	case 0:
	    doAll = true;
	    break;
	default:
	    doAll = true;
	    increaseCnt++;
	    increaseBoundBox(st);
	    break;
	}

	geomUpdate(st, 1);
	voronoi(nextOne, st);
    }

    if (Verbose) {
	fprintf(stderr, "Number of iterations = %u\n", iterCnt);
	fprintf(stderr, "Number of increases = %u\n", increaseCnt);
    }

    cleanup();
    return 1;
}

static void rePos(void) {
    double f = 1.0 + incr;

    for (size_t i = 0; i < nsites; i++) {
	Info_t *ip = &nodeInfo[i];
	ip->site.coord.x *= f;
	ip->site.coord.y *= f;
    }
}

static int sAdjust(state_t *st) {
    unsigned iterCnt = 0;

    const unsigned overlapCnt = countOverlap(iterCnt);

    if (overlapCnt == 0)
	return 0;

    rmEquality(st);
    while (1) {
	rePos();
	iterCnt++;

	const unsigned cnt = countOverlap(iterCnt);
	if (cnt == 0)
	    break;
    }

    if (Verbose) {
	fprintf(stderr, "Number of iterations = %u\n", iterCnt);
    }

    return 1;
}

/// Enter new node positions into the graph
static void updateGraph(void)
{
    for (size_t i = 0; i < nsites; i++) {
	Info_t *ip = &nodeInfo[i];
	ND_pos(ip->node)[0] = ip->site.coord.x;
	ND_pos(ip->node)[1] = ip->site.coord.y;
    }
}

#define ELS "|edgelabel|"
  /* Return true if node name starts with ELS */
#define IS_LNODE(n) startswith(agnameof(n), ELS)

/// Set up array of half sizes in inches.
double *getSizes(Agraph_t * g, pointf pad, int* n_elabels, int** elabels)
{
    double *sizes = gv_calloc(Ndim * agnnodes(g), sizeof(double));
    int nedge_nodes = 0;

    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if (elabels && IS_LNODE(n)) nedge_nodes++;

	const int i = ND_id(n);
	sizes[i * Ndim] = ND_width(n) * .5 + pad.x;
	sizes[i * Ndim + 1] = ND_height(n) * .5 + pad.y;
    }

    if (elabels && nedge_nodes) {
	int* elabs = gv_calloc(nedge_nodes, sizeof(int));
	nedge_nodes = 0;
	for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    if (IS_LNODE(n))
		elabs[nedge_nodes++] = ND_id(n);
	}
	*elabels = elabs;
	*n_elabels = nedge_nodes;
    }

    return sizes;
}

/* Assumes g is connected and simple, i.e., we can have a->b and b->a
 * but not a->b and a->b
 */
SparseMatrix makeMatrix(Agraph_t *g) {
    if (!g)
	return NULL;
    const int nnodes = agnnodes(g);
    const int nedges = agnedges(g);

    /* Assign node ids */
    int i = 0;
    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n))
	ND_id(n) = i++;

    int *I = gv_calloc(nedges, sizeof(int));
    int *J = gv_calloc(nedges, sizeof(int));
    double *val = gv_calloc(nedges, sizeof(double));

    Agsym_t *sym = agfindedgeattr(g, "weight");

    i = 0;
    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	const int row = ND_id(n);
	for (Agedge_t *e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    I[i] = row;
	    J[i] = ND_id(aghead(e));
	    double v;
	    if (!sym || sscanf(agxget(e, sym), "%lf", &v) != 1)
		v = 1;
	    val[i] = v;
	/* edge length */
	    i++;
	}
    }

    SparseMatrix A = SparseMatrix_from_coordinate_arrays(nedges, nnodes, nnodes,
                                                         I, J, val,
                                                         MATRIX_TYPE_REAL,
                                                         sizeof(double));

    free(I);
    free(J);
    free(val);

    return A;
}

#if ((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))
static void fdpAdjust(graph_t *g, adjust_data *am) {
    SparseMatrix A0 = makeMatrix(g);
    SparseMatrix A = A0;
    double *pos = gv_calloc(Ndim * agnnodes(g), sizeof(double));
    expand_t sep = sepFactor(g);
    pointf pad;

    if (sep.doAdd) {
	pad.x = PS2INCH(sep.x);
	pad.y = PS2INCH(sep.y);
    } else {
	pad.x = PS2INCH(DFLT_MARGIN);
	pad.y = PS2INCH(DFLT_MARGIN);
    }
    double *sizes = getSizes(g, pad, NULL, NULL);

    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	double* npos = pos + Ndim * ND_id(n);
	for (int i = 0; i < Ndim; i++) {
	    npos[i] = ND_pos(n)[i];
	}
    }

    if (!SparseMatrix_is_symmetric(A, false) || A->type != MATRIX_TYPE_REAL) {
	A = SparseMatrix_get_real_adjacency_matrix_symmetrized(A);
    } else {
	A = SparseMatrix_remove_diagonal(A);
    }

    remove_overlap(Ndim, A, pos, sizes, am->value, am->scaling, 
                   ELSCHEME_NONE, 0, NULL, NULL,
                   mapBool(agget(g, "overlap_shrink"), true));

    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	double *npos = pos + Ndim * ND_id(n);
	for (int i = 0; i < Ndim; i++) {
	    ND_pos(n)[i] = npos[i];
	}
    }

    free(sizes);
    free(pos);
    if (A != A0)
	SparseMatrix_delete(A);
    SparseMatrix_delete (A0);
}
#endif

#ifdef IPSEPCOLA
static int
vpscAdjust(graph_t* G)
{
    enum { dim = 2 };
    int nnodes = agnnodes(G);
    ipsep_options opt;
    pointf *nsize = gv_calloc(nnodes, sizeof(pointf));
    float* coords[dim];
    float *f_storage = gv_calloc(dim * nnodes, sizeof(float));

    for (size_t i = 0; i < dim; i++) {
	coords[i] = f_storage + i * nnodes;
    }

    size_t j = 0;
    for (Agnode_t *v = agfstnode(G); v; v = agnxtnode(G, v)) {
	for (size_t i = 0; i < dim; i++) {
	    coords[i][j] =  (float)ND_pos(v)[i];
	}
	nsize[j].x = ND_width(v);
	nsize[j].y = ND_height(v);
	j++;
    }

    opt.diredges = 0;
    opt.edge_gap = 0;
    opt.noverlap = 2;
    opt.clusters = (cluster_data){0};
    expand_t exp_margin = sepFactor (G);
 	/* Multiply by 2 since opt.gap is the gap size, not the margin */
    if (exp_margin.doAdd) {
	opt.gap.x = 2.0*PS2INCH(exp_margin.x);
	opt.gap.y = 2.0*PS2INCH(exp_margin.y);
    }
    else {
	opt.gap.x = opt.gap.y = 2.0*PS2INCH(DFLT_MARGIN);
    }
    opt.nsize = nsize;

    removeoverlaps(nnodes, coords, &opt);

    j = 0;
    for (Agnode_t *v = agfstnode(G); v; v = agnxtnode(G, v)) {
	for (size_t i = 0; i < dim; i++) {
	    ND_pos(v)[i] = coords[i][j];
	}
	j++;
    }

    free (f_storage);
    free (nsize);
    return 0;
}
#endif

/* Return true if "normalize" is defined and valid; return angle in phi.
 * Read angle as degrees, convert to radians.
 * Guarantee -PI < phi <= PI.
 */
static int
angleSet (graph_t* g, double* phi)
{
    char* p;
    char* a = agget(g, "normalize");

    if (!a || *a == '\0')
	return 0;
    double ang = strtod (a, &p);
    if (p == a) {  /* no number */
	if (mapbool(a))
	    ang = 0.0;
	else
	    return 0;
    }
    while (ang > 180) ang -= 360;
    while (ang <= -180) ang += 360;

    *phi = RADIANS(ang);
    return 1;
}

/* If normalize is set, move first node to origin, then
 * rotate graph so that the angle of the first edge is given
 * by the degrees from normalize.
 * FIX: Generalize to allow rotation determined by graph shape.
 */
int normalize(graph_t * g)
{
    double phi;
    int ret;

    if (!angleSet(g, &phi))
	return 0;

    node_t *v = agfstnode(g);
    pointf p = {.x = ND_pos(v)[0], .y = ND_pos(v)[1]};
    for (v = agfstnode(g); v; v = agnxtnode(g, v)) {
	ND_pos(v)[0] -= p.x;
	ND_pos(v)[1] -= p.y;
    }
    if (p.x || p.y) ret = 1;
    else ret = 0;

    edge_t *e = NULL;
    for (v = agfstnode(g); v; v = agnxtnode(g, v))
	if ((e = agfstout(g, v)))
	    break;
    if (e == NULL)
	return ret;

	/* rotation necessary; pos => ccw */
    phi -= atan2(ND_pos(aghead(e))[1] - ND_pos(agtail(e))[1],
		   ND_pos(aghead(e))[0] - ND_pos(agtail(e))[0]);

    if (phi) {
	const pointf orig = {.x = ND_pos(agtail(e))[0],
	                     .y = ND_pos(agtail(e))[1]};
	const double cosv = cos(phi);
	const double sinv = sin(phi);
	for (v = agfstnode(g); v; v = agnxtnode(g, v)) {
	    p.x = ND_pos(v)[0] - orig.x;
	    p.y = ND_pos(v)[1] - orig.y;
	    ND_pos(v)[0] = p.x * cosv - p.y * sinv + orig.x;
	    ND_pos(v)[1] = p.x * sinv + p.y * cosv + orig.y;
	}
	return 1;
    }
    else return ret;
}

typedef struct {
    adjust_mode mode;
    char *attrib;
    char *print;
} lookup_t;

/* Translation table from overlap values to algorithms.
 * adjustMode[0] corresponds to overlap=true
 * adjustMode[1] corresponds to overlap=false
 */
static const lookup_t adjustMode[] = {
    {AM_NONE, "", "none"},
#if ((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))
    {AM_PRISM, "prism", "prism"},
#endif
    {AM_VOR, "voronoi", "Voronoi"},
    {AM_NSCALE, "scale", "scaling"},
    {AM_COMPRESS, "compress", "compress"},
    {AM_VPSC, "vpsc", "vpsc"},
    {AM_IPSEP, "ipsep", "ipsep"},
    {AM_SCALE, "oscale", "old scaling"},
    {AM_SCALEXY, "scalexy", "x and y scaling"},
    {AM_ORTHO, "ortho", "orthogonal constraints"},
    {AM_ORTHO_YX, "ortho_yx", "orthogonal constraints"},
    {AM_ORTHOXY, "orthoxy", "xy orthogonal constraints"},
    {AM_ORTHOYX, "orthoyx", "yx orthogonal constraints"},
    {AM_PORTHO, "portho", "pseudo-orthogonal constraints"},
    {AM_PORTHO_YX, "portho_yx", "pseudo-orthogonal constraints"},
    {AM_PORTHOXY, "porthoxy", "xy pseudo-orthogonal constraints"},
    {AM_PORTHOYX, "porthoyx", "yx pseudo-orthogonal constraints"},
#if !((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))
    {AM_PRISM, "prism", 0},
#endif
    {0}
};

/// Initialize and set prism values
static void setPrismValues(Agraph_t *g, const char *s, adjust_data *dp) {
    int v;

    if (sscanf (s, "%d", &v) > 0 && v >= 0)
	dp->value = v;
    else
	dp->value = 1000;
    dp->scaling = late_double(g, agfindgraphattr(g, "overlap_scaling"), -4.0, -1.e10);
}

/// Convert string value to internal value of adjustment mode.
static void getAdjustMode(Agraph_t *g, const char *s, adjust_data *dp) {
    const lookup_t *ap = adjustMode + 1;
    if (s == NULL || *s == '\0') {
	dp->mode = adjustMode[0].mode;
	dp->print = adjustMode[0].print;
    }
    else {
	while (ap->attrib) {
	    bool matches = strcasecmp(s, ap->attrib) == 0;
	    // "prism" takes parameters, so needs to match "prism.*"
	    matches |= ap->mode == AM_PRISM
	            && strncasecmp(s, ap->attrib, strlen(ap->attrib)) == 0;
	    if (matches) {
		if (ap->print == NULL) {
		    agwarningf("Overlap value \"%s\" unsupported - ignored\n", ap->attrib);
		    ap = &adjustMode[1];
		}
		dp->mode = ap->mode;
		dp->print = ap->print;
		if (ap->mode == AM_PRISM)
		    setPrismValues(g, s + strlen(ap->attrib), dp);
		break;
	    }
	    ap++;
	}
	if (ap->attrib == NULL ) {
	    bool v = mapbool(s);
	    bool unmappable = v != mapBool(s, true);
	    if (unmappable) {
		agwarningf("Unrecognized overlap value \"%s\" - using false\n", s);
		v = false;
	    }
	    if (v) {
		dp->mode = adjustMode[0].mode;
		dp->print = adjustMode[0].print;
	    }
	    else {
		dp->mode = adjustMode[1].mode;
		dp->print = adjustMode[1].print;
	    }
	    if (dp->mode == AM_PRISM)
		setPrismValues (g, "", dp);
	}
    }
    if (Verbose) {
	fprintf(stderr, "overlap: %s value %d scaling %.04f\n", dp->print, dp->value, dp->scaling);
    }
}

void graphAdjustMode(graph_t *G, adjust_data *dp, char *dflt) {
    char* am = agget(G, "overlap");
    getAdjustMode (G, am ? am : (dflt ? dflt : ""), dp);
}

#define ISZERO(d) (fabs(d) < 0.000000001)

static int simpleScale (graph_t* g) 
{
    pointf sc;
    int i;
    char* p;

    if ((p = agget(g, "scale"))) {
	if ((i = sscanf(p, "%lf,%lf", &sc.x, &sc.y))) {
	    if (ISZERO(sc.x)) return 0;
	    if (i == 1) sc.y = sc.x;
	    else if (ISZERO(sc.y)) return 0;
	    if (sc.y == 1 && sc.x == 1) return 0;
	    if (Verbose)
		fprintf (stderr, "scale = (%.03f,%.03f)\n", sc.x, sc.y);
	    for (node_t *n = agfstnode(g); n; n = agnxtnode(g,n)) {
		ND_pos(n)[0] *= sc.x;
		ND_pos(n)[1] *= sc.y;
	    }
	    return 1;
	}
    }
    return 0;
}

/* Use adjust_data to determine if and how to remove
 * node overlaps.
 * Return non-zero if nodes are moved.
 */
int 
removeOverlapWith (graph_t * G, adjust_data* am)
{
    int ret;

    if (agnnodes(G) < 2)
	return 0;

    int nret = normalize (G);
    nret += simpleScale (G);

    if (am->mode == AM_NONE)
	return nret;

    if (Verbose)
	fprintf(stderr, "Adjusting %s using %s\n", agnameof(G), am->print);

    if (am->mode > AM_SCALE) {
	switch (am->mode) {
	case AM_NSCALE:
	    ret = scAdjust(G, 1);
	    break;
	case AM_SCALEXY:
	    ret = scAdjust(G, 0);
	    break;
	case AM_PUSH:
        ret = 0;
	    break;
	case AM_PUSHPULL:
        ret = 0;
	    break;
	case AM_PORTHO_YX:
	case AM_PORTHO:
	case AM_PORTHOXY:
	case AM_PORTHOYX:
	case AM_ORTHO_YX:
	case AM_ORTHO:
	case AM_ORTHOXY:
	case AM_ORTHOYX:
	    cAdjust(G, am->mode);
        ret = 0;
	    break;
	case AM_COMPRESS:
	    ret = scAdjust(G, -1);
	    break;
#if ((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))
	case AM_PRISM:
	    fdpAdjust(G, am);
	    ret = 0;
	    break;
#endif
#ifdef IPSEPCOLA
	case AM_IPSEP:
	    return nret;   /* handled during layout */
	    break;
	case AM_VPSC:
	    ret = vpscAdjust(G);
	    break;
#endif
	default:		/* to silence warnings */
	    if (am->mode != AM_VOR && am->mode != AM_SCALE)
		agwarningf("Unhandled adjust option %s\n", am->print);
	    ret = 0;
	    break;
	}
	return nret+ret;
    }

    /* create main array */
    if (makeInfo(G)) {
	freeNodes();
	return nret;
    }

    /* establish and verify bounding box */
    state_t st = {0};
    chkBoundBox(&st, G);

    if (am->mode == AM_SCALE)
	ret = sAdjust(&st);
    else
	ret = vAdjust(&st);

    if (ret)
	updateGraph();

    freeNodes();
    free(st.sites);

    return ret+nret;
}

/// Use flag value to determine if and how to remove node overlaps.
int 
removeOverlapAs(graph_t * G, char* flag)
{
    adjust_data am;

    if (agnnodes(G) < 2)
	return 0;
    getAdjustMode(G, flag, &am);
    return removeOverlapWith (G, &am);
}

/* Remove node overlap relying on graph's overlap attribute.
 * Return non-zero if graph has changed.
 */
int adjustNodes(graph_t * G)
{
    return removeOverlapAs(G, agget(G, "overlap"));
}

/* Convert "sep" attribute into expand_t.
 * Input "+x,y" becomes {x,y,true}
 * Input "x,y" becomes {1 + x/sepfact,1 + y/sepfact,false}
 * Return 1 on success, 0 on failure
 */
static int parseFactor(char *s, expand_t *pp, double sepfact, double dflt) {
    int i;

    while (gv_isspace(*s)) s++;
    if (*s == '+') {
	s++;
	pp->doAdd = true;
    }
    else pp->doAdd = false;

    double x, y;
    if ((i = sscanf(s, "%lf,%lf", &x, &y))) {
	if (i == 1) y = x;
	if (pp->doAdd) {
	    if (sepfact > 1) {
		pp->x = fmin(dflt, x / sepfact);
		pp->y = fmin(dflt, y / sepfact);
	    }
	    else if (sepfact < 1) {
		pp->x = fmax(dflt, x / sepfact);
		pp->y = fmax(dflt, y / sepfact);
	    }
	    else {
		pp->x = x;
		pp->y = y;
	    }
	}
	else {
	    pp->x = 1.0 + x / sepfact;
	    pp->y = 1.0 + y / sepfact;
	}
	return 1;
    }
    else return 0;
}

expand_t
sepFactor(graph_t* g)
{
    expand_t pmargin;
    char*  marg;

    if ((marg = agget(g, "sep")) && parseFactor(marg, &pmargin, 1.0, 0)) {
    }
    else if ((marg = agget(g, "esep")) && parseFactor(marg, &pmargin, SEPFACT, DFLT_MARGIN)) {
    }
    else { /* default */
	pmargin.x = pmargin.y = DFLT_MARGIN;
	pmargin.doAdd = true;
    }
    if (Verbose)
	fprintf (stderr, "Node separation: add=%d (%f,%f)\n",
	    pmargin.doAdd, pmargin.x, pmargin.y);
    return pmargin;
}

/* This value should be smaller than the sep value used to expand
 * nodes during adjustment. If not, when the adjustment pass produces
 * a fairly tight layout, the spline code will find that some nodes
 * still overlap.
 */
expand_t
esepFactor(graph_t* g)
{
    expand_t pmargin;
    char*  marg;

    if ((marg = agget(g, "esep")) && parseFactor(marg, &pmargin, 1.0, 0)) {
    }
    else if ((marg = agget(g, "sep")) &&
             parseFactor(marg, &pmargin, 1.0 / SEPFACT, SEPFACT * DFLT_MARGIN)) {
    }
    else {
	pmargin.x = pmargin.y = SEPFACT*DFLT_MARGIN;
	pmargin.doAdd = true;
    }
    if (Verbose)
	fprintf (stderr, "Edge separation: add=%d (%f,%f)\n",
	    pmargin.doAdd, pmargin.x, pmargin.y);
    return pmargin;
}
