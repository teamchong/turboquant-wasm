/**
 * @file
 * @brief API neatogen/neatoprocs.h:
 * @ref neato_init_node, @ref user_pos, @ref neato_cleanup,
 * @ref init_nop, @ref setSeed, @ref checkStart, @ref neato_layout
 */

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

#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <neatogen/neato.h>
#include <pack/pack.h>
#include <neatogen/stress.h>
#ifdef DIGCOLA
#include <neatogen/digcola.h>
#endif
#include <neatogen/kkutils.h>
#include <common/pointset.h>
#include <common/render.h>
#include <common/utils.h>
#include <neatogen/sgd.h>
#include <cgraph/cgraph.h>
#include <cgraph/gv_ctype.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/bitarray.h>
#include <util/prisize_t.h>
#include <util/startswith.h>
#include <util/strcasecmp.h>
#include <util/streq.h>

#ifndef HAVE_SRAND48
#define srand48 srand
#endif

static attrsym_t *N_pos;
static int Pack;		/* If >= 0, layout components separately and pack together
				 * The value of Pack gives margins around graphs.
				 */
static char *cc_pfx = "_neato_cc";

void neato_init_node(node_t * n)
{
    agbindrec(n, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);	//node custom data
    common_init_node(n);
    ND_pos(n) = gv_calloc(GD_ndim(agraphof(n)), sizeof(double));
    gv_nodesize(n, GD_flip(agraphof(n)));
}

static void neato_init_edge(edge_t * e)
{
    agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//node custom data
    common_init_edge(e);
    ED_factor(e) = late_double(e, E_weight, 1.0, 1.0);
}

bool user_pos(attrsym_t *posptr, attrsym_t *pinptr, node_t *np, int nG) {
    double *pvec;
    char *p, c;
    double z;

    if (posptr == NULL)
	return false;
    pvec = ND_pos(np);
    p = agxget(np, posptr);
    if (p[0]) {
	c = '\0';
	if (Ndim >= 3 && sscanf(p, "%lf,%lf,%lf%c", pvec, pvec+1, pvec+2, &c) >= 3){
	    ND_pinned(np) = P_SET;
	    if (PSinputscale > 0.0) {
		int i;
		for (i = 0; i < Ndim; i++)
		    pvec[i] = pvec[i] / PSinputscale;
	    }
	    if (Ndim > 3)
		jitter_d(np, nG, 3);
	    if (c == '!' || (pinptr && mapbool(agxget(np, pinptr))))
		ND_pinned(np) = P_PIN;
	    return true;
	}
	else if (sscanf(p, "%lf,%lf%c", pvec, pvec + 1, &c) >= 2) {
	    ND_pinned(np) = P_SET;
	    if (PSinputscale > 0.0) {
		int i;
		for (i = 0; i < Ndim; i++)
		    pvec[i] /= PSinputscale;
	    }
	    if (Ndim > 2) {
		if (N_z && (p = agxget(np, N_z)) && sscanf(p,"%lf",&z) == 1) {
		    if (PSinputscale > 0.0) {
			pvec[2] = z / PSinputscale;
		    }
		    else
			pvec[2] = z;
		    jitter_d(np, nG, 3);
		}
		else
		    jitter3d(np, nG);
	    }
	    if (c == '!' || (pinptr && mapbool(agxget(np, pinptr))))
		ND_pinned(np) = P_PIN;
	    return true;
	} else
	    agerrorf("node %s, position %s, expected two doubles\n",
		  agnameof(np), p);
    }
    return false;
}

static void neato_init_node_edge(graph_t * g)
{
    node_t *n;
    edge_t *e;
    int nG = agnnodes(g);
    attrsym_t *N_pin;

    N_pos = agfindnodeattr(g, "pos");
    N_pin = agfindnodeattr(g, "pin");

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	neato_init_node(n);
	user_pos(N_pos, N_pin, n, nG);	/* set user position if given */
    }
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e))
	    neato_init_edge(e);
    }
}

static void neato_cleanup_graph(graph_t * g)
{
    if (Nop || Pack < 0) {
	free_scan_graph(g);
    }
    free(GD_clust(g));
}

void neato_cleanup(graph_t * g)
{
    node_t *n;
    edge_t *e;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    gv_cleanup_edge(e);
	}
	gv_cleanup_node(n);
    }
    neato_cleanup_graph(g);
}

static int numFields(const char *pos) {
    int cnt = 0;
    char c;

    do {
	while (gv_isspace(*pos))
	    pos++;		/* skip white space */
	if ((c = *pos)) { /* skip token */
	    cnt++;
	    while ((c = *pos) && !gv_isspace(c) && c != ';')
		pos++;
	}
    } while (gv_isspace(c));
    return cnt;
}

static void set_label(void* obj, textlabel_t * l, char *name)
{
    double x, y;
    char *lp;
    lp = agget(obj, name);
    if (lp && sscanf(lp, "%lf,%lf", &x, &y) == 2) {
	l->pos = (pointf){x, y};
	l->set = true;
    }
}

#ifdef IPSEPCOLA
static cluster_data cluster_map(graph_t *mastergraph, graph_t *g) {
    graph_t *subg;
    node_t *n;
     /* array of arrays of node indices in each cluster */
    int **cs,*cn;
    int i,j,nclusters=0;
    bitarray_t assigned = bitarray_new(agnnodes(g));
    cluster_data cdata = {0};

    cdata.ntoplevel = agnnodes(g);
    for (subg = agfstsubg(mastergraph); subg; subg = agnxtsubg(subg)) {
        if (is_a_cluster(subg)) {
            nclusters++;
        }
    }
    cdata.nvars=0;
    cdata.nclusters = nclusters;
    cs = cdata.clusters = gv_calloc(nclusters, sizeof(int*));
    cn = cdata.clustersizes = gv_calloc(nclusters, sizeof(int));
    for (subg = agfstsubg(mastergraph); subg; subg = agnxtsubg(subg)) {
        /* clusters are processed by separate calls to ordered_edges */
        if (is_a_cluster(subg)) {
            int *c;

            *cn = agnnodes(subg);
            cdata.nvars += *cn;
            c = *cs++ = gv_calloc(*cn++, sizeof(int));
            for (n = agfstnode(subg); n; n = agnxtnode(subg, n)) {
                node_t *gn;
                int ind = 0;
                for (gn = agfstnode(g); gn; gn = agnxtnode(g, gn)) {
                    if(AGSEQ(gn)==AGSEQ(n)) break;
                    ind++;
                }
                *c++=ind;
                bitarray_set(&assigned, ind, true);
                cdata.ntoplevel--;
            }
        }
    }
    cdata.bb = gv_calloc(cdata.nclusters, sizeof(boxf));
    cdata.toplevel = gv_calloc(cdata.ntoplevel, sizeof(int));
    for(i=j=0;i<agnnodes(g);i++) {
        if(!bitarray_get(assigned, i)) {
            cdata.toplevel[j++] = i;
        }
    }
    assert(cdata.ntoplevel == agnnodes(g) - cdata.nvars);
    bitarray_reset(&assigned);
    return cdata;
}

static void freeClusterData(cluster_data c) {
    if (c.nclusters > 0) {
        free(c.clusters[0]);
        free(c.clusters);
        free(c.clustersizes);
        free(c.toplevel);
        free(c.bb);
    }
}
#endif

/* user_spline:
 * Attempt to use already existing pos info for spline
 * Return 1 if successful, 0 otherwise.
 * Assume E_pos != NULL and ED_spl(e) == NULL.
 */
static int user_spline(attrsym_t * E_pos, edge_t * e)
{
    char *pos;
    int i, n, npts, nc;
    pointf *pp;
    double x, y;
    int sflag = 0, eflag = 0;
    pointf sp = { 0, 0 }, ep = { 0, 0};
    bezier *newspl;
    int more = 1;
    static bool warned;

    pos = agxget(e, E_pos);
    if (*pos == '\0')
	return 0;

    uint32_t stype, etype;
    arrow_flags(e, &stype, &etype);
    do {
	/* check for s head */
	i = sscanf(pos, "s,%lf,%lf%n", &x, &y, &nc);
	if (i == 2) {
	    sflag = 1;
	    pos = pos + nc;
	    sp.x = x;
	    sp.y = y;
	}

	/* check for e head */
	i = sscanf(pos, " e,%lf,%lf%n", &x, &y, &nc);
	if (i == 2) {
	    eflag = 1;
	    pos = pos + nc;
	    ep.x = x;
	    ep.y = y;
	}

	npts = numFields(pos); // count potential points
	n = npts;
	if (n < 4 || n % 3 != 1) {
	    gv_free_splines(e);
	    if (!warned) {
		warned = true;
		agwarningf("pos attribute for edge (%s,%s) doesn't have 3n+1 points\n", agnameof(agtail(e)), agnameof(aghead(e)));
	    }
	    return 0;
	}
	pointf *ps = gv_calloc(n, sizeof(pointf));
	pp = ps;
	while (n) {
	    i = sscanf(pos, "%lf,%lf%n", &x, &y, &nc);
	    if (i < 2) {
		if (!warned) {
		    warned = true;
		    agwarningf("syntax error in pos attribute for edge (%s,%s)\n", agnameof(agtail(e)), agnameof(aghead(e)));
		}
		free(ps);
		gv_free_splines(e);
		return 0;
	    }
	    pos += nc;
	    pp->x = x;
	    pp->y = y;
	    pp++;
	    n--;
	}
 	while (gv_isspace(*pos)) pos++;
	if (*pos == '\0')
	    more = 0;
	else
	    pos++;

	/* parsed successfully; create spline */
	assert(npts >= 0);
	newspl = new_spline(e, (size_t)npts);
	if (sflag) {
	    newspl->sflag = stype;
	    newspl->sp = sp;
	}
	if (eflag) {
	    newspl->eflag = etype;
	    newspl->ep = ep;
	}
	for (i = 0; i < npts; i++) {
	    newspl->list[i] = ps[i];
	}
	free(ps);
    } while (more);

    if (ED_label(e))
	set_label(e, ED_label(e), "lp");
    if (ED_xlabel(e))
	set_label(e, ED_xlabel(e), "xlp");
    if (ED_head_label(e))
	set_label(e, ED_head_label(e), "head_lp");
    if (ED_tail_label(e))
	set_label(e, ED_tail_label(e), "tail_lp");

    return 1;
}

/* Nop can be:
 *  0 - do full layout
 *  1 - assume initial node positions, do (optional) adjust and all splines
 *  2 - assume final node and edges positions, do nothing except compute
 *      missing splines
 */

 /* Indicates the amount of edges with position information */
typedef enum { NoEdges, SomeEdges, AllEdges } pos_edge;

/* nop_init_edges:
 * Check edges for position info.
 * If position info exists, check for edge label positions.
 * Return number of edges with position info.
 */
static pos_edge nop_init_edges(Agraph_t * g)
{
    node_t *n;
    edge_t *e;
    int nedges = 0;
    attrsym_t *E_pos;

    if (agnedges(g) == 0)
	return AllEdges;

    E_pos = agfindedgeattr(g, "pos");
    if (!E_pos || Nop < 2)
	return NoEdges;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    if (user_spline(E_pos, e)) {
		nedges++;
	    }
	}
    }
    if (nedges) {
	if (nedges == agnedges(g))
	    return AllEdges;
	else
	    return SomeEdges;
    } else
	return NoEdges;
}

/* freeEdgeInfo:
 */
static void freeEdgeInfo (Agraph_t * g)
{
    node_t *n;
    edge_t *e;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    gv_free_splines(e);
	    free_label(ED_label(e));
	    free_label(ED_xlabel(e));
	    free_label(ED_head_label(e));
	    free_label(ED_tail_label(e));
	}
    }
}

/* chkBB:
 * Scans for a correct bb attribute. If available, sets it
 * in the graph and returns 1.
 */
#define BS "%lf,%lf,%lf,%lf"

static int chkBB(Agraph_t * g, attrsym_t * G_bb, boxf* bbp)
{
    char *s;
    boxf bb;

    s = agxget(g, G_bb);
    if (sscanf(s, BS, &bb.LL.x, &bb.LL.y, &bb.UR.x, &bb.UR.y) == 4) {
	if (bb.LL.y > bb.UR.y) {
	/* If the LL.y coordinate is bigger than the UR.y coordinate,
         * we assume the input was produced using -y, so we normalize
	 * the bb.
	 */
	    double tmp = bb.LL.y;
	    bb.LL.y = bb.UR.y;
	    bb.UR.y = tmp;
	}
	*bbp = bb;
	return 1;
    } else
	return 0;
}

static void add_cluster(Agraph_t * g, Agraph_t * subg)
{
    int cno;
    cno = ++(GD_n_cluster(g));
    GD_clust(g) = gv_recalloc(GD_clust(g), GD_n_cluster(g), cno + 1,
                              sizeof(graph_t*));
    GD_clust(g)[cno] = subg;
    do_graph_label(subg);
}


static void nop_init_graphs(Agraph_t *, attrsym_t *, attrsym_t *);

/* dfs:
 * Process subgraph subg of parent graph g
 * If subg is a cluster, add its bounding box, if any; attach to
 * cluster array of parent, and recursively initialize subg.
 * If not a cluster, recursively call this function on the subgraphs
 * of subg, using parentg as the parent graph.
 */
static void
dfs(Agraph_t * subg, Agraph_t * parentg, attrsym_t * G_lp, attrsym_t * G_bb)
{
    boxf bb;

    if (is_a_cluster(subg) && chkBB(subg, G_bb, &bb)) {
	agbindrec(subg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
	GD_bb(subg) = bb;
	add_cluster(parentg, subg);
	nop_init_graphs(subg, G_lp, G_bb);
    } else {
	graph_t *sg;
	for (sg = agfstsubg(subg); sg; sg = agnxtsubg(sg)) {
	    dfs(sg, parentg, G_lp, G_bb);
	}
    }
}

/* nop_init_graphs:
 * Read in clusters and graph label info.
 * A subgraph is a cluster if its name starts with "cluster" and
 * it has a valid bb.
 */
static void
nop_init_graphs(Agraph_t * g, attrsym_t * G_lp, attrsym_t * G_bb)
{
    graph_t *subg;
    char *s;
    double x, y;

    if (GD_label(g) && G_lp) {
	s = agxget(g, G_lp);
	if (sscanf(s, "%lf,%lf", &x, &y) == 2) {
	    GD_label(g)->pos = (pointf){x, y};
	    GD_label(g)->set = true;
	}
    }

    if (!G_bb)
	return;
    for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	dfs(subg, g, G_lp, G_bb);
    }
}

/* init_nop:
 * This assumes all nodes have been positioned.
 * It also assumes none of the relevant fields in A*info_t have been set.
 * The input may provide additional position information for
 * clusters, edges and labels. If certain position information
 * is missing, init_nop will use a standard neato technique to
 * supply it.
 *
 * If adjust is false, init_nop does nothing but initialize all
 * of the basic graph information. No tweaking of positions or
 * filling in edge splines is done.
 *
 * Returns 0 on normal success, 1 if layout has a background, and -1
 * on failure.
 */
int init_nop(Agraph_t * g, int adjust)
{
    int i;
    node_t *np;
    pos_edge posEdges;		/* How many edges have spline info */
    attrsym_t *G_lp = agfindgraphattr(g, "lp");
    attrsym_t *G_bb = agfindgraphattr(g, "bb");
    int didAdjust = 0;  /* Have nodes been moved? */
    int haveBackground;
    bool translate = !mapbool(agget(g, "notranslate"));

    /* If G_bb not defined, define it */
    if (!G_bb)
	G_bb = agattr(g, AGRAPH, "bb", "");

    scan_graph(g);		/* mainly to set up GD_neato_nlist */
    for (i = 0; (np = GD_neato_nlist(g)[i]); i++) {
	if (!hasPos(np) && !startswith(agnameof(np), "cluster")) {
	    agerrorf("node %s in graph %s has no position\n",
		  agnameof(np), agnameof(g));
	    return -1;
	}
	if (ND_xlabel(np))
	    set_label(np, ND_xlabel(np), "xlp");
    }
    nop_init_graphs(g, G_lp, G_bb);
    posEdges = nop_init_edges(g);

    if (GD_drawing(g)->xdots) {
	haveBackground = 1;
	GD_drawing(g)->ratio_kind = R_NONE; /* Turn off any aspect change if background present */
    }
    else
	haveBackground = 0;

    if (adjust && Nop == 1 && !haveBackground)
	didAdjust = adjustNodes(g);

    if (didAdjust) {
	if (GD_label(g)) GD_label(g)->set = false;
/* FIX:
 *   - if nodes are moved, clusters are no longer valid.
 */
    }

    compute_bb(g);

    /* Adjust bounding box for any background */
    if (haveBackground)
	GD_bb(g) = xdotBB (g);

    /* At this point, all bounding boxes should be correctly defined.
     */

    if (!adjust) {
	node_t *n;
	State = GVSPLINES;
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    ND_coord(n).x = POINTS_PER_INCH * ND_pos(n)[0];
	    ND_coord(n).y = POINTS_PER_INCH * ND_pos(n)[1];
	}
    }
    else {
	bool didShift;
	if (translate && !haveBackground && (GD_bb(g).LL.x != 0||GD_bb(g).LL.y != 0))
	    neato_translate (g);
	didShift = neato_set_aspect(g);
	/* if we have some edge positions and we either shifted or adjusted, free edge positions */
	if (posEdges != NoEdges && (didShift || didAdjust)) {
	    freeEdgeInfo (g);
	    posEdges = NoEdges;
	}
	if (posEdges != AllEdges)
	    spline_edges0(g, false);   /* add edges */
	else
	    State = GVSPLINES;
    }

    return haveBackground;
}

static void neato_init_graph (Agraph_t * g)
{
    int outdim;

    setEdgeType (g, EDGETYPE_LINE);
    outdim = late_int(g, agfindgraphattr(g, "dimen"), 2, 2);
    GD_ndim(agroot(g)) = late_int(g, agfindgraphattr(g, "dim"), outdim, 2);
    Ndim = GD_ndim(g->root) = MIN(GD_ndim(g->root), MAXDIM);
    GD_odim(g->root) = MIN(outdim, Ndim);
    neato_init_node_edge(g);
}

static int neatoModel(graph_t * g)
{
    char *p = agget(g, "model");

    if (!p || streq(p, ""))    /* if p is NULL or "" */
	return MODEL_SHORTPATH;
    if (streq(p, "circuit"))
	return MODEL_CIRCUIT;
    if (streq(p, "subset"))
	return MODEL_SUBSET;
    if (streq(p, "shortpath"))
	return MODEL_SHORTPATH;
    if (streq(p, "mds")) {
	if (agattr(g, AGEDGE, "len", 0))
	    return MODEL_MDS;
	else {
	    agwarningf(
	        "edges in graph %s have no len attribute. Hence, the mds model\n", agnameof(g));
	    agerr(AGPREV, "is inappropriate. Reverting to the shortest path model.\n");
	    return MODEL_SHORTPATH;
	}
    }
    agwarningf(
	  "Unknown value %s for attribute \"model\" in graph %s - ignored\n",
	  p, agnameof(g));
    return MODEL_SHORTPATH;
}

/* neatoMode:
 */
static int neatoMode(graph_t * g)
{
    char *str;
    int mode = MODE_MAJOR;	/* default mode */

    str = agget(g, "mode");
    if (str && !streq(str, "")) {
	if (streq(str, "KK"))
	    mode = MODE_KK;
	else if (streq(str, "major"))
	    mode = MODE_MAJOR;
	else if (streq(str, "sgd"))
		mode = MODE_SGD;
#ifdef DIGCOLA
	else if (streq(str, "hier"))
	    mode = MODE_HIER;
#ifdef IPSEPCOLA
        else if (streq(str, "ipsep"))
            mode = MODE_IPSEP;
#endif
#endif
	else
	    agwarningf(
		  "Illegal value %s for attribute \"mode\" in graph %s - ignored\n",
		  str, agnameof(g));
    }

    return mode;
}

/* checkEdge:
 *
 */
static int checkEdge(PointMap * pm, edge_t * ep, int idx)
{
    int i = ND_id(agtail(ep));
    int j = ND_id(aghead(ep));
    int tmp;

    if (i > j) {
	tmp = i;
	i = j;
	j = tmp;
    }
    return insertPM(pm, i, j, idx);
}

#ifdef DIGCOLA
/* dfsCycle:
 * dfs for breaking cycles in vtxdata
 */
static void
dfsCycle (vtx_data* graph, int i,int mode, node_t* nodes[])
{
    node_t *np, *hp;
    int j;
    /* if mode is IPSEP make it an in-edge
     * at both ends, so that an edge constraint won't be generated!
     */
    double x = mode==MODE_IPSEP?-1.0:1.0;

    np = nodes[i];
    ND_mark(np) = true;
    ND_onstack(np) = true;
    for (size_t e = 1; e < graph[i].nedges; e++) {
	if (graph[i].edists[e] == 1.0) continue;  /* in edge */
	j = graph[i].edges[e];
	hp = nodes[j];
	if (ND_onstack(hp)) {  /* back edge: reverse it */
            graph[i].edists[e] = x;
            size_t f;
            for (f = 1; f < graph[j].nedges && graph[j].edges[f] != i; f++) ;
            assert (f < graph[j].nedges);
            graph[j].edists[f] = -1.0;
        }
	else if (!ND_mark(hp)) dfsCycle(graph, j, mode, nodes);

    }
    ND_onstack(np) = false;
}

/* acyclic:
 * Do a dfs of the vtx_data, looking for cycles, reversing edges.
 */
static void
acyclic (vtx_data* graph, int nv, int mode, node_t* nodes[])
{
    int i;
    node_t* np;

    for (i = 0; i < nv; i++) {
	np = nodes[i];
	ND_mark(np) = false;
	ND_onstack(np) = false;
    }
    for (i = 0; i < nv; i++) {
	if (ND_mark(nodes[i])) continue;
	dfsCycle (graph, i, mode, nodes);
    }

}
#endif

/* makeGraphData:
 * Create sparse graph representation via arrays.
 * Each node is represented by a vtx_data.
 * The index of each neighbor is stored in the edges array;
 * the corresponding edge lengths and weights go on ewgts and eweights.
 * We do not allocate the latter 2 if the graph does not use them.
 * By convention, graph[i].edges[0] == i.
 * The values graph[i].ewgts[0] and graph[i].eweights[0] are left undefined.
 *
 * In constructing graph from g, we neglect loops. We track multiedges (ignoring
 * direction). Edge weights are additive; the final edge length is the max.
 *
 * If direction is used, we set the edists field, -1 for tail, +1 for head.
 * graph[i].edists[0] is left undefined. If multiedges exist, the direction
 * of the first one encountered is used. Finally, a pass is made to guarantee
 * the graph is acyclic.
 *
 */
static vtx_data *makeGraphData(graph_t * g, int nv, int *nedges, int mode, int model, node_t*** nodedata)
{
    int ne = agnedges(g);	/* upper bound */
    float *ewgts = NULL;
    node_t *np;
    edge_t *ep;
    float *eweights = NULL;
#ifdef DIGCOLA
    float *edists = NULL;
#endif
    PointMap *ps = newPM();
    int i, idx;

    /* lengths and weights unused in reweight model */
    bool haveLen = false;
    bool haveWt = false;
    if (model != MODEL_SUBSET) {
	haveLen = agattr(g, AGEDGE, "len", 0) != NULL;
	haveWt = E_weight != 0;
    }
    bool haveDir = mode == MODE_HIER || mode == MODE_IPSEP;

    vtx_data *graph = gv_calloc(nv, sizeof(vtx_data));
    node_t** nodes = gv_calloc(nv, sizeof(node_t*));
    const size_t edges_size = (size_t)(2 * ne + nv);
    int *edges = gv_calloc(edges_size, sizeof(int)); // reserve space for self loops
    if (haveLen || haveDir)
	ewgts = gv_calloc(edges_size, sizeof(float));
    if (haveWt)
	eweights = gv_calloc(edges_size, sizeof(float));
#ifdef DIGCOLA
    if (haveDir)
	edists = gv_calloc(edges_size, sizeof(float));
#endif

    i = 0;
    ne = 0;
    for (np = agfstnode(g); np; np = agnxtnode(g, np)) {
	int j = 1;		/* index of neighbors */
	clearPM(ps);
	assert(ND_id(np) == i);
	nodes[i] = np;
	graph[i].edges = edges++;	/* reserve space for the self loop */
	if (haveLen || haveDir)
	    graph[i].ewgts = ewgts++;
	else
	    graph[i].ewgts = NULL;
	if (haveWt)
	    graph[i].eweights = eweights++;
	else
	    graph[i].eweights = NULL;
#ifdef DIGCOLA
	if (haveDir) {
	    graph[i].edists = edists++;
	}
	else
	    graph[i].edists = NULL;
#endif
	size_t i_nedges = 1; // one for the self

	for (ep = agfstedge(g, np); ep; ep = agnxtedge(g, ep, np)) {
	    if (aghead(ep) == agtail(ep))
		continue;	/* ignore loops */
	    idx = checkEdge(ps, ep, j);
	    if (idx != j) {	/* seen before */
		if (haveWt)
		    graph[i].eweights[idx] += ED_factor(ep);
		if (haveLen) {
		    graph[i].ewgts[idx] = fmax(graph[i].ewgts[idx], ED_dist(ep));
		}
	    } else {
		node_t *vp = agtail(ep) == np ? aghead(ep) : agtail(ep);
		ne++;
		j++;

		*edges++ = ND_id(vp);
		if (haveWt)
		    *eweights++ = ED_factor(ep);
		if (haveLen)
		    *ewgts++ = ED_dist(ep);
		else if (haveDir)
		    *ewgts++ = 1.0;
#ifdef DIGCOLA
                if (haveDir) {
                    char *s = agget(ep,"dir");
                    if(s && startswith(s, "none")) {
                        *edists++ = 0;
                    } else {
                        *edists++ = np == aghead(ep) ? 1.0 : -1.0;
                    }
                }
#endif
		i_nedges++;
	    }
	}

	graph[i].nedges = i_nedges;
	graph[i].edges[0] = i;
	i++;
    }
#ifdef DIGCOLA
    if (haveDir) {
    /* Make graph acyclic */
	acyclic (graph, nv, mode, nodes);
    }
#endif

    ne /= 2;			/* every edge is counted twice */

    /* If necessary, release extra memory. */
    if (ne != agnedges(g)) {
	edges = gv_recalloc(graph[0].edges, edges_size, 2 * ne + nv, sizeof(int));
	if (haveLen)
	    ewgts = gv_recalloc(graph[0].ewgts, edges_size, 2 * ne + nv, sizeof(float));
	if (haveWt)
	    eweights = gv_recalloc(graph[0].eweights, edges_size, 2 * ne + nv, sizeof(float));

	for (i = 0; i < nv; i++) {
	    const size_t sz = graph[i].nedges;
	    graph[i].edges = edges;
	    edges += sz;
	    if (haveLen) {
		graph[i].ewgts = ewgts;
		ewgts += sz;
	    }
	    if (haveWt) {
		graph[i].eweights = eweights;
		eweights += sz;
	    }
	}
    }

    *nedges = ne;
    if (nodedata)
        *nodedata = nodes;
    else
        free (nodes);
    freePM(ps);
    return graph;
}

static void initRegular(graph_t * G, int nG)
{
    double a, da;
    node_t *np;

    a = 0.0;
    da = 2 * M_PI / nG;
    for (np = agfstnode(G); np; np = agnxtnode(G, np)) {
	ND_pos(np)[0] = nG * Spring_coeff * cos(a);
	ND_pos(np)[1] = nG * Spring_coeff * sin(a);
	ND_pinned(np) = P_SET;
	a = a + da;
	if (Ndim > 2)
	    jitter3d(np, nG);
    }
}

#define SLEN(s) (sizeof(s)-1)
#define SMART   "self"
#define REGULAR "regular"
#define RANDOM  "random"

/* setSeed:
 * Analyze "start" attribute. If unset, return dflt.
 * If it begins with self, regular, or random, return set init to same,
 * else set init to dflt.
 * If init is random, look for value integer suffix to use a seed; if not
 * found, use time to set seed and store seed in graph.
 * Return seed in seedp.
 * Return init.
 */
int
setSeed (graph_t * G, int dflt, long* seedp)
{
    char *p = agget(G, "start");
    int init = dflt;

    if (!p || *p == '\0') return dflt;
    if (gv_isalpha(*p)) {
	if (startswith(p, SMART)) {
	    init = INIT_SELF;
	    p += SLEN(SMART);
	} else if (startswith(p, REGULAR)) {
	    init = INIT_REGULAR;
	    p += SLEN(REGULAR);
	} else if (startswith(p, RANDOM)) {
	    init = INIT_RANDOM;
	    p += SLEN(RANDOM);
	}
	else init = dflt;
    }
    else if (gv_isdigit(*p)) {
	init = INIT_RANDOM;
    }

    if (init == INIT_RANDOM) {
	long seed;
	/* Check for seed value */
	if (!gv_isdigit(*p) || sscanf(p, "%ld", &seed) < 1) {
#if defined(_WIN32)
	    seed = (unsigned) time(NULL);
#else
	    seed = (unsigned) getpid() ^ (unsigned) time(NULL);
#endif
	    agxbuf buf = {0};
	    agxbprint(&buf, "%ld", seed);
	    agset(G, "start", agxbuse(&buf));
	    agxbfree(&buf);
	}
	*seedp = seed;
    }
    return init;
}

/* checkExp:
 * Allow various weights for the scale factor in used to calculate stress.
 * At present, only 1 or 2 are allowed, with 2 the default.
 */
#define exp_name "stresswt"

static int checkExp (graph_t * G)
{
    int exp = late_int(G, agfindgraphattr(G, exp_name), 2, 0);
    if (exp == 0 || exp > 2) {
	agwarningf("%s attribute value must be 1 or 2 - ignoring\n", exp_name);
	exp = 2;
    }
    return exp;
}

/* checkStart:
 * Analyzes start attribute, setting seed.
 * If set,
 *   If start is regular, places nodes and returns INIT_REGULAR.
 *   If start is self, returns INIT_SELF.
 *   If start is random, returns INIT_RANDOM
 *   Set RNG seed
 * else return default
 *
 */
int checkStart(graph_t * G, int nG, int dflt)
{
    long seed;
    int init;

    seed = 1;
    init = setSeed (G, dflt, &seed);
    if (N_pos && init != INIT_RANDOM) {
	agwarningf("node positions are ignored unless start=random\n");
    }
    if (init == INIT_REGULAR) initRegular(G, nG);
    srand48(seed);
    return init;
}

#ifdef DEBUG_COLA
void dumpData(graph_t * g, vtx_data * gp, int nv, int ne)
{
    node_t *v;
    int i;

    fprintf(stderr, "#nodes %d #edges %d\n", nv, ne);
    for (v = agfstnode(g); v; v = agnxtnode(g, v)) {
	fprintf(stderr, "\"%s\" %d\n", agnameof(v), ND_id(v));
    }
    for (i = 0; i < nv; i++) {
	const size_t n = gp[i].nedges;
	fprintf(stderr, "[%d] %" PRISIZE_T "\n", i, n);
	for (size_t j = 0; j < n; j++) {
	    fprintf(stderr, "  %3d", gp[i].edges[j]);
	}
	fputs("\n", stderr);
	if (gp[i].ewgts) {
	    fputs("  ewgts", stderr);
	    for (size_t j = 0; j < n; j++) {
		fprintf(stderr, "  %3f", gp[i].ewgts[j]);
	    }
	    fputs("\n", stderr);
	}
	if (gp[i].eweights) {
	    fputs("  eweights", stderr);
	    for (size_t j = 0; j < n; j++) {
		fprintf(stderr, "  %3f", gp[i].eweights[j]);
	    }
	    fputs("\n", stderr);
	}
	if (gp[i].edists) {
	    fputs("  edists", stderr);
	    for (size_t j = 0; j < n; j++) {
		fprintf(stderr, "  %3f", gp[i].edists[j]);
	    }
	    fputs("\n", stderr);
	}
	fputs("\n", stderr);

    }
}
void dumpClusterData (cluster_data* dp)
{
  int i, j, sz;

  fprintf (stderr, "nvars %d nclusters %d ntoplevel %d\n", dp->nvars, dp->nclusters, dp->ntoplevel);
  fprintf (stderr, "Clusters:\n");
  for (i = 0; i < dp->nclusters; i++) {
    sz = dp->clustersizes[i];
    fprintf (stderr, "  [%d] %d vars\n", i, sz);
    for (j = 0; j < sz; j++)
      fprintf (stderr, "  %d", dp->clusters[i][j]);
    fprintf (stderr, "\n");
  }


  fprintf (stderr, "Toplevel:\n");
  for (i = 0; i < dp->ntoplevel; i++)
    fprintf (stderr, "  %d\n", dp->toplevel[i]);

  fprintf (stderr, "Boxes:\n");
  for (i = 0; i < dp->nclusters; i++) {
    boxf bb = dp->bb[i];
    fprintf (stderr, "  (%f,%f) (%f,%f)\n", bb.LL.x, bb.LL.y, bb.UR.x, bb.UR.y);
  }
}
void dumpOpts (ipsep_options* opp, int nv)
{
  int i;

  fprintf (stderr, "diredges %d edge_gap %f noverlap %d gap (%f,%f)\n", opp->diredges, opp->edge_gap, opp->noverlap, opp->gap.x, opp->gap.y);
  for (i = 0; i < nv; i++)
    fprintf (stderr, "  (%f,%f)\n", opp->nsize[i].x, opp->nsize[i].y);
  if (opp->clusters)
    dumpClusterData (opp->clusters);
}
#endif

/* majorization:
 * Solve stress using majorization.
 * Old neato attributes to incorporate:
 *  weight
 * mode will be MODE_MAJOR, MODE_HIER or MODE_IPSEP
 */
static void
majorization(graph_t *mg, graph_t * g, int nv, int mode, int model, int dim, adjust_data* am)
{
    int ne;
    int rv = 0;
    node_t *v;
    vtx_data *gp;
    node_t** nodes;
#ifdef DIGCOLA
#ifdef IPSEPCOLA
    expand_t margin;
#endif
#endif
    int init = checkStart(g, nv, mode == MODE_HIER ? INIT_SELF : INIT_RANDOM);
    int opts = checkExp (g);

    if (init == INIT_SELF)
	opts |= opt_smart_init;

    double **coords = gv_calloc(dim, sizeof(double *));
    coords[0] = gv_calloc(nv * dim, sizeof(double));
    for (int i = 1; i < Ndim; i++) {
	coords[i] = coords[0] + i * nv;
    }
    if (Verbose) {
	fprintf(stderr, "model %d smart_init %d stresswt %d iterations %d tol %f\n",
		model, init == INIT_SELF, opts & opt_exp_flag, MaxIter, Epsilon);
	fprintf(stderr, "convert graph: ");
	start_timer();
        fprintf(stderr, "majorization\n");
    }
    gp = makeGraphData(g, nv, &ne, mode, model, &nodes);

    if (Verbose) {
	fprintf(stderr, "%d nodes %.2f sec\n", nv, elapsed_sec());
    }

#ifdef DIGCOLA
    if (mode != MODE_MAJOR) {
        double lgap = late_double(g, agfindgraphattr(g, "levelsgap"), 0.0, -DBL_MAX);
        if (mode == MODE_HIER) {
            rv = stress_majorization_with_hierarchy(gp, nv, coords, nodes, Ndim,
                       opts, model, MaxIter, lgap);
        }
#ifdef IPSEPCOLA
	else {
            char* str;
            ipsep_options opt;
	    cluster_data cs = cluster_map(mg,g);
            pointf *nsize = gv_calloc(nv, sizeof(pointf));
            opt.edge_gap = lgap;
            opt.nsize = nsize;
            opt.clusters = cs;
            str = agget(g, "diredgeconstraints");
            if (mapbool(str)) {
                opt.diredges = 1;
                if(Verbose)
                    fprintf(stderr,"Generating Edge Constraints...\n");
	    } else if (str && !strncasecmp(str,"hier",4)) {
                opt.diredges = 2;
                if(Verbose)
                    fprintf(stderr,"Generating DiG-CoLa Edge Constraints...\n");
            }
	    else opt.diredges = 0;
	    if (am->mode == AM_IPSEP) {
                opt.noverlap = 1;
                if(Verbose)
                    fprintf(stderr,"Generating Non-overlap Constraints...\n");
            } else if (am->mode == AM_VPSC) {
                opt.noverlap = 2;
                if(Verbose)
                    fprintf(stderr,"Removing overlaps as postprocess...\n");
            }
            else opt.noverlap = 0;
	    margin = sepFactor (g);
 	    /* Multiply by 2 since opt.gap is the gap size, not the margin */
	    if (margin.doAdd) {
		opt.gap.x = 2.0*PS2INCH(margin.x);
		opt.gap.y = 2.0*PS2INCH(margin.y);
	    }
            else opt.gap.x = opt.gap.y = 2.0*PS2INCH(DFLT_MARGIN);
	    if(Verbose)
		fprintf(stderr,"gap=%f,%f\n",opt.gap.x,opt.gap.y);
            {
                size_t i = 0;
                for (v = agfstnode(g); v; v = agnxtnode(g, v),i++) {
                    nsize[i].x = ND_width(v);
                    nsize[i].y = ND_height(v);
                }
            }

#ifdef DEBUG_COLA
	    fprintf (stderr, "nv %d ne %d Ndim %d model %d MaxIter %d\n", nv, ne, Ndim, model, MaxIter);
	    fprintf (stderr, "Nodes:\n");
	    for (int i = 0; i < nv; i++) {
		fprintf (stderr, "  %s (%f,%f)\n", nodes[i]->name, coords[0][i],  coords[1][i]);
	    }
	    fprintf (stderr, "\n");
	    dumpData(g, gp, nv, ne);
	    fprintf (stderr, "\n");
	    dumpOpts (&opt, nv);
#endif
            rv = stress_majorization_cola(gp, nv, coords, nodes, Ndim, model, MaxIter, &opt);
	    freeClusterData(cs);
	    free (nsize);
        }
#endif
    }
    else
#endif
	rv = stress_majorization_kD_mkernel(gp, nv, coords, nodes, Ndim, opts, model, MaxIter);

    if (rv < 0) {
	agerr(AGPREV, "layout aborted\n");
    }
    else for (v = agfstnode(g); v; v = agnxtnode(g, v)) { /* store positions back in nodes */
	int idx = ND_id(v);
	for (int i = 0; i < Ndim; i++) {
	    ND_pos(v)[i] = coords[i][idx];
	}
    }
    freeGraphData(gp);
    free(coords[0]);
    free(coords);
    free(nodes);
}

static void subset_model(Agraph_t * G, int nG)
{
    int i, j, ne;
    vtx_data *gp;

    gp = makeGraphData(G, nG, &ne, MODE_KK, MODEL_SUBSET, NULL);
    DistType **Dij = compute_apsp_artificial_weights(gp, nG);
    for (i = 0; i < nG; i++) {
	for (j = 0; j < nG; j++) {
	    GD_dist(G)[i][j] = Dij[i][j];
	}
    }
    free(Dij[0]);
    free(Dij);
    freeGraphData(gp);
}

/* mds_model:
 * Assume the matrix already contains shortest path values.
 * Use the actual lengths provided the input for edges.
 */
static void mds_model(graph_t * g)
{
    long i, j;
    node_t *v;
    edge_t *e;

    for (v = agfstnode(g); v; v = agnxtnode(g, v)) {
	for (e = agfstout(g, v); e; e = agnxtout(g, e)) {
	    i = AGSEQ(agtail(e));
	    j = AGSEQ(aghead(e));
	    if (i == j)
		continue;
	    GD_dist(g)[i][j] = GD_dist(g)[j][i] = ED_dist(e);
	}
    }
}

/* kkNeato:
 * Solve using gradient descent a la Kamada-Kawai.
 */
static void kkNeato(Agraph_t * g, int nG, int model)
{
    if (model == MODEL_SUBSET) {
	subset_model(g, nG);
    } else if (model == MODEL_CIRCUIT) {
	if (!circuit_model(g, nG)) {
	    agwarningf(
		  "graph %s is disconnected. Hence, the circuit model\n",
		  agnameof(g));
	    agerr(AGPREV,
		  "is undefined. Reverting to the shortest path model.\n");
	    agerr(AGPREV,
		  "Alternatively, consider running neato using -Gpack=true or decomposing\n");
	    agerr(AGPREV, "the graph into connected components.\n");
	    shortest_path(g, nG);
	}
    } else if (model == MODEL_MDS) {
	shortest_path(g, nG);
	mds_model(g);
    } else
	shortest_path(g, nG);
    initial_positions(g, nG);
    diffeq_model(g, nG);
    if (Verbose) {
	fprintf(stderr, "Solving model %d iterations %d tol %f\n",
		model, MaxIter, Epsilon);
	start_timer();
    }
    solve_model(g, nG);
}

/* neatoLayout:
 * Use stress optimization to layout a single component
 */
static void
neatoLayout(Agraph_t * mg, Agraph_t * g, int layoutMode, int layoutModel,
  adjust_data* am)
{
    int nG;
    char *str;

    if ((str = agget(g, "maxiter")))
	MaxIter = atoi(str);
    else if (layoutMode == MODE_MAJOR)
	MaxIter = DFLT_ITERATIONS;
    else if (layoutMode == MODE_SGD)
	MaxIter = 30;
    else
	MaxIter = 100 * agnnodes(g);

    nG = scan_graph_mode(g, layoutMode);
    if (nG < 2 || MaxIter < 0)
	return;
    if (layoutMode == MODE_KK)
	kkNeato(g, nG, layoutModel);
    else if (layoutMode == MODE_SGD)
	sgd(g, layoutModel);
    else
	majorization(mg, g, nG, layoutMode, layoutModel, Ndim, am);
}

/* addZ;
 * If dimension == 3 and z attribute is declared,
 * attach z value to nodes if not defined.
 */
static void addZ (Agraph_t* g)
{
    node_t* n;
    char    buf[BUFSIZ];

    if (Ndim >= 3 && N_z) {
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    snprintf(buf, sizeof(buf), "%lf", POINTS_PER_INCH * ND_pos(n)[2]);
	    agxset(n, N_z, buf);
	}
    }
}

#ifdef IPSEPCOLA
static void
addCluster (graph_t* g)
{
    graph_t *subg;
    for (subg = agfstsubg(agroot(g)); subg; subg = agnxtsubg(subg)) {
	if (is_a_cluster(subg)) {
	    agbindrec(subg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
	    add_cluster(g, subg);
	    compute_bb(subg);
	}
   }
}
#endif

/* doEdges:
 * Simple wrapper to compute graph's bb, then route edges after
 * a possible aspect ratio adjustment.
 */
static void doEdges(Agraph_t* g)
{
    compute_bb(g);
    spline_edges0(g, true);
}

/* neato_layout:
 */
void neato_layout(Agraph_t * g)
{
    int layoutMode;
    int model;
    pack_mode mode;
    pack_info pinfo;
    adjust_data am;
    double save_scale = PSinputscale;

    if (Nop) {
	int ret;
	PSinputscale = POINTS_PER_INCH;
	neato_init_graph(g);
	addZ (g);
	ret = init_nop(g, 1);
	if (ret < 0) {
	    agerr(AGPREV, "as required by the -n flag\n");
	    return;
	}
	else gv_postprocess(g, 0);
    } else {
	bool noTranslate = mapbool(agget(g, "notranslate"));
	PSinputscale = get_inputscale (g);
	neato_init_graph(g);
	layoutMode = neatoMode(g);
	graphAdjustMode (g, &am, 0);
	model = neatoModel(g);
	mode = getPackModeInfo (g, l_undef, &pinfo);
	Pack = getPack(g, -1, CL_OFFSET);
	/* pack if just packmode defined. */
	if (mode == l_undef) {
	    /* If the user has not indicated packing but we are
	     * using the new neato, turn packing on.
	     */
	    if (Pack < 0 && layoutMode)
		Pack = CL_OFFSET;
	    pinfo.mode = l_node;
	} else if (Pack < 0)
	    Pack = CL_OFFSET;
	if (Pack >= 0) {
	    graph_t *gc;
	    graph_t **cc;
	    size_t n_cc;
	    bool pin;

	    cc = pccomps(g, &n_cc, cc_pfx, &pin);

	    if (n_cc > 1) {
		bool *bp;
		for (size_t i = 0; i < n_cc; i++) {
		    gc = cc[i];
		    (void)graphviz_node_induce(gc, NULL);
		    neatoLayout(g, gc, layoutMode, model, &am);
		    removeOverlapWith(gc, &am);
		    setEdgeType (gc, EDGETYPE_LINE);
		    if (noTranslate) doEdges(gc);
		    else spline_edges(gc);
		}
		if (pin) {
		    bp = gv_calloc(n_cc, sizeof(bool));
		    bp[0] = true;
		} else
		    bp = NULL;
		pinfo.margin = (unsigned)Pack;
		pinfo.fixed = bp;
		pinfo.doSplines = true;
		packGraphs(n_cc, cc, g, &pinfo);
		free(bp);
	    }
	    else {
		neatoLayout(g, g, layoutMode, model, &am);
		removeOverlapWith(g, &am);
		if (noTranslate) doEdges(g);
		else spline_edges(g);
	    }
	    compute_bb(g);
	    addZ (g);

	    /* cleanup and remove component subgraphs */
	    for (size_t i = 0; i < n_cc; i++) {
		gc = cc[i];
		free_scan_graph(gc);
		agdelrec (gc, "Agraphinfo_t");
		agdelete(g, gc);
	    }
	    free (cc);
#ifdef IPSEPCOLA
	    addCluster (g);
#endif
	} else {
	    neatoLayout(g, g, layoutMode, model, &am);
	    removeOverlapWith(g, &am);
	    addZ (g);
	    if (noTranslate) doEdges(g);
	    else spline_edges(g);
	}
	gv_postprocess(g, !noTranslate);
    }
    PSinputscale = save_scale;
}

/**
 * @dir lib/neatogen
 * @brief "spring model" layout engine, API neatogen/neatoprocs.h
 * @ingroup engines
 *
 * [Neato layout user manual](https://graphviz.org/docs/layouts/neato/)
 *
 * Other @ref engines
 */
