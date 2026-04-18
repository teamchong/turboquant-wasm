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
 * position(g): set ND_coord(n) (x and y) for all nodes n of g, using GD_rank(g).
 * (the graph may be modified by merging certain edges with a common endpoint.)
 * the coordinates are computed by constructing and ranking an auxiliary graph.
 * then leaf nodes are inserted in the fast graph.  cluster boundary nodes are
 * created and correctly separated.
 */

#include "config.h"

#include <common/geomprocs.h>
#include <dotgen/dot.h>
#include <dotgen/aspect.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/gv_math.h>

static int nsiter2(graph_t * g);
static void create_aux_edges(graph_t * g);
static void remove_aux_edges(graph_t * g);
static void set_xcoords(graph_t * g);
static void set_ycoords(graph_t * g);
static void set_aspect(graph_t *g);
static void expand_leaves(graph_t * g);
static void make_lrvn(graph_t * g);
static void contain_nodes(graph_t * g);
static bool idealsize(graph_t * g, double);

#if defined(DEBUG) && DEBUG > 1
static void
dumpNS (graph_t * g)
{
    node_t* n = GD_nlist(g);
    elist el;
    edge_t* e;

    while (n) {
	el = ND_out(n);
	for (size_t i = 0; i < el.size; i++) {
	    e = el.list[i];
	    fprintf (stderr, "%s(%x) -> ", agnameof(agtail(e)),agtail(e));
	    fprintf (stderr, "%s(%x) : %d\n", agnameof(aghead(e)), aghead(e),
		ED_minlen(e));
	}
	n = ND_next(n); 
    }
}
#endif

static double
largeMinlen (double l)
{
  agerrorf(
        "Edge length %f larger than maximum %d allowed.\nCheck for overwide "
        "node(s).\n",
        l, INT_MAX);
  return INT_MAX;
}

/* When source and/or sink nodes are defined, it is possible that
 * after the auxiliary edges are added, the graph may still have 2 or
 * 3 components. To fix this, we put trivial constraints connecting the
 * first items of each rank.
 */
static void
connectGraph (graph_t* g)
{
    int i, j, r;
    bool found;
    node_t* tp;
    node_t* hp;
    node_t* sn;
    edge_t* e;
    rank_t* rp;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	rp = GD_rank(g)+r;
	found = false;
        tp = NULL;
	for (i = 0; i < rp->n; i++) {
	    tp = rp->v[i];
	    if (ND_save_out(tp).list) {
        	for (j = 0; (e = ND_save_out(tp).list[j]); j++) {
		    if (ND_rank(aghead(e)) > r || ND_rank(agtail(e)) > r) {
			found = true;
			break;
		    }
        	}
		if (found) break;
	    }
	    if (ND_save_in(tp).list) {
        	for (j = 0; (e = ND_save_in(tp).list[j]); j++) {
		    if (ND_rank(agtail(e)) > r || ND_rank(aghead(e)) > r) {
			found = true;
			break;
		    }
        	}
		if (found) break;
	    }
	}
	if (found || !tp) continue;
	tp = rp->v[0];
	if (r < GD_maxrank(g)) hp = (rp+1)->v[0];
	else hp = (rp-1)->v[0];
	assert (hp);
	sn = virtual_node(g);
	ND_node_type(sn) = SLACKNODE;
	make_aux_edge(sn, tp, 0, 0);
	make_aux_edge(sn, hp, 0, 0);
	ND_rank(sn) = MIN(ND_rank(tp), ND_rank(hp));
    }
}

void dot_position(graph_t *g) {
    if (GD_nlist(g) == NULL)
	return;			/* ignore empty graph */
    mark_lowclusters(g);	/* we could remove from splines.c now */
    set_ycoords(g);
    if (Concentrate)
	dot_concentrate(g);
    expand_leaves(g);
    if (flat_edges(g))
	set_ycoords(g);
    create_aux_edges(g);
    if (rank(g, 2, nsiter2(g))) { /* LR balance == 2 */
	connectGraph (g);
	const int rank_result = rank(g, 2, nsiter2(g));
	assert(rank_result == 0);
	(void)rank_result;
    }
    set_xcoords(g);
    set_aspect(g);
    remove_aux_edges(g);	/* must come after set_aspect since we now
				 * use GD_ln and GD_rn for bbox width.
				 */
}

static int nsiter2(graph_t * g)
{
    int maxiter = INT_MAX;
    char *s;

    if ((s = agget(g, "nslimit")))
	maxiter = scale_clamp(agnnodes(g), atof(s));
    return maxiter;
}

static bool go(node_t *u, node_t *v) {
    int i;
    edge_t *e;

    if (u == v)
	return true;
    for (i = 0; (e = ND_out(u).list[i]); i++) {
	if (go(aghead(e), v))
	    return true;
    }
    return false;
}

static bool canreach(node_t *u, node_t *v) {
    return go(u, v);
}

edge_t *make_aux_edge(node_t * u, node_t * v, double len, int wt)
{
    Agedgepair_t* e2 = gv_alloc(sizeof(Agedgepair_t));
    AGTYPE(&e2->in) = AGINEDGE;
    AGTYPE(&e2->out) = AGOUTEDGE;
    e2->out.base.data = gv_alloc(sizeof(Agedgeinfo_t));
    edge_t *const e = &e2->out;

    agtail(e) = u;
    aghead(e) = v;
    if (len > INT_MAX)
	len = largeMinlen (len);
    ED_minlen(e) = ROUND(len);
    ED_weight(e) = wt;
    fast_edge(e);
    return e;
}

static void allocate_aux_edges(graph_t * g)
{
    int i, j, n_in;
    node_t *n;

    /* allocate space for aux edge lists */
    for (n = GD_nlist(g); n; n = ND_next(n)) {
	ND_save_in(n) = ND_in(n);
	ND_save_out(n) = ND_out(n);
	for (i = 0; ND_out(n).list[i]; i++);
	for (j = 0; ND_in(n).list[j]; j++);
	n_in = i + j;
	alloc_elist(n_in + 3, ND_in(n));
	alloc_elist(3, ND_out(n));
    }
}

static void 
make_LR_constraints(graph_t * g)
{
    int i, j;
    int m0;
    double width;
    int sep[2];
    int nodesep;      /* separation between nodes on same rank */
    edge_t *e, *e0, *e1;
    node_t *u, *v, *t0, *h0;
    rank_t *rank = GD_rank(g);

    /* Use smaller separation on odd ranks if g has edge labels */
    if (GD_has_labels(g->root) & EDGE_LABEL) {
	sep[0] = GD_nodesep(g);
	sep[1] = 5;
    }
    else {
	sep[1] = sep[0] = GD_nodesep(g);
    }
    /* make edges to constrain left-to-right ordering */
    for (i = GD_minrank(g); i <= GD_maxrank(g); i++) {
	double last = ND_rank(rank[i].v[0]) = 0;
	nodesep = sep[i & 1];
	for (j = 0; j < rank[i].n; j++) {
	    u = rank[i].v[j];
	    ND_mval(u) = ND_rw(u);	/* keep it somewhere safe */
	    if (ND_other(u).size > 0) {	/* compute self size */
		/* FIX: dot assumes all self-edges go to the right. This
                 * is no longer true, though makeSelfEdge still attempts to
                 * put as many as reasonable on the right. The dot code
                 * should be modified to allow a box reflecting the placement
                 * of all self-edges, and use that to reposition the nodes.
                 * Note that this would not only affect left and right
                 * positioning but may also affect interrank spacing.
                 */
		double sw = 0; // self width
		for (size_t k = 0; (e = ND_other(u).list[k]); k++) {
		    if (agtail(e) == aghead(e)) {
			sw += selfRightSpace (e);
		    }
		}
		ND_rw(u) += sw;	/* increment to include self edges */
	    }
	    v = rank[i].v[j + 1];
	    if (v) {
		width = ND_rw(u) + ND_lw(v) + nodesep;
		e0 = make_aux_edge(u, v, width, 0);
		last = (ND_rank(v) = last + width);
	    }

	    /* constraints from labels of flat edges on previous rank */
	    if ((e = ND_alg(u))) {
		e0 = ND_save_out(u).list[0];
		e1 = ND_save_out(u).list[1];
		if (ND_order(aghead(e0)) > ND_order(aghead(e1))) {
		    SWAP(&e0, &e1);
		}
		m0 = ED_minlen(e) * GD_nodesep(g) / 2;
		double m1 = m0 + ND_rw(aghead(e0)) + ND_lw(agtail(e0));
		/* these guards are needed because the flat edges
		 * work very poorly with cluster layout */
		if (!canreach(agtail(e0), aghead(e0)))
		    make_aux_edge(aghead(e0), agtail(e0), m1,
			ED_weight(e));
		m1 = m0 + ND_rw(agtail(e1)) + ND_lw(aghead(e1));
		if (!canreach(aghead(e1), agtail(e1)))
		    make_aux_edge(agtail(e1), aghead(e1), m1,
			ED_weight(e));
	    }

	    /* position flat edge endpoints */
	    for (size_t k = 0; k < ND_flat_out(u).size; k++) {
		e = ND_flat_out(u).list[k];
		if (ND_order(agtail(e)) < ND_order(aghead(e))) {
		    t0 = agtail(e);
		    h0 = aghead(e);
		} else {
		    t0 = aghead(e);
		    h0 = agtail(e);
		}

		width = ND_rw(t0) + ND_lw(h0);
		m0 = ED_minlen(e) * GD_nodesep(g) + width;

		if ((e0 = find_fast_edge(t0, h0))) {
		    /* flat edge between adjacent neighbors 
                     * ED_dist contains the largest label width.
                     */
		    m0 = MAX(m0, width + GD_nodesep(g) + ROUND(ED_dist(e)));
		    ED_minlen(e0) = MAX(ED_minlen(e0), m0);
		    ED_weight(e0) = MAX(ED_weight(e0), ED_weight(e));
		}
		else if (!ED_label(e)) {
		    /* unlabeled flat edge between non-neighbors 
		     * ED_minlen(e) is max of ED_minlen of all equivalent 
                     * edges.
                     */
		    make_aux_edge(t0, h0, m0, ED_weight(e));
		}
		/* labeled flat edges between non-neighbors have already
                 * been constrained by the label above. 
                 */ 
	    }
	}
    }
}

/// make virtual edge pairs corresponding to input edges
static void make_edge_pairs(graph_t * g)
{
    int i, m0, m1;
    node_t *n, *sn;
    edge_t *e;

    for (n = GD_nlist(g); n; n = ND_next(n)) {
	if (ND_save_out(n).list)
	    for (i = 0; (e = ND_save_out(n).list[i]); i++) {
		sn = virtual_node(g);
		ND_node_type(sn) = SLACKNODE;
		m0 = (ED_head_port(e).p.x - ED_tail_port(e).p.x);
		if (m0 > 0)
		    m1 = 0;
		else {
		    m1 = -m0;
		    m0 = 0;
		}
		make_aux_edge(sn, agtail(e), m0 + 1, ED_weight(e));
		make_aux_edge(sn, aghead(e), m1 + 1, ED_weight(e));
		ND_rank(sn) =
		    MIN(ND_rank(agtail(e)) - m0 - 1,
			ND_rank(aghead(e)) - m1 - 1);
	    }
    }
}

static void contain_clustnodes(graph_t * g)
{
    int c;
    edge_t	*e;

    if (g != dot_root(g)) {
	contain_nodes(g);
	if ((e = find_fast_edge(GD_ln(g),GD_rn(g))))	/* maybe from lrvn()?*/
	    ED_weight(e) += 128;
	else
	    make_aux_edge(GD_ln(g), GD_rn(g), 1, 128);	/* clust compaction edge */
    }
    for (c = 1; c <= GD_n_cluster(g); c++)
	contain_clustnodes(GD_clust(g)[c]);
}

static bool vnode_not_related_to(graph_t *g, node_t *v) {
    edge_t *e;

    if (ND_node_type(v) != VIRTUAL)
	return false;
    for (e = ND_save_out(v).list[0]; ED_to_orig(e); e = ED_to_orig(e));
    if (agcontains(g, agtail(e)))
	return false;
    if (agcontains(g, aghead(e)))
	return false;
    return true;
}

/* Guarantee nodes outside the cluster g are placed outside of it.
 * This is done by adding constraints to make sure such nodes have
 * a gap of margin from the left or right bounding box node ln or rn.
 * 
 * We could probably reduce some of these constraints by checking if
 * the node is in a cluster, since elsewhere we make constrain a
 * separate between clusters. Also, we should be able to skip the
 * first loop if g is the root graph.
 */
static void keepout_othernodes(graph_t * g)
{
    int i, c, r, margin;
    node_t *u, *v;

    margin = late_int (g, G_margin, CL_OFFSET, 0);
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	if (GD_rank(g)[r].n == 0)
	    continue;
	v = GD_rank(g)[r].v[0];
	if (v == NULL)
	    continue;
	for (i = ND_order(v) - 1; i >= 0; i--) {
	    u = GD_rank(dot_root(g))[r].v[i];
	    /* can't use "is_a_vnode_of" because elists are swapped */
	    if (ND_node_type(u) == NORMAL || vnode_not_related_to(g, u)) {
		make_aux_edge(u, GD_ln(g), margin + ND_rw(u), 0);
		break;
	    }
	}
	for (i = ND_order(v) + GD_rank(g)[r].n; i < GD_rank(dot_root(g))[r].n;
	     i++) {
	    u = GD_rank(dot_root(g))[r].v[i];
	    if (ND_node_type(u) == NORMAL || vnode_not_related_to(g, u)) {
		make_aux_edge(GD_rn(g), u, margin + ND_lw(u), 0);
		break;
	    }
	}
    }

    for (c = 1; c <= GD_n_cluster(g); c++)
	keepout_othernodes(GD_clust(g)[c]);
}

/* Make sure boxes of subclusters of g are offset from the
 * box of g. This is done by a constraint between the left and
 * right bounding box nodes ln and rn of g and a subcluster.
 * The gap needs to include any left or right labels.
 */
static void contain_subclust(graph_t * g)
{
    int margin, c;
    graph_t *subg;

    margin = late_int (g, G_margin, CL_OFFSET, 0);
    make_lrvn(g);
    for (c = 1; c <= GD_n_cluster(g); c++) {
	subg = GD_clust(g)[c];
	make_lrvn(subg);
	make_aux_edge(GD_ln(g), GD_ln(subg),
		      margin + GD_border(g)[LEFT_IX].x, 0);
	make_aux_edge(GD_rn(subg), GD_rn(g),
		      margin + GD_border(g)[RIGHT_IX].x, 0);
	contain_subclust(subg);
    }
}

/* Guarantee space between subcluster of g.
 * This is done by adding a constraint between the right bbox node rn
 * of the left cluster and the left bbox node ln of the right cluster.
 * This is only done if the two clusters overlap in some rank.
 */
static void separate_subclust(graph_t * g)
{
    int i, j, margin;
    graph_t *low, *high;
    graph_t *left, *right;

    margin = late_int (g, G_margin, CL_OFFSET, 0);
    for (i = 1; i <= GD_n_cluster(g); i++)
	make_lrvn(GD_clust(g)[i]);
    for (i = 1; i <= GD_n_cluster(g); i++) {
	for (j = i + 1; j <= GD_n_cluster(g); j++) {
	    low = GD_clust(g)[i];
	    high = GD_clust(g)[j];
	    if (GD_minrank(low) > GD_minrank(high)) {
		SWAP(&low, &high);
	    }
	    if (GD_maxrank(low) < GD_minrank(high))
		continue;
	    if (ND_order(GD_rank(low)[GD_minrank(high)].v[0])
		< ND_order(GD_rank(high)[GD_minrank(high)].v[0])) {
		left = low;
		right = high;
	    } else {
		left = high;
		right = low;
	    }
	    make_aux_edge(GD_rn(left), GD_ln(right), margin, 0);
	}
	separate_subclust(GD_clust(g)[i]);
    }
}

/*	create constraints for:
 *	node containment in clusters,
 *	cluster containment in clusters,
 *	separation of sibling clusters.
 */
static void pos_clusters(graph_t * g)
{
    if (GD_n_cluster(g) > 0) {
	contain_clustnodes(g);
	keepout_othernodes(g);
	contain_subclust(g);
	separate_subclust(g);
    }
}

static void compress_graph(graph_t * g)
{
    double x;
    pointf p;

    if (GD_drawing(g)->ratio_kind != R_COMPRESS)
	return;
    p = GD_drawing(g)->size;
    if (p.x * p.y <= 1)
	return;
    contain_nodes(g);
    if (!GD_flip(g))
	x = p.x;
    else
	x = p.y;

    /* Guard against huge size attribute since max. edge length is USHRT_MAX
     * A warning might be called for. Also, one could check that the graph
     * already fits GD_drawing(g)->size and return immediately.
     */
    x = MIN(x,USHRT_MAX); 
    make_aux_edge(GD_ln(g), GD_rn(g), x, 1000);
}

static void create_aux_edges(graph_t * g)
{
    allocate_aux_edges(g);
    make_LR_constraints(g);
    make_edge_pairs(g);
    pos_clusters(g);
    compress_graph(g);
}

static void remove_aux_edges(graph_t * g)
{
    int i;
    node_t *n, *nnext, *nprev;
    edge_t *e;

    for (n = GD_nlist(g); n; n = ND_next(n)) {
	for (i = 0; (e = ND_out(n).list[i]); i++) {
	    free(e->base.data);
	    free(e);
	}
	free_list(ND_out(n));
	free_list(ND_in(n));
	ND_out(n) = ND_save_out(n);
	ND_in(n) = ND_save_in(n);
    }
    /* cannot be merged with previous loop */
    nprev = NULL;
    for (n = GD_nlist(g); n; n = nnext) {
	nnext = ND_next(n);
	if (ND_node_type(n) == SLACKNODE) {
	    if (nprev)
		ND_next(nprev) = nnext;
	    else
		GD_nlist(g) = nnext;
	    if (nnext != NULL) {
		ND_prev(nnext) = nprev;
	    }
	    free(n->base.data);
	    free(n);
	} else
	    nprev = n;
    }
}

/// Set x coords of nodes.
static void 
set_xcoords(graph_t * g)
{
    int i, j;
    node_t *v;
    rank_t *rank = GD_rank(g);

    for (i = GD_minrank(g); i <= GD_maxrank(g); i++) {
	for (j = 0; j < rank[i].n; j++) {
	    v = rank[i].v[j];
	    ND_coord(v).x = ND_rank(v);
	    ND_rank(v) = i;
	}
    }
}

/* Expand cluster height by delta, adding half to top
 * and half to bottom. If the bottom expansion exceeds the
 * ht1 of the rank, shift the ranks in the cluster up.
 * If the top expansion, including any shift from the bottom
 * expansion, exceeds to ht2 of the rank, shift the ranks above
 * the cluster up.
 *
 * FIX: There can be excess space between ranks. Not sure where this is
 * coming from but it could be cleaned up.
 */
static void adjustSimple(graph_t *g, double delta, int margin_total) {
    int r;
    double deltop;
    graph_t *root = dot_root(g);
    rank_t *rank = GD_rank(root);
    int maxr = GD_maxrank(g);
    int minr = GD_minrank(g);

    const double bottom = (delta + 1) / 2;
    const double delbottom = GD_ht1(g) + bottom - (rank[maxr].ht1 - margin_total);
    if (delbottom > 0) {
	for (r = maxr; r >= minr; r--) {
	    if (rank[r].n > 0)
		ND_coord(rank[r].v[0]).y += delbottom;
 	}
	deltop = GD_ht2(g) + (delta-bottom) + delbottom - (rank[minr].ht2 - margin_total);
    }
    else
	deltop = GD_ht2(g) + (delta-bottom) - (rank[minr].ht2 - margin_total);
    if (deltop > 0) {
	for (r = minr-1; r >= GD_minrank(root); r--) {
	    if (rank[r].n > 0)
		ND_coord(rank[r].v[0]).y += deltop;
	}
    }
    GD_ht2(g) += delta - bottom;
    GD_ht1(g) += bottom;
}

/* Recursively adjust ranks to take into account
 * wide cluster labels when rankdir=LR.
 * We divide the extra space between the top and bottom.
 * Adjust the ht1 and ht2 values in the process.
 */
static void adjustRanks(graph_t * g, int margin_total)
{
    double lht;			/* label height */
    double rht;			/* height between top and bottom ranks */
    int maxr, minr, margin;
    int c;
    double delta, ht1, ht2;

    rank_t *rank = GD_rank(dot_root(g));
    if (g == dot_root(g))
	margin = 0;
    else
	margin = late_int (g, G_margin, CL_OFFSET, 0);

    ht1 = GD_ht1(g);
    ht2 = GD_ht2(g);

    for (c = 1; c <= GD_n_cluster(g); c++) {
	graph_t *subg = GD_clust(g)[c];
	adjustRanks(subg, margin+margin_total);
	if (GD_maxrank(subg) == GD_maxrank(g))
	    ht1 = fmax(ht1, GD_ht1(subg) + margin);
	if (GD_minrank(subg) == GD_minrank(g))
	    ht2 = fmax(ht2, GD_ht2(subg) + margin);
    }

    GD_ht1(g) = ht1;
    GD_ht2(g) = ht2;

    if (g != dot_root(g) && GD_label(g)) {
	lht = MAX(GD_border(g)[LEFT_IX].y, GD_border(g)[RIGHT_IX].y);
	maxr = GD_maxrank(g);
	minr = GD_minrank(g);
	rht = ND_coord(rank[minr].v[0]).y - ND_coord(rank[maxr].v[0]).y;
	delta = lht - (rht + ht1 + ht2);
	if (delta > 0) {
	    adjustSimple(g, delta, margin_total);
	}
    }

    /* update the global ranks */
    if (g != dot_root(g)) {
	rank[GD_minrank(g)].ht2 = fmax(rank[GD_minrank(g)].ht2, GD_ht2(g));
	rank[GD_maxrank(g)].ht1 = fmax(rank[GD_maxrank(g)].ht1, GD_ht1(g));
    }
}

/* recursively compute cluster ht requirements.  assumes GD_ht1(subg) and ht2
 * are computed from primitive nodes only.  updates ht1 and ht2 to reflect
 * cluster nesting and labels.  also maintains global rank ht1 and ht2.
 * Return true if some cluster has a label.
 */
static int clust_ht(Agraph_t * g)
{
    int c;
    double ht1, ht2;
    graph_t *subg;
    rank_t *rank = GD_rank(dot_root(g));
    int margin, haveClustLabel = 0;

    if (g == dot_root(g)) 
	margin = CL_OFFSET;
    else
	margin = late_int (g, G_margin, CL_OFFSET, 0);

    ht1 = GD_ht1(g);
    ht2 = GD_ht2(g);

    /* account for sub-clusters */
    for (c = 1; c <= GD_n_cluster(g); c++) {
	subg = GD_clust(g)[c];
	haveClustLabel |= clust_ht(subg);
	if (GD_maxrank(subg) == GD_maxrank(g))
	    ht1 = MAX(ht1, GD_ht1(subg) + margin);
	if (GD_minrank(subg) == GD_minrank(g))
	    ht2 = MAX(ht2, GD_ht2(subg) + margin);
    }

    /* account for a possible cluster label in clusters */
    /* room for root graph label is handled in dotneato_postprocess */
    if (g != dot_root(g) && GD_label(g)) {
	haveClustLabel = 1;
	if (!GD_flip(agroot(g))) {
	    ht1 += GD_border(g)[BOTTOM_IX].y;
	    ht2 += GD_border(g)[TOP_IX].y;
	}
    }
    GD_ht1(g) = ht1;
    GD_ht2(g) = ht2;

    /* update the global ranks */
    if (g != dot_root(g)) {
	rank[GD_minrank(g)].ht2 = MAX(rank[GD_minrank(g)].ht2, ht2);
	rank[GD_maxrank(g)].ht1 = MAX(rank[GD_maxrank(g)].ht1, ht1);
    }

    return haveClustLabel;
}

/* set y coordinates of nodes, a rank at a time */
static void set_ycoords(graph_t * g)
{
    edge_t *e;
    rank_t *rank = GD_rank(g);
    graph_t *clust;

    /* scan ranks for tallest nodes.  */
    for (int r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	for (int i = 0; i < rank[r].n; i++) {
	    node_t *const n = rank[r].v[i];

	    /* assumes symmetry, ht1 = ht2 */
	    double ht2 = ND_ht(n) / 2;


	    /* have to look for high self-edge labels, too */
	    if (ND_other(n).list)
		for (int j = 0; (e = ND_other(n).list[j]); j++) {
		    if (agtail(e) == aghead(e)) {
			if (ED_label(e))
			    ht2 = fmax(ht2, ED_label(e)->dimen.y / 2);
		    }
		}

	    /* update global rank ht */
	    if (rank[r].pht2 < ht2)
		rank[r].pht2 = rank[r].ht2 = ht2;
	    if (rank[r].pht1 < ht2)
		rank[r].pht1 = rank[r].ht1 = ht2;

	    /* update nearest enclosing cluster rank ht */
	    if ((clust = ND_clust(n))) {
		int yoff = clust == g ? 0 : late_int (clust, G_margin, CL_OFFSET, 0);
		if (ND_rank(n) == GD_minrank(clust))
		    GD_ht2(clust) = fmax(GD_ht2(clust), ht2 + yoff);
		if (ND_rank(n) == GD_maxrank(clust))
		    GD_ht1(clust) = fmax(GD_ht1(clust), ht2 + yoff);
	    }
	}
    }

    /* scan sub-clusters */
    const int lbl = clust_ht(g);

    /* make the initial assignment of ycoords to leftmost nodes by ranks */
    double maxht = 0;
    int r = GD_maxrank(g);
    ND_coord(rank[r].v[0]).y = rank[r].ht1;
    while (--r >= GD_minrank(g)) {
	const double d0 = rank[r + 1].pht2 + rank[r].pht1 + GD_ranksep(g); // prim node sep
	const double d1 = rank[r + 1].ht2 + rank[r].ht1 + CL_OFFSET; // cluster sep
	const double delta = fmax(d0, d1);
	if (rank[r].n > 0)	/* this may reflect some problem */
		ND_coord(rank[r].v[0]).y = ND_coord(rank[r + 1].v[0]).y + delta;
#ifdef DEBUG
	else
	    fprintf(stderr, "dot set_ycoords: rank %d is empty\n",
		    rank[r].n);
#endif
	maxht = fmax(maxht, delta);
    }

    /* If there are cluster labels and the drawing is rotated, we need special processing to
     * allocate enough room. We use adjustRanks for this, and then recompute the maxht if
     * the ranks are to be equally spaced. This seems simpler and appears to work better than
     * handling equal spacing as a special case.
     */
    if (lbl && GD_flip(g)) {
	adjustRanks(g, 0);
	if (GD_exact_ranksep(g)) {  /* recompute maxht */
	    maxht = 0;
	    r = GD_maxrank(g);
	    double d0 = ND_coord(rank[r].v[0]).y;
	    while (--r >= GD_minrank(g)) {
		const double d1 = ND_coord(rank[r].v[0]).y;
		const double delta = d1 - d0;
		maxht = fmax(maxht, delta);
		d0 = d1;
	    }
	}
    }

    /* re-assign if ranks are equally spaced */
    if (GD_exact_ranksep(g)) {
	for (r = GD_maxrank(g) - 1; r >= GD_minrank(g); r--)
	    if (rank[r].n > 0)	/* this may reflect the same problem :-() */
			ND_coord(rank[r].v[0]).y = ND_coord(rank[r + 1].v[0]).y + maxht;
    }

    /* copy ycoord assignment from leftmost nodes to others */
    for (node_t *n = GD_nlist(g); n; n = ND_next(n))
	ND_coord(n).y = ND_coord(rank[ND_rank(n)].v[0]).y;
}

/* Compute bounding box of g.
 * The x limits of clusters are given by the x positions of ln and rn.
 * This information is stored in the rank field, since it was calculated
 * using network simplex.
 * For the root graph, we don't enforce all the constraints on lr and 
 * rn, so we traverse the nodes and subclusters.
 */
static void dot_compute_bb(graph_t * g, graph_t * root)
{
    int r, c;
    double x, offset;
    node_t *v;
    pointf LL, UR;

    if (g == dot_root(g)) {
	LL.x = INT_MAX;
	UR.x = -INT_MAX;
	for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	    int rnkn = GD_rank(g)[r].n;
	    if (rnkn == 0)
		continue;
	    if ((v = GD_rank(g)[r].v[0]) == NULL)
		continue;
	    for (c = 1; ND_node_type(v) != NORMAL && c < rnkn; c++)
		v = GD_rank(g)[r].v[c];
	    if (ND_node_type(v) == NORMAL) {
		x = ND_coord(v).x - ND_lw(v);
		LL.x = MIN(LL.x, x);
	    }
	    else continue;
		/* At this point, we know the rank contains a NORMAL node */
	    v = GD_rank(g)[r].v[rnkn - 1];
	    for (c = rnkn-2; ND_node_type(v) != NORMAL; c--)
		v = GD_rank(g)[r].v[c];
	    x = ND_coord(v).x + ND_rw(v);
	    UR.x = MAX(UR.x, x);
	}
	offset = CL_OFFSET;
	for (c = 1; c <= GD_n_cluster(g); c++) {
	    x = (double)(GD_bb(GD_clust(g)[c]).LL.x - offset);
	    LL.x = MIN(LL.x, x);
	    x = (double)(GD_bb(GD_clust(g)[c]).UR.x + offset);
	    UR.x = MAX(UR.x, x);
	}
    } else {
	LL.x = ND_rank(GD_ln(g));
	UR.x = ND_rank(GD_rn(g));
    }
    LL.y = ND_coord(GD_rank(root)[GD_maxrank(g)].v[0]).y - GD_ht1(g);
    UR.y = ND_coord(GD_rank(root)[GD_minrank(g)].v[0]).y + GD_ht2(g);
    GD_bb(g).LL = LL;
    GD_bb(g).UR = UR;
}

static void rec_bb(graph_t * g, graph_t * root)
{
    int c;
    for (c = 1; c <= GD_n_cluster(g); c++)
	rec_bb(GD_clust(g)[c], root);
    dot_compute_bb(g, root);
}

/* Recursively rescale all bounding boxes using scale factors
 * xf and yf. We assume all the bboxes have been computed.
 */
static void scale_bb(graph_t *g, double xf, double yf) {
    int c;

    for (c = 1; c <= GD_n_cluster(g); c++)
	scale_bb(GD_clust(g)[c], xf, yf);
    GD_bb(g).LL.x *= xf;
    GD_bb(g).LL.y *= yf;
    GD_bb(g).UR.x *= xf;
    GD_bb(g).UR.y *= yf;
}

/* Set bounding boxes and, if ratio is set, rescale graph.
 * Note that if some dimension shrinks, there may be problems
 * with labels.
 */
static void set_aspect(graph_t *g) {
    double xf = 0.0, yf = 0.0, actual, desired;
    node_t *n;
    bool filled;

    rec_bb(g, g);
    if (GD_maxrank(g) > 0 && GD_drawing(g)->ratio_kind) {
	pointf sz = sub_pointf(GD_bb(g).UR, GD_bb(g).LL); // normalize
	if (GD_flip(g)) {
	    sz = exch_xyf(sz);
	}
	bool scale_it = true;
	if (GD_drawing(g)->ratio_kind == R_AUTO)
	    filled = idealsize(g, .5);
	else
	    filled = GD_drawing(g)->ratio_kind == R_FILL;
	if (filled) {
	    /* fill is weird because both X and Y can stretch */
	    if (GD_drawing(g)->size.x <= 0)
		scale_it = false;
	    else {
		xf = GD_drawing(g)->size.x / sz.x;
		yf = GD_drawing(g)->size.y / sz.y;
		if (xf < 1.0 || yf < 1.0) {
		    if (xf < yf) {
			yf /= xf;
			xf = 1.0;
		    } else {
			xf /= yf;
			yf = 1.0;
		    }
		}
	    }
	} else if (GD_drawing(g)->ratio_kind == R_EXPAND) {
	    if (GD_drawing(g)->size.x <= 0)
		scale_it = false;
	    else {
		xf = GD_drawing(g)->size.x / GD_bb(g).UR.x;
		yf = GD_drawing(g)->size.y / GD_bb(g).UR.y;
		if (xf > 1.0 && yf > 1.0) {
		    const double scale = fmin(xf, yf);
		    xf = yf = scale;
		} else
		    scale_it = false;
	    }
	} else if (GD_drawing(g)->ratio_kind == R_VALUE) {
	    desired = GD_drawing(g)->ratio;
	    actual = sz.y / sz.x;
	    if (actual < desired) {
		yf = desired / actual;
		xf = 1.0;
	    } else {
		xf = actual / desired;
		yf = 1.0;
	    }
	} else
	    scale_it = false;
	if (scale_it) {
	    if (GD_flip(g)) {
		SWAP(&xf, &yf);
	    }
	    for (n = GD_nlist(g); n; n = ND_next(n)) {
		ND_coord(n).x = round(ND_coord(n).x * xf);
		ND_coord(n).y = round(ND_coord(n).y * yf);
	    }
	    scale_bb(g, xf, yf);
	}
    }
}

/* make space for the leaf nodes of each rank */
static void make_leafslots(graph_t * g)
{
    int i, j, r;
    node_t *v;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	j = 0;
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    v = GD_rank(g)[r].v[i];
	    ND_order(v) = j;
	    if (ND_ranktype(v) == LEAFSET)
		j = j + ND_UF_size(v);
	    else
		j++;
	}
	if (j <= GD_rank(g)[r].n)
	    continue;
	node_t **new_v = gv_calloc(j + 1, sizeof(node_t*));
	for (i = GD_rank(g)[r].n - 1; i >= 0; i--) {
	    v = GD_rank(g)[r].v[i];
	    new_v[ND_order(v)] = v;
	}
	GD_rank(g)[r].n = j;
	new_v[j] = NULL;
	free(GD_rank(g)[r].v);
	GD_rank(g)[r].v = new_v;
    }
}

int ports_eq(edge_t * e, edge_t * f)
{
    return ED_head_port(e).defined == ED_head_port(f).defined
	    && ((ED_head_port(e).p.x == ED_head_port(f).p.x &&
		 ED_head_port(e).p.y == ED_head_port(f).p.y)
		|| !ED_head_port(e).defined)
	    && ((ED_tail_port(e).p.x == ED_tail_port(f).p.x &&
		 ED_tail_port(e).p.y == ED_tail_port(f).p.y)
		|| !ED_tail_port(e).defined);
}

static void expand_leaves(graph_t * g)
{
    int i, d;
    node_t *n;
    edge_t *e, *f;

    make_leafslots(g);
    for (n = GD_nlist(g); n; n = ND_next(n)) {
	if (ND_other(n).list)
	    for (i = 0; (e = ND_other(n).list[i]); i++) {
		if ((d = ND_rank(aghead(e)) - ND_rank(aghead(e))) == 0)
		    continue;
		f = ED_to_orig(e);
		if (!ports_eq(e, f)) {
		    zapinlist(&(ND_other(n)), e);
		    if (d == 1)
			fast_edge(e);
		    /*else unitize(e); ### */
		    i--;
		}
	    }
    }
}

/* Add left and right slacknodes to a cluster which
 * are used in the LP to constrain nodes not in g but
 * sharing its ranks to be to the left or right of g
 * by a specified amount.
 * The slacknodes ln and rn give the x position of the
 * left and right side of the cluster's bounding box. In
 * particular, any cluster labels on the left or right side
 * are inside.
 * If a cluster has a label, and we have rankdir!=LR, we make
 * sure the cluster is wide enough for the label. Note that
 * if the label is wider than the cluster, the nodes in the
 * cluster may not be centered.
 */
static void make_lrvn(graph_t * g)
{
    node_t *ln, *rn;

    if (GD_ln(g))
	return;
    ln = virtual_node(dot_root(g));
    ND_node_type(ln) = SLACKNODE;
    rn = virtual_node(dot_root(g));
    ND_node_type(rn) = SLACKNODE;

    if (GD_label(g) && g != dot_root(g) && !GD_flip(agroot(g))) {
	const double w = fmax(GD_border(g)[BOTTOM_IX].x, GD_border(g)[TOP_IX].x);
	make_aux_edge(ln, rn, w, 0);
    }

    GD_ln(g) = ln;
    GD_rn(g) = rn;
}

/* make left and right bounding box virtual nodes ln and rn
 * constrain interior nodes
 */
static void contain_nodes(graph_t * g)
{
    int margin, r;
    node_t *ln, *rn, *v;

    margin = late_int (g, G_margin, CL_OFFSET, 0);
    make_lrvn(g);
    ln = GD_ln(g);
    rn = GD_rn(g);
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	if (GD_rank(g)[r].n == 0)
	    continue;
	v = GD_rank(g)[r].v[0];
	if (v == NULL) {
	    agerrorf("contain_nodes clust %s rank %d missing node\n",
		  agnameof(g), r);
	    continue;
	}
	make_aux_edge(ln, v,
		      ND_lw(v) + margin + GD_border(g)[LEFT_IX].x, 0);
	v = GD_rank(g)[r].v[GD_rank(g)[r].n - 1];
	make_aux_edge(v, rn,
		      ND_rw(v) + margin + GD_border(g)[RIGHT_IX].x, 0);
    }
}

/* set g->drawing->size to a reasonable default.
 * returns a boolean to indicate if drawing is to
 * be scaled and filled */
static bool idealsize(graph_t * g, double minallowed)
{
    double xf, yf, f, R;
    pointf b, relpage, margin;

    /* try for one page */
    relpage = GD_drawing(g)->page;
    if (relpage.x < 0.001 || relpage.y < 0.001)
	return false;		/* no page was specified */
    margin = GD_drawing(g)->margin;
    relpage = sub_pointf(relpage, margin);
    relpage = sub_pointf(relpage, margin);
    b.x = GD_bb(g).UR.x;
    b.y = GD_bb(g).UR.y;
    xf = relpage.x / b.x;
    yf = relpage.y / b.y;
    if (xf >= 1.0 && yf >= 1.0)
	return false;		/* fits on one page */

    f = MIN(xf, yf);
    xf = yf = MAX(f, minallowed);

    R = ceil(xf * b.x / relpage.x);
    xf = R * relpage.x / b.x;
    R = ceil(yf * b.y / relpage.y);
    yf = R * relpage.y / b.y;
    GD_drawing(g)->size.x = b.x * xf;
    GD_drawing(g)->size.y = b.y * yf;
    return true;
}
