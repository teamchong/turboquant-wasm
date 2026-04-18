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
#include <limits.h>
#include <time.h>
#include <dotgen/dot.h>
#include <pack/pack.h>
#include <dotgen/aspect.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/debug.h>
#include <util/itos.h>
#include <util/streq.h>

static void
dot_init_subg(graph_t * g, graph_t* droot)
{
    graph_t* subg;

    if ((g != agroot(g)))
	agbindrec(g, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
    if (g == droot)
	GD_dotroot(agroot(g)) = droot;
	
    for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	dot_init_subg(subg, droot);
    }
}


static void 
dot_init_node(node_t * n)
{
    agbindrec(n, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);	//graph custom data
    common_init_node(n);
    gv_nodesize(n, GD_flip(agraphof(n)));
    alloc_elist(4, ND_in(n));
    alloc_elist(4, ND_out(n));
    alloc_elist(2, ND_flat_in(n));
    alloc_elist(2, ND_flat_out(n));
    alloc_elist(2, ND_other(n));
    ND_UF_size(n) = 1;
}

static void 
dot_init_edge(edge_t * e)
{
    char *tailgroup, *headgroup;
    agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//graph custom data
    common_init_edge(e);

    ED_weight(e) = late_int(e, E_weight, 1, 0);
    tailgroup = late_string(agtail(e), N_group, "");
    headgroup = late_string(aghead(e), N_group, "");
    ED_count(e) = ED_xpenalty(e) = 1;
    if (tailgroup[0] && (tailgroup == headgroup)) {
	ED_xpenalty(e) = CL_CROSS;
	ED_weight(e) *= 100;
    }
    if (nonconstraint_edge(e)) {
	ED_xpenalty(e) = 0;
	ED_weight(e) = 0;
    }

    {
	int showboxes = late_int(e, E_showboxes, 0, 0);
	if (showboxes > UCHAR_MAX) {
	    showboxes = UCHAR_MAX;
	}
	ED_showboxes(e) = (unsigned char)showboxes;
    }
    ED_minlen(e) = late_int(e, E_minlen, 1, 0);
}

void 
dot_init_node_edge(graph_t * g)
{
    node_t *n;
    edge_t *e;

    for (n = agfstnode(g); n; n = agnxtnode(g, n))
	dot_init_node(n);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e))
	    dot_init_edge(e);
    }
}

static void 
dot_cleanup_node(node_t * n)
{
    free_list(ND_in(n));
    free_list(ND_out(n));
    free_list(ND_flat_out(n));
    free_list(ND_flat_in(n));
    free_list(ND_other(n));
    free_label(ND_label(n));
    free_label(ND_xlabel(n));
    if (ND_shape(n))
	ND_shape(n)->fns->freefn(n);
    agdelrec(n, "Agnodeinfo_t");	
}

static void free_virtual_edge_list(node_t * n)
{
    edge_t *e;

    for (size_t i = ND_in(n).size - 1; i != SIZE_MAX; i--) {
	e = ND_in(n).list[i];
	delete_fast_edge(e);
	free(e->base.data);
	free(e);
    }
    for (size_t i = ND_out(n).size - 1; i != SIZE_MAX; i--) {
	e = ND_out(n).list[i];
	delete_fast_edge(e);
	free(e->base.data);
	free(e);
    }
}

static void free_virtual_node_list(node_t * vn)
{
    node_t *next_vn;

    while (vn) {
	next_vn = ND_next(vn);
	free_virtual_edge_list(vn);
	if (ND_node_type(vn) == VIRTUAL) {
	    free_list(ND_out(vn));
	    free_list(ND_in(vn));
	    free(vn->base.data);
	    free(vn);
	}
	vn = next_vn;
    }
}

static void 
dot_cleanup_graph(graph_t * g)
{
    int i;
    graph_t *subg;
    for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	dot_cleanup_graph(subg);
    }
    if (! agbindrec(g, "Agraphinfo_t", 0, true)) return;
    free(GD_drawing(g));
    GD_drawing(g) = NULL;
    free (GD_clust(g));
    free (GD_rankleader(g));

    free_list(GD_comp(g));
    if (GD_rank(g)) {
	for (i = GD_minrank(g); i <= GD_maxrank(g); i++)
	    free(GD_rank(g)[i].av);
	if (GD_minrank(g) == -1)
	    free(GD_rank(g)-1);
	else
	    free(GD_rank(g));
    }
    if (g != agroot(g)) {
	free_label (GD_label(g));
    }
}

/* delete the layout (but retain the underlying graph) */
void dot_cleanup(graph_t * g)
{
    node_t *n;
    edge_t *e;

    free_virtual_node_list(GD_nlist(g));
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    gv_cleanup_edge(e);
	}
	dot_cleanup_node(n);
    }
    dot_cleanup_graph(g);
}

#ifdef DEBUG
int
fastn (graph_t * g)
{
    node_t* u;
    int cnt = 0;
    for (u = GD_nlist(g); u; u = ND_next(u)) cnt++;
    return cnt;
}

#if DEBUG > 1
static void
dumpRanks (graph_t * g)
{
    int i, j;
    node_t* u;
    rank_t *rank = GD_rank(g);
    int rcnt = 0;
    for (i = GD_minrank(g); i <= GD_maxrank(g); i++) {
	fprintf (stderr, "[%d] :", i);
	for (j = 0; j < rank[i].n; j++) {
	    u = rank[i].v[j];
            rcnt++;
	    if (streq(agnameof(u),"virtual"))
	        fprintf (stderr, " %x", u);
	    else
	        fprintf (stderr, " %s", agnameof(u));
      
        }
	fprintf (stderr, "\n");
    }
    fprintf (stderr, "count %d rank count = %d\n", fastn(g), rcnt);
}
#endif
#endif


static void
remove_from_rank (Agraph_t * g, Agnode_t* n)
{
    Agnode_t* v = NULL;
    int j, rk = ND_rank(n);

    for (j = 0; j < GD_rank(g)[rk].n; j++) {
	v = GD_rank(g)[rk].v[j];
	if (v == n) {
	    for (j++; j < GD_rank(g)[rk].n; j++) {
		GD_rank(g)[rk].v[j-1] = GD_rank(g)[rk].v[j];
	    }
	    GD_rank(g)[rk].n--;
	    break;
	}
    }
    assert (v == n);  /* if found */
}

/* removeFill:
 * This removes all of the fill nodes added in mincross.
 * It appears to be sufficient to remove them only from the
 * rank array and fast node list of the root graph.
 */
static void
removeFill (Agraph_t * g)
{
    Agnode_t* n;
    Agnode_t* nxt;
    Agraph_t* sg = agsubg (g, "_new_rank", 0);

    if (!sg) return;
    for (n = agfstnode(sg); n; n = nxt) {
	nxt = agnxtnode(sg, n);
	delete_fast_node (g, n);
	remove_from_rank (g, n);
	dot_cleanup_node (n);
	agdelnode(g, n);
    }
    agdelsubg (g, sg);

}

#define agnodeattr(g,n,v) agattr_text(g,AGNODE,n,v)

static void
attach_phase_attrs (Agraph_t * g, int maxphase)
{
    Agsym_t* rk = agnodeattr(g,"rank","");
    Agsym_t* order = agnodeattr(g,"order","");
    Agnode_t* n;

    for (n = agfstnode(g); n; n = agnxtnode(g,n)) {
	if (maxphase >= 1) {
	    agxset(n, rk, ITOS(ND_rank(n)));
	}
	if (maxphase >= 2) {
	    agxset(n, order, ITOS(ND_order(n)));
	}
    }
}

/// @return 0 on success
static int dotLayout(Agraph_t *g) {
    int maxphase = late_int(g, agfindgraphattr(g,"phase"), -1, 1);

    setEdgeType (g, EDGETYPE_SPLINE);
    setAspect(g);

    dot_init_subg(g,g);
    dot_init_node_edge(g);

    GV_INFO("Starting phase 1 [dot_rank]");
    dot_rank(g);
    if (maxphase == 1) {
        attach_phase_attrs (g, 1);
        return 0;
    }
    GV_INFO("Starting phase 2 [dot_mincross]");
    const int rc = dot_mincross(g);
    if (rc != 0) {
        return rc;
    }
    if (maxphase == 2) {
        attach_phase_attrs (g, 2);
        return 0;
    }
    GV_INFO("Starting phase 3 [dot_position]");
    dot_position(g);
    if (maxphase == 3) {
        attach_phase_attrs (g, 2);  /* positions will be attached on output */
        return 0;
    }
    if (GD_flags(g) & NEW_RANK)
	removeFill (g);
    dot_sameports(g);
    const int r = dot_splines(g);
    if (r != 0) {
	return r;
    }
    if (mapbool(agget(g, "compound")))
	dot_compoundEdges(g);
    return 0;
}

static void
initSubg (Agraph_t* sg, Agraph_t* g)
{
    agbindrec(sg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
    GD_drawing(sg) = gv_alloc(sizeof(layout_t));
    GD_drawing(sg)->quantum = GD_drawing(g)->quantum; 
    GD_drawing(sg)->dpi = GD_drawing(g)->dpi;
    GD_gvc(sg) = GD_gvc (g);
    GD_charset(sg) = GD_charset (g);
    GD_rankdir2(sg) = GD_rankdir2 (g);
    GD_nodesep(sg) = GD_nodesep(g);
    GD_ranksep(sg) = GD_ranksep(g);
    GD_fontnames(sg) = GD_fontnames(g);
}

/* the packing library assumes all units are in inches stored in ND_pos, so we
 * have to copy the position info there.
 */
static void
attachPos (Agraph_t* g)
{
    node_t* np;

    for (np = agfstnode(g); np; np = agnxtnode(g, np)) {
	ND_pos(np) = gv_calloc(2, sizeof(double));
	ND_pos(np)[0] = PS2INCH(ND_coord(np).x);
	ND_pos(np)[1] = PS2INCH(ND_coord(np).y);
    }
}

/* Store new position info from pack library call, stored in ND_pos in inches,
 * back to ND_coord in points.
 */
static void
resetCoord (Agraph_t* g)
{
    for (node_t *np = agfstnode(g); np; np = agnxtnode(g, np)) {
	ND_coord(np).x = INCH2PS(ND_pos(np)[0]);
	ND_coord(np).y = INCH2PS(ND_pos(np)[1]);
	free(ND_pos(np));
	ND_pos(np) = NULL;
    }
}

static void
copyCluster (Agraph_t* scl, Agraph_t* cl)
{
    int nclust, j;
    Agraph_t* cg;

    agbindrec(cl, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
    GD_bb(cl) = GD_bb(scl);
    GD_label_pos(cl) = GD_label_pos(scl);
    memcpy(GD_border(cl), GD_border(scl), 4*sizeof(pointf));
    nclust = GD_n_cluster(cl) = GD_n_cluster(scl);
    GD_clust(cl) = gv_calloc(nclust + 1, sizeof(Agraph_t*));
    for (j = 1; j <= nclust; j++) {
	cg = mapClust(GD_clust(scl)[j]);
	GD_clust(cl)[j] = cg;
	copyCluster (GD_clust(scl)[j], cg);
    }
    /* transfer cluster label to original cluster */
    GD_label(cl) = GD_label(scl);
    GD_label(scl) = NULL;
}

/* Copy cluster tree and info from components to main graph.
 * Note that the original clusters have no Agraphinfo_t at this time.
 */
static void copyClusterInfo(size_t ncc, Agraph_t **ccs, Agraph_t *root) {
    int j, nclust = 0;
    Agraph_t* sg;
    Agraph_t* cg;

    for (size_t i = 0; i < ncc; i++)
	nclust += GD_n_cluster(ccs[i]);

    GD_n_cluster(root) = nclust;
    GD_clust(root) = gv_calloc(nclust + 1, sizeof(Agraph_t*));
    nclust = 1;
    for (size_t i = 0; i < ncc; i++) {
	sg = ccs[i];
	for (j = 1; j <= GD_n_cluster(sg); j++) {
	    cg = mapClust(GD_clust(sg)[j]);
	    GD_clust(root)[nclust++] = cg;
	    copyCluster (GD_clust(sg)[j], cg);
	}
    } 
}

/* Assume g has nodes.
 *
 * @return 0 on success
 */
static int doDot(Agraph_t *g) {
    Agraph_t **ccs;
    Agraph_t *sg;
    pack_info pinfo;
    int Pack = getPack(g, -1, CL_OFFSET);
    pack_mode mode = getPackModeInfo (g, l_undef, &pinfo);
    getPackInfo(g, l_node, CL_OFFSET, &pinfo);

    if (mode == l_undef && Pack < 0) {
	/* No pack information; use old dot with components
         * handled during layout
         */
	const int rc = dotLayout(g);
	if (rc != 0) {
	    return rc;
	}
    } else {
	/* fill in default values */
	if (mode == l_undef) 
	    pinfo.mode = l_graph;
	else if (Pack < 0)
	    Pack = CL_OFFSET;
	assert(Pack >= 0);
	pinfo.margin = (unsigned)Pack;
	pinfo.fixed = NULL;

          /* components using clusters */
	size_t ncc;
	ccs = cccomps(g, &ncc, 0);
	if (ncc == 1) {
	    const int rc = dotLayout(g);
	    if (rc != 0) {
		free(ccs);
		return rc;
	    }
	} else if (GD_drawing(g)->ratio_kind == R_NONE) {
	    pinfo.doSplines = true;

	    for (size_t i = 0; i < ncc; i++) {
		sg = ccs[i];
		initSubg (sg, g);
		const int rc = dotLayout (sg);
		if (rc != 0) {
		    free(ccs);
		    return rc;
		}
	    }
	    attachPos (g);
	    packSubgraphs(ncc, ccs, g, &pinfo);
	    resetCoord (g);
	    copyClusterInfo (ncc, ccs, g);
	} else {
	    /* Not sure what semantics should be for non-trivial ratio
             * attribute with multiple components.
             * One possibility is to layout nodes, pack, then apply the ratio
             * adjustment. We would then have to re-adjust all positions.
             */
	    const int rc = dotLayout(g);
	    if (rc != 0) {
		free(ccs);
		return rc;
	    }
	}

	for (size_t i = 0; i < ncc; i++) {
	    dot_cleanup_graph(ccs[i]);
	    agdelete(g, ccs[i]);
	}
	free(ccs);
    }
    return 0;
}

void dot_layout(Agraph_t * g)
{
    if (agnnodes(g)) {
	if (doDot(g) != 0) { // error?
	    return;
	}
    }
    dotneato_postprocess(g);
}

Agraph_t * dot_root (void* p)
{
    return GD_dotroot(agroot(p));
}

/**
 * @defgroup engines Graphviz layout engines
 * @{
 * @dir lib/dotgen
 * @brief [hierarchical or layered](https://en.wikipedia.org/wiki/Layered_graph_drawing) layout engine, API dotgen/dotprocs.h
 *
 * [Dot layout user manual](https://graphviz.org/docs/layouts/dot/)
 *
 * Other @ref engines
 * @}
 */
