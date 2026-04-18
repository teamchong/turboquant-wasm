/**
 * @file
 * @brief API circogen/circo.h: @ref circo_init_graph, @ref circoLayout, @ref circo_layout, @ref circo_cleanup
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

#include    <circogen/circular.h>
#include    <neatogen/adjust.h>
#include    <pack/pack.h>
#include    <neatogen/neatoprocs.h>
#include    <stddef.h>
#include    <stdbool.h>
#include    <string.h>
#include    <util/alloc.h>

static void circular_init_edge(edge_t * e)
{
    agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//node custom data
    common_init_edge(e);

    ED_factor(e) = late_double(e, E_weight, 1.0, 0.0);
}


static void circular_init_node_edge(graph_t * g)
{
    node_t *n;
    edge_t *e;
    int i = 0;
    ndata* alg = gv_calloc(agnnodes(g), sizeof(ndata));

    GD_neato_nlist(g) = gv_calloc(agnnodes(g) + 1, sizeof(node_t*));
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	neato_init_node(n);
	ND_alg(n) = alg + i;
	GD_neato_nlist(g)[i++] = n;
    }
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    circular_init_edge(e);
	}
    }
}


void circo_init_graph(graph_t * g)
{
    setEdgeType (g, EDGETYPE_LINE);
    /* GD_ndim(g) = late_int(g,agfindattr(g,"dim"),2,2); */
    Ndim = GD_ndim(agroot(g)) = 2;	/* The algorithm only makes sense in 2D */
    circular_init_node_edge(g);
}

/* Make a node in the derived graph, with the given name.
 * orig points to what it represents, either a real node or
 * a cluster. Copy size info from original node; needed for
 * adjustNodes and packSubgraphs.
 */
static node_t *makeDerivedNode(graph_t * dg, char *name, int isNode,
			       void *orig)
{
    node_t *n = agnode(dg, name,1);
    agbindrec(n, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);	//node custom data
    ND_alg(n) = gv_alloc(sizeof(cdata));
    if (isNode) {
	ND_pos(n) = gv_calloc(Ndim, sizeof(double));
	ND_lw(n) = ND_lw(orig);
	ND_rw(n) = ND_rw(orig);
	ND_ht(n) = ND_ht(orig);
	ORIGN(n) = orig;
    } else
	ORIGG(n) = orig;
    return n;
}

/* Construct a strict, undirected graph with no loops from g.
 * Construct the connected components with the provision that all
 * nodes in a block subgraph are considered connected.
 * Return array of components with number of components in cnt.
 * Each component has its blocks as subgraphs.
 * FIX: Check that blocks are disjoint.
 */
static Agraph_t **circomps(Agraph_t *g, size_t *cnt) {
    Agraph_t **ccs;
    Agraph_t *dg;
    Agnode_t *n, *v, *dt, *dh;
    Agedge_t *e;
    Agraph_t *sg;
    Agedge_t *ep;
    Agnode_t *p;

    dg = agopen("derived", Agstrictundirected,NULL);
    agbindrec (dg, "info", sizeof(Agraphinfo_t), true);
    GD_alg(g) = dg;  /* store derived graph for closing later */

    for (v = agfstnode(g); v; v = agnxtnode(g, v)) {
	if (DNODE(v))
	    continue;
	n = makeDerivedNode(dg, agnameof(v), 1, v);
	DNODE(v) = n;
    }

    for (v = agfstnode(g); v; v = agnxtnode(g, v)) {
	for (e = agfstout(g, v); e; e = agnxtout(g, e)) {
	    dt = DNODE(agtail(e));
	    dh = DNODE(aghead(e));
	    if (dt != dh) {
		agbindrec(agedge(dg, dt, dh, NULL, 1), "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//node custom data
	    }
	}
    }

    size_t c_cnt;
    ccs = ccomps(dg, &c_cnt, 0);

    /* replace block nodes with block contents */
    for (size_t i = 0; i < c_cnt; i++) {
	sg = ccs[i];

	/* add edges: since sg is a union of components, all edges
	 * of any node should be added, except loops.
	 */
	for (n = agfstnode(sg); n; n = agnxtnode(sg, n)) {
	    p = ORIGN(n);
	    for (e = agfstout(g, p); e; e = agnxtout(g, e)) {
		/* n = DNODE(agtail(e)); by construction since agtail(e) == p */
		dh = DNODE(aghead(e));
		if (n != dh) {
		    ep = agedge(dg, n, dh, NULL, 1);
		    agbindrec(ep, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//node custom data
		    agsubedge(sg,ep,1);
		}
	    }
	}
    }

    /* Finally, add edge data to edges */
    for (n = agfstnode(dg); n; n = agnxtnode(dg, n)) {
	for (e = agfstout(dg, n); e; e = agnxtout(dg, e)) {
	    ED_alg(e) = gv_alloc(sizeof(edata));
	}
    }

    *cnt = c_cnt;
    return ccs;
}

static void closeDerivedGraph(graph_t * g)
{
    node_t *n;
    edge_t *e;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    free(ED_alg(e));
	}
	free(ND_alg(n));
	free(ND_pos(n));
    }
    agclose(g);
}

/* Copy position of nodes in given subgraph of derived graph
 * to corresponding node in original graph.
 * FIX: consider assigning from n directly to ORIG(n).
 */
static void copyPosns(graph_t * g)
{
    node_t *n;
    node_t *v;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	v = ORIGN(n);
	ND_pos(v)[0] = ND_pos(n)[0];
	ND_pos(v)[1] = ND_pos(n)[1];
    }
}

void circoLayout(Agraph_t * g)
{
    Agraph_t **ccs;
    Agraph_t *sg;

    if (agnnodes(g)) {
	size_t ncc;
	ccs = circomps(g, &ncc);

	int blockCount = 0;
	if (ncc == 1) {
	    circularLayout(ccs[0], g, &blockCount);
	    copyPosns(ccs[0]);
	    adjustNodes(g);
	} else {
	    Agraph_t *dg = ccs[0]->root;
	    pack_info pinfo;
	    getPackInfo(g, l_node, CL_OFFSET, &pinfo);

	    for (size_t i = 0; i < ncc; i++) {
		sg = ccs[i];
		circularLayout(sg, g, &blockCount);
		adjustNodes(sg);
	    }
	    /* FIX: splines have not been calculated for dg
	     * To use, either do splines in dg and copy to g, or
	     * construct components of g from ccs and use that in packing.
	     */
	    packSubgraphs(ncc, ccs, dg, &pinfo);
	    for (size_t i = 0; i < ncc; i++)
		copyPosns(ccs[i]);
	}
	free(ccs);
    }
}

void circo_layout(Agraph_t * g)
{
    if (agnnodes(g) == 0) return;
    circo_init_graph(g);
    circoLayout(g);
	/* Release ND_alg as we need to reuse it during edge routing */
    free(ND_alg(agfstnode(g)));
    spline_edges(g);
    dotneato_postprocess(g);
}

/// ND_alg is freed in circo_layout
void circo_cleanup(graph_t * g)
{
    node_t *n;
    edge_t *e;

    n = agfstnode(g);
    if (n == NULL)
	return;			/* g is empty */

    closeDerivedGraph(GD_alg(g)); // delete derived graph

    /* free (ND_alg(n)); */
    for (; n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    gv_cleanup_edge(e);
	}
	gv_cleanup_node(n);
    }
    free(GD_neato_nlist(g));
}

/**
 * @dir lib/circogen
 * @brief [circular layout](https://en.wikipedia.org/wiki/Circular_layout) engine, API circogen/circo.h
 * @ingroup engines
 *
 * Biconnected components are put on circles.
 * block-cutnode tree is done recursively, with children placed
 * about parent block.
 *
 * [Circo layout user manual](https://graphviz.org/docs/layouts/circo/)
 *
 * Based on:
 * - Six and Tollis, "A Framework for Circular Drawings of
 *   Networks", GD '99, LNCS 1731, pp. 107-116;
 * - Six and Tollis, "Circular Drawings of Biconnected Graphs",
 *   Proc. ALENEX '99, pp 57-73.
 * - Kaufmann and Wiese, "Maintaining the Mental Map for Circular
 *   Drawings", GD '02, LNCS 2528, pp. 12-22.
 *
 * Other @ref engines
 */
