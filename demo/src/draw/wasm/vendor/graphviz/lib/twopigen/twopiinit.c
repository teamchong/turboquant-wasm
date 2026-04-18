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
 * Written by Emden R. Gansner
 * Derived from Graham Wills' algorithm described in GD'97.
 */

#include    <cgraph/cgraph.h>
#include    <twopigen/circle.h>
#include    <neatogen/adjust.h>
#include    <pack/pack.h>
#include    <neatogen/neatoprocs.h>
#include    <stdbool.h>
#include    <stddef.h>
#include    <util/alloc.h>

static void twopi_init_edge(edge_t * e)
{
    agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//edge custom data
    common_init_edge(e);
    ED_factor(e) = late_double(e, E_weight, 1.0, 0.0);
}

static void twopi_init_node_edge(graph_t * g)
{
    node_t *n;
    edge_t *e;
    int i = 0;
    int n_nodes = agnnodes(g);

    rdata* alg = gv_calloc(n_nodes, sizeof(rdata));
    GD_neato_nlist(g) = gv_calloc(n_nodes + 1, sizeof(node_t*));
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	neato_init_node(n);
	ND_alg(n) = alg + i;
	GD_neato_nlist(g)[i++] = n;
    }
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    twopi_init_edge(e);
	}
    }
}

void twopi_init_graph(graph_t * g)
{
    setEdgeType (g, EDGETYPE_LINE);
    /* GD_ndim(g) = late_int(g,agfindgraphattr(g,"dim"),2,2); */
	Ndim = GD_ndim(agroot(g)) = 2;	/* The algorithm only makes sense in 2D */
    twopi_init_node_edge(g);
}

static Agnode_t* findRootNode (Agraph_t* sg, Agsym_t* rootattr)
{
    Agnode_t* n;

    for (n = agfstnode(sg); n; n = agnxtnode(sg,n)) {
	if (mapbool(agxget(n,rootattr))) return n;
    }
    return NULL;

}

/* twopi_layout:
 */
void twopi_layout(Agraph_t * g)
{
    Agnode_t *ctr = 0;
    char *s;
    int setRoot = 0;
    int setLocalRoot = 0;
    pointf sc;
    int r;
    Agsym_t* rootattr;

    if (agnnodes(g) == 0) return;

    twopi_init_graph(g);
    if ((s = agget(g, "root"))) {
	if (*s) {
	    ctr = agfindnode(g, s);
	    if (!ctr) {
		agwarningf("specified root node \"%s\" was not found.", s);
		agerr(AGPREV, "Using default calculation for root node\n");
		setRoot = 1;
	    }
	}
	else {
	    setRoot = 1;
	}
    }
    if ((rootattr = agattr(g, AGNODE, "root", 0))) {
	setLocalRoot = 1;
    }

    if ((s = agget(g, "scale")) && *s) {
	if ((r = sscanf (s, "%lf,%lf",&sc.x,&sc.y))) {
	    if (r == 1) sc.y = sc.x;
	}
    }

    if (agnnodes(g)) {
	Agraph_t **ccs;
	Agraph_t *sg;
	Agnode_t *c = NULL;
	Agnode_t *n;
	Agnode_t* lctr;

	size_t ncc;
	ccs = ccomps(g, &ncc, 0);
	if (ncc == 1) {
	    if (ctr)
		lctr = ctr;
	    else if (!rootattr || !(lctr = findRootNode(g, rootattr)))
		lctr = 0;
	    c = circleLayout(g, lctr);
	    if (setRoot && !ctr)
		ctr = c;
	    if (setLocalRoot && !lctr)
		agxset (c, rootattr, "1"); 
	    n = agfstnode(g);
	    free(ND_alg(n));
	    ND_alg(n) = NULL;
	    adjustNodes(g);
	    spline_edges(g);
	} else {
	    pack_info pinfo;
	    getPackInfo (g, l_node, CL_OFFSET, &pinfo);
	    pinfo.doSplines = false;

	    for (size_t i = 0; i < ncc; i++) {
		sg = ccs[i];
		if (ctr && agcontains(sg, ctr))
		    lctr = ctr;
		else if (!rootattr || !(lctr = findRootNode(sg, rootattr)))
		    lctr = 0;
		(void)graphviz_node_induce(sg, NULL);
		c = circleLayout(sg, lctr);
	        if (setRoot && !ctr)
		    ctr = c;
		if (setLocalRoot && (!lctr || (lctr == ctr)))
		    agxset (c, rootattr, "1"); 
		adjustNodes(sg);
	    }
	    n = agfstnode(g);
	    free(ND_alg(n));
	    ND_alg(n) = NULL;
	    packSubgraphs(ncc, ccs, g, &pinfo);
	    spline_edges(g);
	}
	for (size_t i = 0; i < ncc; i++) {
	    agdelete(g, ccs[i]);
	}
	free(ccs);
    }
    if (setRoot)
	agset (g, "root", agnameof (ctr)); 
    dotneato_postprocess(g);

}

static void twopi_cleanup_graph(graph_t * g)
{
    free(GD_neato_nlist(g));
}

/* twopi_cleanup:
 * The ND_alg data used by twopi is freed in twopi_layout
 * before edge routing as edge routing may use this field.
 */
void twopi_cleanup(graph_t * g)
{
    node_t *n;
    edge_t *e;

    n = agfstnode (g);
    if (!n) return; /* empty graph */
    /* free (ND_alg(n)); */
    for (; n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    gv_cleanup_edge(e);
	}
	gv_cleanup_node(n);
    }
    twopi_cleanup_graph(g);
}

/**
 * @dir lib/twopigen
 * @brief radial layout engine, API twopigen/circle.h
 * @ingroup engines
 *
 * Twopi means 2π.
 *
 * [Twopi layout user manual](https://graphviz.org/docs/layouts/twopi/)
 *
 * Other @ref engines
 */
