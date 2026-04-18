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
#include <float.h>
#include <limits.h>
#include <sfdpgen/sfdp.h>
#include <neatogen/neato.h>
#include <neatogen/adjust.h>
#include <pack/pack.h>
#include <assert.h>
#include <sfdpgen/spring_electrical.h>
#include <neatogen/overlap.h>
#include <sfdpgen/stress_model.h>
#include <cgraph/cgraph.h>
#include <cgraph/gv_ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/alloc.h>
#include <util/strcasecmp.h>

static void sfdp_init_edge(edge_t * e)
{
    agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);	//node custom data
    common_init_edge(e);
}

static void sfdp_init_node_edge(graph_t * g)
{
    node_t *n;
    edge_t *e;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	neato_init_node(n);
    }
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e))
	    sfdp_init_edge(e);
    }
}

static void sfdp_init_graph(Agraph_t * g)
{
    int outdim;

    setEdgeType(g, EDGETYPE_LINE);
    outdim = late_int(g, agfindgraphattr(g, "dimen"), 2, 2);
    GD_ndim(agroot(g)) = late_int(g, agfindgraphattr(g, "dim"), outdim, 2);
    Ndim = GD_ndim(agroot(g)) = MIN(GD_ndim(agroot(g)), MAXDIM);
    GD_odim(agroot(g)) = MIN(outdim, Ndim);
    sfdp_init_node_edge(g);
}

/* getPos:
 */
static double *getPos(Agraph_t * g)
{
    Agnode_t *n;
    double *pos = gv_calloc(Ndim * agnnodes(g), sizeof(double));
    int ix, i;

    if (agfindnodeattr(g, "pos") == NULL)
	return pos;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	i = ND_id(n);
	if (hasPos(n)) {
	    for (ix = 0; ix < Ndim; ix++) {
		pos[i * Ndim + ix] = ND_pos(n)[ix];
	    }
	}
    }

    return pos;
}

static void sfdpLayout(graph_t * g, spring_electrical_control ctrl,
                       pointf pad) {
    double *sizes;
    double *pos;
    Agnode_t *n;
    int flag, i;
    int n_edge_label_nodes = 0, *edge_label_nodes = NULL;
    SparseMatrix A = makeMatrix(g);

    if (ctrl->overlap >= 0) {
	if (ctrl->edge_labeling_scheme > 0)
	    sizes = getSizes(g, pad, &n_edge_label_nodes, &edge_label_nodes);
	else
	    sizes = getSizes(g, pad, NULL, NULL);
    }
    else
	sizes = NULL;
    pos = getPos(g);

    multilevel_spring_electrical_embedding(Ndim, A, ctrl, sizes, pos, n_edge_label_nodes, edge_label_nodes, &flag);

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	double *npos = pos + (Ndim * ND_id(n));
	for (i = 0; i < Ndim; i++) {
	    ND_pos(n)[i] = npos[i];
	}
    }

    free(sizes);
    free(pos);
    SparseMatrix_delete (A);
    free(edge_label_nodes);
}

static int
late_smooth (graph_t* g, Agsym_t* sym, int dflt)
{
    char* s;
    int v;
    int rv;

    if (!sym) return dflt;
    s = agxget (g, sym);
    if (gv_isdigit(*s)) {
#if (defined(HAVE_GTS) || defined(HAVE_TRIANGLE))
	if ((v = atoi (s)) <= SMOOTHING_RNG)
#else
	if ((v = atoi (s)) <= SMOOTHING_SPRING)
#endif
	    rv = v;
	else
	    rv = dflt;
    }
    else if (gv_isalpha(*s)) {
	if (!strcasecmp(s, "avg_dist"))
	    rv = SMOOTHING_STRESS_MAJORIZATION_AVG_DIST;
	else if (!strcasecmp(s, "graph_dist"))
	    rv = SMOOTHING_STRESS_MAJORIZATION_GRAPH_DIST;
	else if (!strcasecmp(s, "none"))
	    rv = SMOOTHING_NONE;
	else if (!strcasecmp(s, "power_dist"))
	    rv = SMOOTHING_STRESS_MAJORIZATION_POWER_DIST;
#if (defined(HAVE_GTS) || defined(HAVE_TRIANGLE))
	else if (!strcasecmp(s, "rng"))
	    rv = SMOOTHING_RNG;
#endif
	else if (!strcasecmp(s, "spring"))
	    rv = SMOOTHING_SPRING;
#if (defined(HAVE_GTS) || defined(HAVE_TRIANGLE))
	else if (!strcasecmp(s, "triangle"))
	    rv = SMOOTHING_TRIANGLE;
#endif
	else
	    rv = dflt;
    }
    else
	rv = dflt;
    return rv;
}

static int
late_quadtree_scheme (graph_t* g, Agsym_t* sym, int dflt)
{
    char* s;
    int v;
    int rv;

    if (!sym) return dflt;
    s = agxget (g, sym);
    if (gv_isdigit(*s)) {
      if ((v = atoi (s)) <= QUAD_TREE_FAST && v >= QUAD_TREE_NONE){
	rv = v;
      }	else {
	rv = dflt;
      }
    } else if (gv_isalpha(*s)) {
      if (!strcasecmp(s, "none") || !strcasecmp(s, "false") ){
	rv = QUAD_TREE_NONE;
      } else if (!strcasecmp(s, "normal") || !strcasecmp(s, "true") || !strcasecmp(s, "yes")){
	rv = QUAD_TREE_NORMAL;
      } else if (!strcasecmp(s, "fast")){
	rv = QUAD_TREE_FAST;
      }	else {
	rv = dflt;
      }
    } else {
      rv = dflt;
    }
    return rv;
}


/* tuneControl:
 * Use user values to reset control
 */
static void
tuneControl (graph_t* g, spring_electrical_control ctrl)
{
    long seed;
    int init;

    seed = ctrl->random_seed;
    init = setSeed (g, INIT_RANDOM, &seed);
    if (init != INIT_RANDOM) {
        agwarningf("sfdp only supports start=random\n");
    }
    ctrl->random_seed = seed;

    ctrl->K = late_double(g, agfindgraphattr(g, "K"), -1.0, 0.0);
    ctrl->p = -1.0*late_double(g, agfindgraphattr(g, "repulsiveforce"), -AUTOP, 0.0);
    ctrl->multilevels = late_int(g, agfindgraphattr(g, "levels"), INT_MAX, 0);
    ctrl->smoothing = late_smooth(g, agfindgraphattr(g, "smoothing"), SMOOTHING_NONE);
    ctrl->tscheme = late_quadtree_scheme(g, agfindgraphattr(g, "quadtree"), QUAD_TREE_NORMAL);
    ctrl->beautify_leaves = mapbool(agget(g, "beautify"));
    ctrl->do_shrinking = mapBool(agget(g, "overlap_shrink"), true);
    ctrl->rotation = late_double(g, agfindgraphattr(g, "rotation"), 0.0, -DBL_MAX);
    ctrl->edge_labeling_scheme = late_int(g, agfindgraphattr(g, "label_scheme"), 0, 0);
    if (ctrl->edge_labeling_scheme > 4) {
	agwarningf("label_scheme = %d > 4 : ignoring\n", ctrl->edge_labeling_scheme);
	ctrl->edge_labeling_scheme = 0;
    }
}

void sfdp_layout(graph_t * g)
{
    int doAdjust;
    adjust_data am;
    sfdp_init_graph(g);
    doAdjust = (Ndim == 2);

    if (agnnodes(g)) {
	Agraph_t **ccs;
	Agraph_t *sg;
	expand_t sep;
	pointf pad;
	spring_electrical_control ctrl = spring_electrical_control_new();

	tuneControl (g, ctrl);
#if (defined(HAVE_GTS) || defined(HAVE_TRIANGLE))
	graphAdjustMode(g, &am, "prism0");
#else
	graphAdjustMode(g, &am, 0);
#endif

	pad.x = PS2INCH(DFLT_MARGIN);
	pad.y = PS2INCH(DFLT_MARGIN);

	if ((am.mode == AM_PRISM) && doAdjust) {
	    doAdjust = 0;  /* overlap removal done in sfdp */
	    ctrl->overlap = am.value;
    	    ctrl->initial_scaling = am.scaling;
	    sep = sepFactor(g);
	    if (sep.doAdd) {
		pad.x = PS2INCH(sep.x);
		pad.y = PS2INCH(sep.y);
	    }
	}
	else {
   		/* Turn off overlap removal in sfdp if prism not used */
	    ctrl->overlap = -1;
	}

	if (Verbose)
	    spring_electrical_control_print(ctrl);

	size_t ncc;
	ccs = ccomps(g, &ncc, 0);
	if (ncc == 1) {
	    sfdpLayout(g, ctrl, pad);
	    if (doAdjust) removeOverlapWith(g, &am);
	    spline_edges(g);
	} else {
	    pack_info pinfo;
	    getPackInfo(g, l_node, CL_OFFSET, &pinfo);
	    pinfo.doSplines = true;

	    for (size_t i = 0; i < ncc; i++) {
		sg = ccs[i];
		(void)graphviz_node_induce(sg, NULL);
		sfdpLayout(sg, ctrl, pad);
		if (doAdjust) removeOverlapWith(sg, &am);
		setEdgeType(sg, EDGETYPE_LINE);
		spline_edges(sg);
	    }
	    packSubgraphs(ncc, ccs, g, &pinfo);
	}
	for (size_t i = 0; i < ncc; i++) {
	    agdelete(g, ccs[i]);
	}
	free(ccs);
	spring_electrical_control_delete(ctrl);
    }

    dotneato_postprocess(g);
}

void sfdp_cleanup(graph_t * g)
{
    node_t *n;
    edge_t *e;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    gv_cleanup_edge(e);
	}
	gv_cleanup_node(n);
    }
}
 
/**
 * @dir lib/sfdpgen
 * @brief scalable [Force-Directed Placement](https://en.wikipedia.org/wiki/Force-directed_graph_drawing) layout engine, API sfdpgen/sfdp.h
 * @ingroup engines
 *
 * [SFDP layout user manual](https://graphviz.org/docs/layouts/sfdp/)
 *
 * Other @ref engines
 */
