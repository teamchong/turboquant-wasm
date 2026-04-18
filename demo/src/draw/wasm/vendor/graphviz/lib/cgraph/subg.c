/**
 * @file
 * @brief implementation of subgraphs
 * @ingroup cgraph_subgraph
 * @ingroup cgraph_core
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

#include <cgraph/cghdr.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/alloc.h>

static Agraph_t *agfindsubg_by_id(Agraph_t * g, IDTYPE id)
{
    Agraph_t template;

    agdtdisc(g, g->g_id, &Ag_subgraph_id_disc);
    AGID(&template) = id;
    return dtsearch(g->g_id, &template);
}

static Agraph_t *localsubg(Agraph_t * g, IDTYPE id)
{
    Agraph_t *subg;

    subg = agfindsubg_by_id(g, id);
    if (subg)
	return subg;

    subg = gv_alloc(sizeof(Agraph_t));
    subg->clos = g->clos;
    subg->desc = g->desc;
    subg->desc.maingraph = false;
    subg->parent = g;
    subg->root = g->root;
    AGID(subg) = id;
    return agopen1(subg);
}

Agraph_t *agidsubg(Agraph_t *g, IDTYPE id) {
    return agfindsubg_by_id(g, id);
}

Agraph_t *agsubg(Agraph_t * g, char *name, int cflag)
{
    IDTYPE id;
    Agraph_t *subg;

    if (name && agmapnametoid(g, AGRAPH, name, &id, false)) {
	/* might already exist */
	if ((subg = agfindsubg_by_id(g, id)))
	    return subg;
    }

    if (cflag && agmapnametoid(g, AGRAPH, name, &id, true)) {	/* reserve id */
	subg = localsubg(g, id);
	agregister(g, AGRAPH, subg);
	return subg;
    }

    return NULL;
}

Agraph_t *agfstsubg(Agraph_t * g)
{
    return dtfirst(g->g_seq);
}

Agraph_t *agnxtsubg(Agraph_t * subg)
{
    Agraph_t *g;

    g = agparent(subg);
    return g? dtnext(g->g_seq, subg) : 0;
}

Agraph_t *agparent(Agraph_t * g)
{
    return g->parent;
}

/* this function is only responsible for deleting the entry
 * in the parent's subg dict.  the rest is done in agclose().
 */
int agdelsubg(Agraph_t * g, Agraph_t * subg)
{
    return dtdelete(g->g_seq, subg) != NULL  && dtdelete(g->g_id, subg) != NULL;
}
