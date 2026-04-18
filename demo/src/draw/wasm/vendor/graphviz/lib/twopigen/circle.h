/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#pragma once

#include "render.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
	uint64_t nStepsToLeaf;
	uint64_t subtreeSize;
	uint64_t nChildren;
	uint64_t nStepsToCenter;
	node_t *parent;
	double span;
	double theta;
    } rdata;

#define RDATA(n) ((rdata*)(ND_alg(n)))
#define SLEAF(n) (RDATA(n)->nStepsToLeaf)
#define STSIZE(n) (RDATA(n)->subtreeSize)
#define NCHILD(n) (RDATA(n)->nChildren)
#define SCENTER(n) (RDATA(n)->nStepsToCenter)
#define SPARENT(n) (RDATA(n)->parent)
#define SPAN(n) (RDATA(n)->span)
#define THETA(n) (RDATA(n)->theta)

    extern Agnode_t* circleLayout(Agraph_t * sg, Agnode_t * center);
    extern void twopi_layout(Agraph_t * g);
    extern void twopi_cleanup(Agraph_t * g);
    extern void twopi_init_graph(graph_t * g);

#ifdef __cplusplus
}
#endif
