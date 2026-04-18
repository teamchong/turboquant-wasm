/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include    <assert.h>
#include    <cgraph/gv_ctype.h>
#include    <cgraph/gv_math.h>
#include    <cgraph/list.h>
#include    <twopigen/circle.h>
#include    <inttypes.h>
#include    <limits.h>
#include    <math.h>
#include    <stdbool.h>
#include    <stdint.h>
#include    <stdlib.h>
#include    <string.h>
#include    <util/alloc.h>
#include    <util/streq.h>
#define DEF_RANKSEP 1.00
#define UNSET 10.00

/* dfs to set distance from a particular leaf.
 * Note that termination is implicit in the test
 * for reduced number of steps. Proof?
 */
static void setNStepsToLeaf(Agraph_t * g, Agnode_t * n, Agnode_t * prev)
{
    Agnode_t *next;

    uint64_t nsteps = SLEAF(n) + 1;

    for (Agedge_t *ep = agfstedge(g, n); ep; ep = agnxtedge(g, ep, n)) {
	if ((next = agtail(ep)) == n)
	    next = aghead(ep);

	if (prev == next)
	    continue;

	if (nsteps < SLEAF(next)) {	/* handles loops and multiedges */
	    SLEAF(next) = nsteps;
	    setNStepsToLeaf(g, next, n);
	}
    }
}

/* isLeaf:
 * Return true if n is a leaf node.
 */
static bool isLeaf(Agraph_t * g, Agnode_t * n)
{
    Agnode_t *neighp = 0;
    Agnode_t *np;

    for (Agedge_t *ep = agfstedge(g, n); ep; ep = agnxtedge(g, ep, n)) {
	if ((np = agtail(ep)) == n)
	    np = aghead(ep);
	if (n == np)
	    continue;		/* loop */
	if (neighp) {
	    if (neighp != np)
		return false;	/* two different neighbors */
	} else
	    neighp = np;
    }
    return true;
}

static void initLayout(Agraph_t * g)
{
    int nnodes = agnnodes(g);
    assert(nnodes >= 0);
    uint64_t INF = (uint64_t)nnodes * (uint64_t)nnodes;

    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	SCENTER(n) = INF;
	THETA(n) = UNSET;	/* marks theta as unset, since 0 <= theta <= 2PI */
	if (isLeaf(g, n))
	    SLEAF(n) = 0;
	else
	    SLEAF(n) = INF;
    }
}

/*
 * Working recursively in from each leaf node (ie, each node
 * with nStepsToLeaf == 0; see initLayout), set the
 * minimum value of nStepsToLeaf for each node.  Using
 * that information, assign some node to be the centerNode.
*/
static Agnode_t *findCenterNode(Agraph_t * g)
{
    Agnode_t *center = NULL;
    uint64_t maxNStepsToLeaf = 0;

    /* dfs from each leaf node */
    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if (SLEAF(n) == 0)
	    setNStepsToLeaf(g, n, 0);
    }

    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if (center == NULL || SLEAF(n) > maxNStepsToLeaf) {
	    maxNStepsToLeaf = SLEAF(n);
	    center = n;
	}
    }
    return center;
}

DEFINE_LIST(node_queue, Agnode_t *)

/* bfs to create tree structure */
static void setNStepsToCenter(Agraph_t * g, Agnode_t * n)
{
    Agnode_t *next;
    Agsym_t* wt = agfindedgeattr(g,"weight");
    node_queue_t q = {0};

    node_queue_push_back(&q, n);
    while (!node_queue_is_empty(&q)) {
	n = node_queue_pop_front(&q);
	uint64_t nsteps = SCENTER(n) + 1;
	for (Agedge_t *ep = agfstedge(g, n); ep; ep = agnxtedge(g, ep, n)) {
	    if (wt && streq(ag_xget(ep,wt),"0")) continue;
	    if ((next = agtail(ep)) == n)
		next = aghead(ep);
	    if (nsteps < SCENTER(next)) {
		SCENTER(next) = nsteps;
		SPARENT(next) = n;
		NCHILD(n)++;
		node_queue_push_back(&q, next);
	    }
	}
    }
    node_queue_free(&q);
}

/*
 * Work out from the center and determine the value of
 * nStepsToCenter and parent node for each node.
 * Return UINT64_MAX if some node was not reached.
 */
static uint64_t setParentNodes(Agraph_t * sg, Agnode_t * center)
{
    uint64_t maxn = 0;
    uint64_t unset = SCENTER(center);

    SCENTER(center) = 0;
    SPARENT(center) = 0;
    setNStepsToCenter(sg, center);

    /* find the maximum number of steps from the center */
    for (Agnode_t *n = agfstnode(sg); n; n = agnxtnode(sg, n)) {
	if (SCENTER(n) == unset) {
	    return UINT64_MAX;
	}
	else if (SCENTER(n) > maxn) {
	    maxn = SCENTER(n);
	}
    }
    return maxn;
}

/* Sets each node's subtreeSize, which counts the number of 
 * leaves in subtree rooted at the node.
 * At present, this is done bottom-up.
 */
static void setSubtreeSize(Agraph_t * g)
{
    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if (NCHILD(n) > 0)
	    continue;
	STSIZE(n)++;
	for (Agnode_t *parent = SPARENT(n); parent; parent = SPARENT(parent)) {
	    STSIZE(parent)++;
	}
    }
}

static void setChildSubtreeSpans(Agraph_t * g, Agnode_t * n)
{
    Agnode_t *next;

    double ratio = SPAN(n) / STSIZE(n);
    for (Agedge_t *ep = agfstedge(g, n); ep; ep = agnxtedge(g, ep, n)) {
	if ((next = agtail(ep)) == n)
	    next = aghead(ep);
	if (SPARENT(next) != n)
	    continue;		/* handles loops */

	if (SPAN(next) != 0.0)
	    continue;		/* multiedges */
	SPAN(next) = ratio * STSIZE(next);

	if (NCHILD(next) > 0) {
	    setChildSubtreeSpans(g, next);
	}
    }
}

static void setSubtreeSpans(Agraph_t * sg, Agnode_t * center)
{
    SPAN(center) = 2 * M_PI;
    setChildSubtreeSpans(sg, center);
}

/// has the given value been assigned?
static bool is_set(double a) { return !is_exactly_equal(a, UNSET); }

 /* Set the node positions for the 2nd and later rings. */
static void setChildPositions(Agraph_t * sg, Agnode_t * n)
{
    Agnode_t *next;
    double theta;		/* theta is the lower boundary radius of the fan */

    if (SPARENT(n) == 0)	/* center */
	theta = 0;
    else
	theta = THETA(n) - SPAN(n) / 2;

    for (Agedge_t *ep = agfstedge(sg, n); ep; ep = agnxtedge(sg, ep, n)) {
	if ((next = agtail(ep)) == n)
	    next = aghead(ep);
	if (SPARENT(next) != n)
	    continue;		/* handles loops */
	if (is_set(THETA(next)))
	    continue;		/* handles multiedges */

	THETA(next) = theta + SPAN(next) / 2.0;
	theta += SPAN(next);

	if (NCHILD(next) > 0)
	    setChildPositions(sg, next);
    }
}

static void setPositions(Agraph_t * sg, Agnode_t * center)
{
    THETA(center) = 0;
    setChildPositions(sg, center);
}

/* getRankseps:
 * Return array of doubles of size maxrank+1 containing the radius of each
 * rank.  Position 0 always contains 0. Use the colon-separated list of 
 * doubles provided by ranksep to get the deltas for each additional rank.
 * If not enough values are provided, the last value is repeated.
 * If the ranksep attribute is not provided, use DEF_RANKSEP for all values. 
 */ 
static double*
getRankseps (Agraph_t* g, uint64_t maxrank)
{
    char *p;
    char *endp;
    char c;
    uint64_t rk = 1;
    double* ranks = gv_calloc(maxrank + 1, sizeof(double));
    double xf = 0.0, delx = 0.0, d;

    if ((p = late_string(g, agfindgraphattr(g->root, "ranksep"), NULL))) {
	while (rk <= maxrank && (d = strtod (p, &endp)) > 0) {
	    delx = fmax(d, MIN_RANKSEP);
	    xf += delx;
	    ranks[rk++] = xf;
	    p = endp;
	    while ((c = *p) && (gv_isspace(c) || c == ':'))
		p++;
	}
    }
    else {
	delx = DEF_RANKSEP;
    }

    for (uint64_t i = rk; i <= maxrank; i++) {
	xf += delx;
	ranks[i] = xf;
    }

    return ranks;
}

static void setAbsolutePos(Agraph_t * g, uint64_t maxrank)
{
    double* ranksep = getRankseps (g, maxrank);
    if (Verbose) {
	fputs ("Rank separation = ", stderr);
	for (uint64_t i = 0; i <= maxrank; i++)
	    fprintf (stderr, "%.03lf ", ranksep[i]);
	fputs ("\n", stderr);
    }

    /* Convert circular to cartesian coordinates */
    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	double hyp = ranksep[SCENTER(n)];
	ND_pos(n)[0] = hyp * cos(THETA(n));
	ND_pos(n)[1] = hyp * sin(THETA(n));
    }
    free (ranksep);
}

/* circleLayout:
 *  We assume sg is is connected and non-empty.
 *  Also, if center != 0, we are guaranteed that center is
 *  in the graph.
 */
Agnode_t* circleLayout(Agraph_t * sg, Agnode_t * center)
{
    if (agnnodes(sg) == 1) {
	Agnode_t *n = agfstnode(sg);
	ND_pos(n)[0] = 0;
	ND_pos(n)[1] = 0;
	return center;
    }

    initLayout(sg);

    if (!center)
	center = findCenterNode(sg);

    uint64_t maxNStepsToCenter = setParentNodes(sg,center);
    if (Verbose)
	fprintf(stderr, "root = %s max steps to root = %" PRIu64 "\n",
	        agnameof(center), maxNStepsToCenter);
    if (maxNStepsToCenter == UINT64_MAX) {
	agerrorf("twopi: use of weight=0 creates disconnected component.\n");
	return center;
    }

    setSubtreeSize(sg);

    setSubtreeSpans(sg, center);

    setPositions(sg, center);

    setAbsolutePos(sg, maxNStepsToCenter);
    return center;
}
