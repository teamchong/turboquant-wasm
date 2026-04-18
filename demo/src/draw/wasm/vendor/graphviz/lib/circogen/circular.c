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
#include    <circogen/blocktree.h>
#include    <circogen/circpos.h>
#include    <util/agxbuf.h>

#define		MINDIST			1.0

/* Set attributes based on original root graph.
 * This is obtained by taking a node of g, finding its node
 * in the original graph, and finding that node's graph.
 */
static void initGraphAttrs(Agraph_t * g, circ_state * state)
{
    node_t *n = agfstnode(g);

    Agraph_t *rootg = agraphof(ORIGN(n));
    attrsym_t *G_mindist = agattr(rootg, AGRAPH, "mindist", NULL);
    attrsym_t *N_root = agattr(rootg, AGNODE, "root", NULL);
    char *rootname = agget(rootg, "root");
    initBlocklist(&state->bl);
    state->orderCount = 1;
    state->min_dist = late_double(rootg, G_mindist, MINDIST, 0.0);
    state->N_root = N_root;
    state->rootname = rootname;
}

static block_t*
createOneBlock(Agraph_t * g, circ_state * state)
{
    Agraph_t *subg;
    agxbuf name = {0};
    block_t *bp;
    Agnode_t* n;

    agxbprint(&name, "_block_%d", state->blockCount++);
    subg = agsubg(g, agxbuse(&name), 1);
    agxbfree(&name);
    bp = mkBlock(subg);

    for (n = agfstnode(g); n; n = agnxtnode(g,n)) {
	agsubnode(bp->sub_graph, n, 1);
	BLOCK(n) = bp;
    }

    return bp;
}

/* Do circular layout of g.
 * Assume g is strict.
 * g is a "connected" component of the derived graph of the
 * original graph.
 * We make state static so that it keeps a record of block numbers used
 * in a graph; it gets reset when a new root graph is used.
 */
void circularLayout(Agraph_t *g, Agraph_t *realg, int *blockCount) {
    block_t *root;

    if (agnnodes(g) == 1) {
	Agnode_t *n = agfstnode(g);
	ND_pos(n)[0] = 0;
	ND_pos(n)[1] = 0;
	return;
    }

    circ_state state = {.blockCount = *blockCount};
    initGraphAttrs(g, &state);

    if (mapbool(agget(realg, "oneblock")))
	root = createOneBlock(g, &state);
    else
	root = createBlocktree(g, &state);
    circPos(g, root, &state);

    /* cleanup:
     * We need to cleanup objects created in initGraphAttrs
     * and all blocks. All graph objects are components of the
     * initial derived graph and will be freed when it is closed.
     */
    freeBlocktree(root);

    *blockCount = state.blockCount;
}

#ifdef DEBUG
void prGraph(Agraph_t * g)
{
    Agnode_t *n;
    Agedge_t *e;

    fprintf(stderr, "%s\n", agnameof(g));
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	fprintf(stderr, "%s (%p)\n", agnameof(n), n);
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    fprintf(stderr, "%s", agnameof(n));
	    fprintf(stderr, " -- %s (%p)\n", agnameof(aghead(e)), e);
	}
    }
}

void prData(Agnode_t * n, int pass)
{
    char *pname;
    char *bname;
    char *tname;
    char *name1;
    char *name2;
    int dist1, dist2;

    if (PARENT(n))
	pname = agnameof(PARENT(n));
    else
	pname = "<P0>";
    if (BLOCK(n))
	bname = agnameof(BLOCK(n)->sub_graph);
    else {
	pname = "<B0>";
    bname = "";
    }
    fprintf(stderr, "%s: %x %s %s ", agnameof(n), FLAGS(n), pname, bname);
    switch (pass) {
    case 0:
	fprintf(stderr, "%d %d\n", VAL(n), LOWVAL(n));
	break;
    case 1:
	if (TPARENT(n))
	    tname = agnameof(TPARENT(n));
	else
	    tname = "<ROOT>";
	dist1 = DISTONE(n);
	if (dist1 > 0)
	    name1 = agnameof(LEAFONE(n));
	else
	    name1 = "<null>";
	dist2 = DISTTWO(n);
	if (dist2 > 0)
	    name2 = agnameof(LEAFTWO(n));
	else
	    name2 = "<null>";
	fprintf(stderr, "%s %s %d %s %d\n", tname, name1, dist1, name2,
		dist2);
	break;
    default:
	fprintf(stderr, "%d\n", POSITION(n));
	break;
    }
}
#endif
