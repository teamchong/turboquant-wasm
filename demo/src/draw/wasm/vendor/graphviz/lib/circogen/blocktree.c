/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <cgraph/gv_math.h>
#include <circogen/blocktree.h>
#include <stdbool.h>
#include <util/agxbuf.h>

static void addNode(block_t * bp, Agnode_t * n)
{
    agsubnode(bp->sub_graph, n,1);
    BLOCK(n) = bp;
}

static Agraph_t *makeBlockGraph(Agraph_t * g, circ_state * state)
{
    agxbuf name = {0};

    agxbprint(&name, "_block_%d", state->blockCount++);
    Agraph_t *subg = agsubg(g, agxbuse(&name), 1);
    agxbfree(&name);
    agbindrec(subg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);	//node custom data
    return subg;
}

static block_t *makeBlock(Agraph_t * g, circ_state * state)
{
    Agraph_t *subg = makeBlockGraph(g, state);
    block_t *bp = mkBlock(subg);

    return bp;
}

DEFINE_LIST(estack, Agedge_t*)

/* Current scheme adds articulation point to first non-trivial child
 * block. If none exists, it will be added to its parent's block, if
 * non-trivial, or else given its own block.
 *
 * FIX:
 * This should be modified to:
 *  - allow user to specify which block gets a node, perhaps on per-node basis.
 *  - if an articulation point is not used in one of its non-trivial blocks,
 *    dummy edges should be added to preserve biconnectivity
 *  - turn on user-supplied blocks.
 *  - Post-process to move articulation point to largest block
 */
static void dfs(Agraph_t *g, Agnode_t *u, circ_state *state, bool isRoot,
                estack_t *stk) {
    LOWVAL(u) = VAL(u) = state->orderCount++;
    for (Agedge_t *e = agfstedge(g, u); e; e = agnxtedge(g, e, u)) {
	Agnode_t *v = aghead (e);
	if (v == u) {
            v = agtail(e);
	    if (!EDGEORDER(e)) EDGEORDER(e) = -1;
	}
	else {
	    if (!EDGEORDER(e)) EDGEORDER(e) = 1;
	}

        if (VAL(v) == 0) {   /* Since VAL(root) == 0, it gets treated as artificial cut point */
	    PARENT(v) = u;
            estack_push_back(stk, e);
            dfs(g, v, state, false, stk);
            LOWVAL(u) = imin(LOWVAL(u), LOWVAL(v));
            if (LOWVAL(v) >= VAL(u)) {       /* u is an articulation point */
		block_t *block = NULL;
		Agnode_t *np;
		Agedge_t *ep;
                do {
                    ep = estack_pop_back(stk);
		    if (EDGEORDER(ep) == 1)
			np = aghead (ep);
		    else
			np = agtail (ep);
		    if (!BLOCK(np)) {
			if (!block)
			    block = makeBlock(g, state);
			addNode(block, np);
		    }
                } while (ep != e);
		if (block) {	/* If block != NULL, it's not empty */
		    if (!BLOCK(u) && blockSize (block) > 1)
			addNode(block, u);
		    if (isRoot && BLOCK(u) == block)
			insertBlock(&state->bl, block);
		    else
			appendBlock(&state->bl, block);
		}
            }
        } else if (PARENT(u) != v) {
            LOWVAL(u) = imin(LOWVAL(u), VAL(v));
        }
    }
    if (isRoot && !BLOCK(u)) {
	block_t *block = makeBlock(g, state);
	addNode(block, u);
	insertBlock(&state->bl, block);
    }
}

static void find_blocks(Agraph_t * g, circ_state * state)
{
    Agnode_t *root = NULL;

    /*      check to see if there is a node which is set to be the root
     */
    if (state->rootname) {
	root = agfindnode(g, state->rootname);
    }
    if (!root && state->N_root) {
	for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    if (late_bool(ORIGN(n), state->N_root, false)) {
		root = n;
		break;
	    }
	}
    }

    if (!root)
	root = agfstnode(g);
    if (Verbose)
	fprintf (stderr, "root = %s\n", agnameof(root));
    estack_t stk = {0};
    dfs(g, root, state, true, &stk);
    estack_free(&stk);
}

/* Construct block tree by peeling nodes from block list in state.
 * When done, return root. The block list is empty
 * FIX: use largest block as root
 */
block_t *createBlocktree(Agraph_t * g, circ_state * state)
{
    block_t *next;

    find_blocks(g, state);

    block_t *bp = state->bl.first; // if root chosen, will be first
    /* Otherwise, just pick first as root */
    block_t *root = bp;

    /* Find node with minimum VAL value to find parent block */
    /* FIX: Should be some way to avoid search below.               */
    for (bp = bp->next; bp; bp = next) {
	Agnode_t *n;
	Agnode_t *parent;
	Agnode_t *child;
	Agraph_t *subg = bp->sub_graph;

	child = n = agfstnode(subg);

	int min = VAL(n);
	parent = PARENT(n);
	for (n = agnxtnode(subg, n); n; n = agnxtnode(subg, n)) {
	    if (VAL(n) < min) {
		child = n;
		min = VAL(n);
		parent = PARENT(n);
	    }
	}
	SET_PARENT(parent);
	CHILD(bp) = child;
	next = bp->next;	/* save next since list insertion destroys it */
	appendBlock(&BLOCK(parent)->children, bp);
    }
    initBlocklist(&state->bl);	/* zero out list */
    return root;
}

void freeBlocktree(block_t * bp)
{
    for (block_t *child = bp->children.first, *next; child; child = next) {
	next = child->next;
	freeBlocktree(child);
    }

    freeBlock(bp);
}

#ifdef DEBUG
static void indent(int i)
{
    while (i--)
	fputs("  ", stderr);
}

void print_blocktree(block_t * sn, int depth)
{
    indent(depth);
    Agraph_t *g = sn->sub_graph;
    fprintf(stderr, "%s:", agnameof(g));
    for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
	fprintf(stderr, " %s", agnameof(n));
    }
    fputs("\n", stderr);

    depth++;
    for (block_t *child = sn->children.first; child; child = child->next) {
	print_blocktree(child, depth);
    }
}

#endif
