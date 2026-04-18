/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/


#include <assert.h>
#include <circogen/circular.h>
#include <circogen/block.h>
#include <util/alloc.h>

void initBlocklist(blocklist_t * bl)
{
    bl->first = NULL;
    bl->last = NULL;
}

block_t *mkBlock(Agraph_t * g)
{
    block_t *sn = gv_alloc(sizeof(block_t));
    initBlocklist(&sn->children);
    sn->sub_graph = g;
    return sn;
}

void freeBlock(block_t * sp)
{
    if (!sp)
	return;
    nodelist_free(&sp->circle_list);
    free(sp);
}

int blockSize(block_t * sp)
{
    return agnnodes (sp->sub_graph);
}

/// add block at end
void appendBlock(blocklist_t * bl, block_t * bp)
{
    bp->next = NULL;
    if (bl->last) {
	bl->last->next = bp;
	bl->last = bp;
    } else {
	bl->first = bp;
	bl->last = bp;
    }
}

/// add block at beginning
void insertBlock(blocklist_t * bl, block_t * bp)
{
    if (bl->first) {
	bp->next = bl->first;
	bl->first = bp;
    } else {
	bl->first = bp;
	bl->last = bp;
    }
}

#ifdef DEBUG
void printBlocklist(blocklist_t * snl)
{
    block_t *bp;
    for (bp = snl->first; bp; bp = bp->next) {
	Agnode_t *n;
	char *p;
	Agraph_t *g = bp->sub_graph;
	fprintf(stderr, "block=%s\n", agnameof(g));
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    Agedge_t *e;
	    if (PARENT(n))
		p = agnameof(PARENT(n));
	    else
		p = "<nil>";
	    fprintf(stderr, "  %s (%d %s)\n", agnameof(n), VAL(n), p);
	    for (e = agfstedge(g, n); e; e = agnxtedge(g, e, n)) {
		fprintf(stderr, "    %s--", agnameof(agtail(e)));
		fprintf(stderr, "%s\n", agnameof(aghead(e)));
	    }
	}
    }
}
#endif
