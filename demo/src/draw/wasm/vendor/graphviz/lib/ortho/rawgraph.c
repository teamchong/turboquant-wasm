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

#include <ortho/rawgraph.h>
#include <stddef.h>
#include <util/alloc.h>
#include <util/list.h>

#define UNSCANNED 0
#define SCANNING  1
#define SCANNED   2

rawgraph *make_graph(size_t n) {
    rawgraph* g = gv_alloc(sizeof(rawgraph));
    g->nvs = n;
    g->vertices = gv_calloc(n, sizeof(vertex));
    for(size_t i = 0; i < n; ++i) {
        g->vertices[i].color = UNSCANNED;
    }
    return g;
}

void
free_graph(rawgraph* g)
{
    for(size_t i = 0; i < g->nvs; ++i)
        LIST_FREE(&g->vertices[i].adj_list);
    free (g->vertices);
    free (g);
}
 
void insert_edge(rawgraph *g, size_t v1, size_t v2) {
    if (!edge_exists(g, v1, v2)) {
      LIST_APPEND(&g->vertices[v1].adj_list, v2);
    }
}

void remove_redge(rawgraph *g, size_t v1, size_t v2) {
    LIST_REMOVE(&g->vertices[v1].adj_list, v2);
    LIST_REMOVE(&g->vertices[v2].adj_list, v1);
}

bool edge_exists(rawgraph *g, size_t v1, size_t v2) {
  return LIST_CONTAINS(&g->vertices[v1].adj_list, v2);
}

typedef LIST(size_t) int_stack_t;

static int DFS_visit(rawgraph *g, size_t v, int time, int_stack_t *sp) {
    vertex* vp;

    vp = g->vertices + v;
    vp->color = SCANNING;
    const adj_list_t adj = vp->adj_list;
    time = time + 1;

    for (size_t i = 0; i < LIST_SIZE(&adj); ++i) {
        const size_t id = LIST_GET(&adj, i);
        if(g->vertices[id].color == UNSCANNED)
            time = DFS_visit(g, id, time, sp);
    }
    vp->color = SCANNED;
    LIST_PUSH_BACK(sp, v);
    return time + 1;
}

void
top_sort(rawgraph* g)
{
    int time = 0;
    int count = 0;

    if (g->nvs == 0) return;
    if (g->nvs == 1) {
        g->vertices[0].topsort_order = count;
		return;
	}

    int_stack_t sp = {0};
    LIST_RESERVE(&sp, g->nvs);
    for(size_t i = 0; i < g->nvs; ++i) {
        if(g->vertices[i].color == UNSCANNED)
            time = DFS_visit(g, i, time, &sp);
    }
    while (!LIST_IS_EMPTY(&sp)) {
        const size_t v = LIST_POP_BACK(&sp);
        g->vertices[v].topsort_order = count;
        count++;
    }
    LIST_FREE(&sp);
}
