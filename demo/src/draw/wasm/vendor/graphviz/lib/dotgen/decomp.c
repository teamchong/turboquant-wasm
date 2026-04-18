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
 * Decompose finds the connected components of a graph.
 * It searches the temporary edges and ignores non-root nodes.
 * The roots of the search are the real nodes of the graph,
 * but any virtual nodes discovered are also included in the
 * component.
 */

#include "config.h"

#include <dotgen/dot.h>
#include <stddef.h>
#include <stdint.h>
#include <util/alloc.h>
#include <util/list.h>

static node_t *Last_node;
static size_t Cmark;

static void 
begin_component(graph_t* g)
{
    Last_node = GD_nlist(g) = NULL;
}

static void 
add_to_component(graph_t* g, node_t * n)
{
    ND_mark(n) = Cmark;
    if (Last_node) {
	ND_prev(n) = Last_node;
	ND_next(Last_node) = n;
    } else {
	ND_prev(n) = NULL;
	GD_nlist(g) = n;
    }
    Last_node = n;
    ND_next(n) = NULL;
}

static void 
end_component(graph_t* g)
{
    size_t i = GD_comp(g).size++;
    GD_comp(g).list = gv_recalloc(GD_comp(g).list, GD_comp(g).size - 1,
                                  GD_comp(g).size, sizeof(node_t *));
    GD_comp(g).list[i] = GD_nlist(g);
}

typedef LIST(node_t *) node_stack_t;

static void push(node_stack_t *sp, node_t *np) {
  ND_mark(np) = Cmark + 1;
  LIST_PUSH_BACK(sp, np);
}

static node_t *pop(node_stack_t *sp) {

  if (LIST_IS_EMPTY(sp)) {
    return NULL;
  }

  return LIST_POP_BACK(sp);
}

/* iterative dfs for components.
 * We process the edges in reverse order of the recursive version to maintain
 * the processing order of the nodes.
 * Since we are using a stack, we need to indicate nodes on the stack. Nodes
 * unprocessed in this call to decompose will have mark < Cmark; processed nodes
 * will have mark=Cmark; so we use mark = Cmark+1 to indicate nodes on the
 * stack.
 */
static void search_component(node_stack_t *stk, graph_t *g, node_t *n) {
    node_t *other;

    push(stk, n);
    while ((n = pop(stk))) {
	if (ND_mark(n) == Cmark) continue;
	add_to_component(g, n);
	elist vec[] = {ND_flat_in(n), ND_flat_out(n), ND_in(n), ND_out(n)};

	for (size_t c = 0; c < sizeof(vec) / sizeof(vec[0]); ++c) {
	    if (vec[c].list && vec[c].size != 0) {
		for (size_t i = vec[c].size - 1; i != SIZE_MAX; i--) {
		  edge_t *const e = vec[c].list[i];
		    if ((other = aghead(e)) == n)
			other = agtail(e);
		    if (ND_mark(other) != Cmark && other == UF_find(other))
			push(stk, other);
		}
	    }
	}
    }
}

void decompose(graph_t * g, int pass)
{
    graph_t *subg;
    node_t *n, *v;
    node_stack_t stk = {0};

    if (++Cmark == 0)
	Cmark = 1;
    GD_comp(g).size = 0;
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	v = n;
	if (pass > 0 && (subg = ND_clust(v)))
	    v = GD_rankleader(subg)[ND_rank(v)];
	else if (v != UF_find(v))
	    continue;
	if (ND_mark(v) != Cmark) {
	    begin_component(g);
	    search_component(&stk, g, v);
	    end_component(g);
	}
    }
    LIST_FREE(&stk);
}
