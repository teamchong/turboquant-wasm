/**
 * @file
 * @brief Network Simplex algorithm for ranking nodes of a DAG, @ref rank, @ref rank2
 * @ingroup common_render
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

#include <assert.h>
#include <common/render.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <util/alloc.h>
#include <util/exit.h>
#include <util/gv_math.h>
#include <util/list.h>
#include <util/overflow.h>
#include <util/prisize_t.h>
#include <util/streq.h>

static void dfs_cutval(node_t * v, edge_t * par);
static int dfs_range_init(node_t *v);
static int dfs_range(node_t * v, edge_t * par, int low);
static int x_val(edge_t * e, node_t * v, int dir);
#ifdef DEBUG
static void check_cycles(graph_t * g);
#endif

#define LENGTH(e)		(ND_rank(aghead(e)) - ND_rank(agtail(e)))
#define SLACK(e)		(LENGTH(e) - ED_minlen(e))
#define SEQ(a,b,c)		((a) <= (b) && (b) <= (c))
#define TREE_EDGE(e)	(ED_tree_index(e) >= 0)

typedef struct {
    graph_t *G;
    LIST(edge_t *) Tree_edge;
    size_t S_i;			/* search index for enter_edge */
    size_t N_edges, N_nodes;
    int Search_size;
} network_simplex_ctx_t;

enum { SEARCHSIZE = 30 };

static int add_tree_edge(network_simplex_ctx_t *ctx, edge_t * e)
{
    if (TREE_EDGE(e)) {
	agerrorf("add_tree_edge: missing tree edge\n");
	return -1;
    }
    assert(LIST_SIZE(&ctx->Tree_edge) <= INT_MAX);
    ED_tree_index(e) = (int)LIST_SIZE(&ctx->Tree_edge);
    LIST_APPEND(&ctx->Tree_edge, e);
    node_t *n = agtail(e);
    ND_mark(n) = true;
    ND_tree_out(n).list[ND_tree_out(n).size++] = e;
    ND_tree_out(n).list[ND_tree_out(n).size] = NULL;
    if (ND_out(n).list[ND_tree_out(n).size - 1] == 0) {
	agerrorf("add_tree_edge: empty outedge list\n");
	return -1;
    }
    n = aghead(e);
    ND_mark(n) = true;
    ND_tree_in(n).list[ND_tree_in(n).size++] = e;
    ND_tree_in(n).list[ND_tree_in(n).size] = NULL;
    if (ND_in(n).list[ND_tree_in(n).size - 1] == 0) {
	agerrorf("add_tree_edge: empty inedge list\n");
	return -1;
    }
    return 0;
}

/**
 * Invalidate DFS attributes by walking up the tree from to_node till lca
 * (inclusively). Called when updating tree to improve pruning in dfs_range().
 * Assigns ND_low(n) = -1 for the affected nodes.
 */
static void invalidate_path(node_t *lca, node_t *to_node) {
    while (true) {
        if (ND_low(to_node) == -1)
          break;

        ND_low(to_node) = -1;

        edge_t *e = ND_par(to_node);
        if (e == NULL)
          break;

        if (ND_lim(to_node) >= ND_lim(lca)) {
          if (to_node != lca)
            agerrorf("invalidate_path: skipped over LCA\n");
          break;
        }

        if (ND_lim(agtail(e)) > ND_lim(aghead(e)))
          to_node = agtail(e);
        else
          to_node = aghead(e);
    }
}

static void exchange_tree_edges(network_simplex_ctx_t *ctx, edge_t * e, edge_t * f)
{
    ED_tree_index(f) = ED_tree_index(e);
    assert(ED_tree_index(e) >= 0);
    LIST_SET(&ctx->Tree_edge, (size_t)ED_tree_index(e), f);
    ED_tree_index(e) = -1;

    node_t *n = agtail(e);
    size_t i = --ND_tree_out(n).size;
    size_t j;
    for (j = 0; j <= i; j++)
	if (ND_tree_out(n).list[j] == e)
	    break;
    ND_tree_out(n).list[j] = ND_tree_out(n).list[i];
    ND_tree_out(n).list[i] = NULL;
    n = aghead(e);
    i = --ND_tree_in(n).size;
    for (j = 0; j <= i; j++)
	if (ND_tree_in(n).list[j] == e)
	    break;
    ND_tree_in(n).list[j] = ND_tree_in(n).list[i];
    ND_tree_in(n).list[i] = NULL;

    n = agtail(f);
    ND_tree_out(n).list[ND_tree_out(n).size++] = f;
    ND_tree_out(n).list[ND_tree_out(n).size] = NULL;
    n = aghead(f);
    ND_tree_in(n).list[ND_tree_in(n).size++] = f;
    ND_tree_in(n).list[ND_tree_in(n).size] = NULL;
}

static
void init_rank(network_simplex_ctx_t *ctx)
{
    edge_t *e;

    LIST(node_t *) Q = {0};
    LIST_RESERVE(&Q, ctx->N_nodes);
    size_t ctr = 0;

    for (node_t *v = GD_nlist(ctx->G); v; v = ND_next(v)) {
	if (ND_priority(v) == 0)
	    LIST_PUSH_BACK(&Q, v);
    }

    while (!LIST_IS_EMPTY(&Q)) {
	node_t *const v = LIST_POP_FRONT(&Q);
	ND_rank(v) = 0;
	ctr++;
	for (int i = 0; (e = ND_in(v).list[i]); i++)
	    ND_rank(v) = MAX(ND_rank(v), ND_rank(agtail(e)) + ED_minlen(e));
	for (int i = 0; (e = ND_out(v).list[i]); i++) {
	    if (--ND_priority(aghead(e)) <= 0)
		LIST_PUSH_BACK(&Q, aghead(e));
	}
    }
    if (ctr != ctx->N_nodes) {
	agerrorf("trouble in init_rank\n");
	for (node_t *v = GD_nlist(ctx->G); v; v = ND_next(v))
	    if (ND_priority(v))
		agerr(AGPREV, "\t%s %d\n", agnameof(v), ND_priority(v));
    }
    LIST_FREE(&Q);
}

static edge_t *leave_edge(network_simplex_ctx_t *ctx)
{
    edge_t *f, *rv = NULL;
    int cnt = 0;

    size_t j = ctx->S_i;
    while (ctx->S_i < LIST_SIZE(&ctx->Tree_edge)) {
	if (ED_cutvalue(f = LIST_GET(&ctx->Tree_edge, ctx->S_i)) < 0) {
	    if (rv) {
		if (ED_cutvalue(rv) > ED_cutvalue(f))
		    rv = f;
	    } else
		rv = LIST_GET(&ctx->Tree_edge, ctx->S_i);
	    if (++cnt >= ctx->Search_size)
		return rv;
	}
	ctx->S_i++;
    }
    if (j > 0) {
	ctx->S_i = 0;
	while (ctx->S_i < j) {
	    if (ED_cutvalue(f = LIST_GET(&ctx->Tree_edge, ctx->S_i)) < 0) {
		if (rv) {
		    if (ED_cutvalue(rv) > ED_cutvalue(f))
			rv = f;
		} else
		    rv = LIST_GET(&ctx->Tree_edge, ctx->S_i);
		if (++cnt >= ctx->Search_size)
		    return rv;
	    }
	    ctx->S_i++;
	}
    }
    return rv;
}

static edge_t *dfs_enter_outedge(node_t *v, int Low, int Lim) {
    edge_t *e;
    edge_t *Enter = NULL;
    int Slack = INT_MAX;

    LIST(node_t *) todo = {0};
    LIST_APPEND(&todo, v);

    while (!LIST_IS_EMPTY(&todo)) {
	v = LIST_POP_BACK(&todo);

	for (int i = 0; (e = ND_out(v).list[i]); i++) {
	    if (!TREE_EDGE(e)) {
		if (!SEQ(Low, ND_lim(aghead(e)), Lim)) {
		    const int slack = SLACK(e);
		    if (slack < Slack || Enter == NULL) {
			Enter = e;
			Slack = slack;
		    }
		}
	    } else if (ND_lim(aghead(e)) < ND_lim(v))
		LIST_APPEND(&todo, aghead(e));
	}
	for (int i = 0; (e = ND_tree_in(v).list[i]) && Slack > 0; i++)
	    if (ND_lim(agtail(e)) < ND_lim(v))
		LIST_APPEND(&todo, agtail(e));

    }
    LIST_FREE(&todo);

    return Enter;
}

static edge_t *dfs_enter_inedge(node_t *v, int Low, int Lim) {
    edge_t *e;

    edge_t *Enter = NULL;
    int Slack = INT_MAX;

    LIST(node_t *) todo = {0};
    LIST_APPEND(&todo, v);

    while (!LIST_IS_EMPTY(&todo)) {
	v = LIST_POP_BACK(&todo);

	for (int i = 0; (e = ND_in(v).list[i]); i++) {
	    if (!TREE_EDGE(e)) {
		if (!SEQ(Low, ND_lim(agtail(e)), Lim)) {
		    const int slack = SLACK(e);
		    if (slack < Slack || Enter == NULL) {
			Enter = e;
			Slack = slack;
		    }
		}
	    } else if (ND_lim(agtail(e)) < ND_lim(v))
		LIST_APPEND(&todo, agtail(e));
	}
	for (int i = 0; (e = ND_tree_out(v).list[i]) && Slack > 0; i++)
	    if (ND_lim(aghead(e)) < ND_lim(v))
		LIST_APPEND(&todo, aghead(e));

    }
    LIST_FREE(&todo);

    return Enter;
}

static edge_t *enter_edge(edge_t *e) {
    node_t *v;
    bool outsearch;

    /* v is the down node */
    if (ND_lim(agtail(e)) < ND_lim(aghead(e))) {
	v = agtail(e);
	outsearch = false;
    } else {
	v = aghead(e);
	outsearch = true;
    }
    if (outsearch)
	return dfs_enter_outedge(v, ND_low(v), ND_lim(v));
    return dfs_enter_inedge(v, ND_low(v), ND_lim(v));
}

static void init_cutvalues(network_simplex_ctx_t *ctx)
{
    dfs_range_init(GD_nlist(ctx->G));
    dfs_cutval(GD_nlist(ctx->G), NULL);
}

/* functions for initial tight tree construction */
// borrow field from network simplex - overwritten in init_cutvalues() forgive me
#define ND_subtree(n) (subtree_t*)ND_par(n)
#define ND_subtree_set(n,value) (ND_par(n) = (edge_t*)value)

typedef struct subtree_s {
        node_t *rep;            /* some node in the tree */
        int    size;            /* total tight tree size */
        size_t    heap_index; ///< required to find non-min elts when merged
        struct subtree_s *par;  /* union find */
} subtree_t;

/// is this subtree stored in an STheap?
static bool on_heap(const subtree_t *tree) {
  return tree->heap_index != SIZE_MAX;
}

/// state for use in `tight_subtree_search`
typedef struct {
  Agnode_t *v;
  int in_i;  ///< iteration counter through `ND_in(v).list`
  int out_i; ///< iteration counter through `ND_out(v).list`
  int rv;    ///< result value
} tst_t;

/* find initial tight subtrees */
static int tight_subtree_search(network_simplex_ctx_t *ctx, Agnode_t *v, subtree_t *st)
{
    Agedge_t *e;

    int rv = 1;
    ND_subtree_set(v,st);

    LIST(tst_t) todo = {0};
    LIST_PUSH_BACK(&todo, ((tst_t){.v = v, .rv = 1}));

    while (!LIST_IS_EMPTY(&todo)) {
        bool updated = false;
        tst_t *top = LIST_BACK(&todo);

        for (; (e = ND_in(top->v).list[top->in_i]); top->in_i++) {
            if (TREE_EDGE(e)) continue;
            if (ND_subtree(agtail(e)) == 0 && SLACK(e) == 0) {
                if (add_tree_edge(ctx, e) != 0) {
                    LIST_DROP_BACK(&todo);
                    if (LIST_IS_EMPTY(&todo)) {
                        rv = -1;
                    } else {
                        --LIST_BACK(&todo)->rv;
                    }
                } else {
                    ++top->in_i;
                    ND_subtree_set(agtail(e), st);
                    const tst_t next = {.v = agtail(e), .rv = 1};
                    LIST_PUSH_BACK(&todo, next);
                }
                updated = true;
                break;
            }
        }
        if (updated) {
            continue;
        }

        for (; (e = ND_out(top->v).list[top->out_i]); top->out_i++) {
            if (TREE_EDGE(e)) continue;
            if (ND_subtree(aghead(e)) == 0 && SLACK(e) == 0) {
                if (add_tree_edge(ctx, e) != 0) {
                    LIST_DROP_BACK(&todo);
                    if (LIST_IS_EMPTY(&todo)) {
                        rv = -1;
                    } else {
                        --LIST_BACK(&todo)->rv;
                    }
                } else {
                    ++top->out_i;
                    ND_subtree_set(aghead(e), st);
                    const tst_t next = {.v = aghead(e), .rv = 1};
                    LIST_PUSH_BACK(&todo, next);
                }
                updated = true;
                break;
            }
        }
        if (updated) {
          continue;
        }

        const tst_t last = LIST_POP_BACK(&todo);
        if (LIST_IS_EMPTY(&todo)) {
            rv = last.rv;
        } else {
            LIST_BACK(&todo)->rv += last.rv;
        }
    }

    LIST_FREE(&todo);

    return rv;
}

static subtree_t *find_tight_subtree(network_simplex_ctx_t *ctx, Agnode_t *v)
{
    subtree_t *rv = gv_alloc(sizeof(subtree_t));
    rv->rep = v;
    rv->size = tight_subtree_search(ctx,v,rv);
    if (rv->size < 0) {
        free(rv);
        return NULL;
    }
    rv->par = rv;
    return rv;
}

typedef struct STheap_s {
        subtree_t       **elt;
        size_t          size;
} STheap_t;

static subtree_t *STsetFind(Agnode_t *n0)
{
  subtree_t *s0 = ND_subtree(n0);
  while  (s0->par && s0->par != s0) {
    if (s0->par->par) {s0->par = s0->par->par;}  /* path compression for the code weary */
    s0 = s0->par;
  }
  return s0;
}
 
static subtree_t *STsetUnion(subtree_t *s0, subtree_t *s1)
{
  subtree_t *r0, *r1, *r;

  for (r0 = s0; r0->par && r0->par != r0; r0 = r0->par);
  for (r1 = s1; r1->par && r1->par != r1; r1 = r1->par);
  if (r0 == r1) return r0;  /* safety code but shouldn't happen */
  assert(on_heap(r0) || on_heap(r1));
  if (!on_heap(r1)) r = r0;
  else if (!on_heap(r0)) r = r1;
  else if (r1->size < r0->size) r = r0;
  else r = r1;

  r0->par = r1->par = r;
  r->size = r0->size + r1->size;
  assert(on_heap(r));
  return r;
}

/* find tightest edge to another tree incident on the given tree */
static Agedge_t *inter_tree_edge_search(Agnode_t *v) {

    // per-node state
    typedef struct {
      Agnode_t *v;
      subtree_t *ts;
      Agnode_t *from;
      int out_i; ///< counter for iterating through `ND_out(v).list`
      int in_i;  ///< counter for iterating through `ND_in(v).list`
    } state_t;

    LIST(state_t) todo = {0};
    LIST_PUSH_BACK(&todo, ((state_t){.v = v, .ts = STsetFind(v)}));

    Agedge_t *best = NULL;

    while (!LIST_IS_EMPTY(&todo)) {
      state_t *const s = LIST_BACK(&todo);
      if (s->out_i == 0 && s->in_i == 0 && best != NULL && SLACK(best) == 0) {
          LIST_DROP_BACK(&todo);
          continue;
      }

      bool updated = false;
      Agedge_t *e;
      for (; (e = ND_out(s->v).list[s->out_i]) != NULL; ++s->out_i) {
          if (TREE_EDGE(e)) {
            if (aghead(e) == s->from) continue; // do not search back in tree
            ++s->out_i;
            LIST_PUSH_BACK(&todo, ((state_t){.v = aghead(e),
                                             .ts = STsetFind(aghead(e)),
                                             .from = s->v}));
              // search forward in tree
            updated = true;
            break;
          } else {
            if (STsetFind(aghead(e)) != s->ts) { // encountered candidate edge
              if (best == NULL || SLACK(e) < SLACK(best)) best = e;
            }
            // else ignore non-tree edge between nodes in the same tree
          }
      }
      if (updated) {
          continue;
      }

      // the following code must mirror the above, but for in-edges
      for (; (e = ND_in(s->v).list[s->in_i]); ++s->in_i) {
          if (TREE_EDGE(e)) {
            if (agtail(e) == s->from) continue;
            ++s->in_i;
            LIST_PUSH_BACK(&todo, ((state_t){.v = agtail(e),
                                             .ts = STsetFind(agtail(e)),
                                             .from = s->v}));
            updated = true;
            break;
          } else {
            if (STsetFind(agtail(e)) != s->ts) {
              if (best == NULL || SLACK(e) < SLACK(best)) best = e;
            }
          }
      }
      if (updated) {
          continue;
      }

      LIST_DROP_BACK(&todo);
    }

    LIST_FREE(&todo);
    return best;
}

static Agedge_t *inter_tree_edge(subtree_t *tree)
{
    return inter_tree_edge_search(tree->rep);
}

static size_t STheapsize(const STheap_t *heap) { return heap->size; }

static void STheapify(STheap_t *heap, size_t i) {
    subtree_t **elt = heap->elt;
    do {
        const size_t left = 2 * (i + 1) - 1;
        const size_t right = 2 * (i + 1);
        size_t smallest = i;
        if (left < heap->size && elt[left]->size < elt[smallest]->size) smallest = left;
        if (right < heap->size && elt[right]->size < elt[smallest]->size) smallest = right;
        if (smallest != i) {
            SWAP(&elt[i], &elt[smallest]);
            elt[i]->heap_index = i;
            elt[smallest]->heap_index = smallest;
            i = smallest;
        }
        else break;
    } while (i < heap->size);
}

static STheap_t *STbuildheap(subtree_t **elt, size_t size) {
    STheap_t *heap = gv_alloc(sizeof(STheap_t));
    heap->elt = elt;
    heap->size = size;
    for (size_t i = 0; i < heap->size; i++) heap->elt[i]->heap_index = i;
    for (size_t i = heap->size / 2; i != SIZE_MAX; i--)
        STheapify(heap,i);
    return heap;
}

static
subtree_t *STextractmin(STheap_t *heap)
{
    subtree_t *rv = heap->elt[0];
    rv->heap_index = SIZE_MAX;
      // mark this as not participating in the heap anymore
    heap->elt[0] = heap->elt[heap->size - 1];
    heap->elt[0]->heap_index = 0;
    heap->elt[heap->size -1] = rv;    /* needed to free storage later */
    heap->size--;
    STheapify(heap,0);
    return rv;
}

static
void tree_adjust(Agnode_t *v, Agnode_t *from, int delta)
{
    Agedge_t *e;
    ND_rank(v) += delta;
    for (int i = 0; (e = ND_tree_in(v).list[i]); i++) {
      Agnode_t *const w = agtail(e);
      if (w != from)
        tree_adjust(w, v, delta);
    }
    for (int i = 0; (e = ND_tree_out(v).list[i]); i++) {
      Agnode_t *const w = aghead(e);
      if (w != from)
        tree_adjust(w, v, delta);
    }
}

static
subtree_t *merge_trees(network_simplex_ctx_t *ctx, Agedge_t *e)   /* entering tree edge */
{
  assert(!TREE_EDGE(e));

  subtree_t *const t0 = STsetFind(agtail(e));
  subtree_t *const t1 = STsetFind(aghead(e));

  if (!on_heap(t0)) { // move t0
    const int delta = SLACK(e);
    if (delta != 0)
      tree_adjust(t0->rep,NULL,delta);
  }
  else {  // move t1
    const int delta = -SLACK(e);
    if (delta != 0)
      tree_adjust(t1->rep,NULL,delta);
  }
  if (add_tree_edge(ctx, e) != 0) {
    return NULL;
  }
  return STsetUnion(t0, t1);
}

/* Construct initial tight tree. Graph must be connected, feasible.
 * Adjust ND_rank(v) as needed.  add_tree_edge() on tight tree edges.
 * trees are basically lists of nodes stored in `LIST(node_t *)`s.
 * Return 1 if input graph is not connected; 0 on success.
 */
static
int feasible_tree(network_simplex_ctx_t *ctx)
{
  Agedge_t *ee;
  size_t subtree_count = 0;
  STheap_t *heap = NULL;
  int error = 0;

  /* initialization */
  for (Agnode_t *n = GD_nlist(ctx->G); n != NULL; n = ND_next(n)) {
      ND_subtree_set(n,0);
  }

  subtree_t **tree = gv_calloc(ctx->N_nodes, sizeof(subtree_t *));
  /* given init_rank, find all tight subtrees */
  for (Agnode_t *n = GD_nlist(ctx->G); n != NULL; n = ND_next(n)) {
        if (ND_subtree(n) == 0) {
                tree[subtree_count] = find_tight_subtree(ctx, n);
                if (tree[subtree_count] == NULL) {
                    error = 2;
                    goto end;
                }
                subtree_count++;
        }
  }

  /* incrementally merge subtrees */
  heap = STbuildheap(tree,subtree_count);
  while (STheapsize(heap) > 1) {
    subtree_t *tree0 = STextractmin(heap);
    if (!(ee = inter_tree_edge(tree0))) {
      error = 1;
      break;
    }
    subtree_t *tree1 = merge_trees(ctx, ee);
    if (tree1 == NULL) {
      error = 2;
      break;
    }
    STheapify(heap,tree1->heap_index);
  }

end:
  free(heap);
  for (size_t i = 0; i < subtree_count; i++) free(tree[i]);
  free(tree);
  if (error) return error;
  assert(LIST_SIZE(&ctx->Tree_edge) == ctx->N_nodes - 1);
  init_cutvalues(ctx);
  return 0;
}

/* walk up from v to LCA(v,w), setting new cutvalues. */
static Agnode_t *treeupdate(Agnode_t *v, Agnode_t *w, int cutvalue, bool dir) {
    while (!SEQ(ND_low(v), ND_lim(w), ND_lim(v))) {
	edge_t *const e = ND_par(v);
	const bool d = v == agtail(e) ? dir : !dir;
	if (d)
	    ED_cutvalue(e) += cutvalue;
	else
	    ED_cutvalue(e) -= cutvalue;
	if (ND_lim(agtail(e)) > ND_lim(aghead(e)))
	    v = agtail(e);
	else
	    v = aghead(e);
    }
    return v;
}

static void rerank(Agnode_t * v, int delta)
{
    edge_t *e;

    ND_rank(v) -= delta;
    for (int i = 0; (e = ND_tree_out(v).list[i]); i++)
	if (e != ND_par(v))
	    rerank(aghead(e), delta);
    for (int i = 0; (e = ND_tree_in(v).list[i]); i++)
	if (e != ND_par(v))
	    rerank(agtail(e), delta);
}

/* e is the tree edge that is leaving and f is the nontree edge that
 * is entering.  compute new cut values, ranks, and exchange e and f.
 */
static int
update(network_simplex_ctx_t *ctx, edge_t * e, edge_t * f)
{
    const int delta = SLACK(f);
    /* "for (v = in nodes in tail side of e) do ND_rank(v) -= delta;" */
    if (delta > 0) {
	size_t s = ND_tree_in(agtail(e)).size + ND_tree_out(agtail(e)).size;
	if (s == 1)
	    rerank(agtail(e), delta);
	else {
	    s = ND_tree_in(aghead(e)).size + ND_tree_out(aghead(e)).size;
	    if (s == 1)
		rerank(aghead(e), -delta);
	    else {
		if (ND_lim(agtail(e)) < ND_lim(aghead(e)))
		    rerank(agtail(e), delta);
		else
		    rerank(aghead(e), -delta);
	    }
	}
    }

    const int cutvalue = ED_cutvalue(e);
    Agnode_t *const lca = treeupdate(agtail(f), aghead(f), cutvalue, true);
    if (treeupdate(aghead(f), agtail(f), cutvalue, false) != lca) {
	agerrorf("update: mismatched lca in treeupdates\n");
	return 2;
    }

    // invalidate paths from LCA till affected nodes:
    int lca_low = ND_low(lca);
    invalidate_path(lca, aghead(f));
    invalidate_path(lca, agtail(f));

    ED_cutvalue(f) = -cutvalue;
    ED_cutvalue(e) = 0;
    exchange_tree_edges(ctx, e, f);
    dfs_range(lca, ND_par(lca), lca_low);
    return 0;
}

static int scan_and_normalize(network_simplex_ctx_t *ctx) {
    int Minrank = INT_MAX;
    int Maxrank = INT_MIN;
    for (node_t *n = GD_nlist(ctx->G); n; n = ND_next(n)) {
	if (ND_node_type(n) == NORMAL) {
	    Minrank = MIN(Minrank, ND_rank(n));
	    Maxrank = MAX(Maxrank, ND_rank(n));
	}
    }
    for (node_t *n = GD_nlist(ctx->G); n; n = ND_next(n))
	ND_rank(n) -= Minrank;
    Maxrank -= Minrank;
    return Maxrank;
}

static void reset_lists(network_simplex_ctx_t *ctx) {
  LIST_FREE(&ctx->Tree_edge);
}

static void
freeTreeList (network_simplex_ctx_t *ctx, graph_t* g)
{
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	free_list(ND_tree_in(n));
	free_list(ND_tree_out(n));
	ND_mark(n) = false;
    }
    reset_lists(ctx);
}

static void LR_balance(network_simplex_ctx_t *ctx)
{
    for (size_t i = 0; i < LIST_SIZE(&ctx->Tree_edge); i++) {
	edge_t *const e = LIST_GET(&ctx->Tree_edge, i);
	if (ED_cutvalue(e) == 0) {
	    edge_t *const f = enter_edge(e);
	    if (f == NULL)
		continue;
	    const int delta = SLACK(f);
	    if (delta <= 1)
		continue;
	    if (ND_lim(agtail(e)) < ND_lim(aghead(e)))
		rerank(agtail(e), delta / 2);
	    else
		rerank(aghead(e), -delta / 2);
	}
    }
    freeTreeList(ctx, ctx->G);
}

static int decreasingrankcmpf(const void *x, const void *y) {
  node_t *const *n0 = x;
  node_t *const *n1 = y;
  if (ND_rank(*n1) < ND_rank(*n0)) {
    return -1;
  }
  if (ND_rank(*n1) > ND_rank(*n0)) {
    return 1;
  }
  return 0;
}

static int increasingrankcmpf(const void *x, const void *y) {
  return -decreasingrankcmpf(x, y);
}

static void TB_balance(network_simplex_ctx_t *ctx)
{
    edge_t *e;
    int adj = 0;
    char *s;

    const int Maxrank = scan_and_normalize(ctx);

    /* find nodes that are not tight and move to less populated ranks */
    assert(Maxrank >= 0);
    int *nrank = gv_calloc((size_t)Maxrank + 1, sizeof(int));
    if ( (s = agget(ctx->G,"TBbalance")) ) {
         if (streq(s,"min")) adj = 1;
         else if (streq(s,"max")) adj = 2;
         if (adj) for (node_t *n = GD_nlist(ctx->G); n; n = ND_next(n))
              if (ND_node_type(n) == NORMAL) {
                if (ND_in(n).size == 0 && adj == 1) {
                   ND_rank(n) = 0;
                }
                if (ND_out(n).size == 0 && adj == 2) {
                   ND_rank(n) = Maxrank;
                }
              }
    }
    LIST(node_t *) Tree_node = {0};
    LIST_RESERVE(&Tree_node, ctx->N_nodes);
    for (node_t *n = GD_nlist(ctx->G); n; n = ND_next(n)) {
      LIST_APPEND(&Tree_node, n);
    }
    LIST_SORT(&Tree_node, adj > 1 ? decreasingrankcmpf: increasingrankcmpf);
    for (size_t i = 0; i < LIST_SIZE(&Tree_node); i++) {
        node_t *const n = LIST_GET(&Tree_node, i);
        if (ND_node_type(n) == NORMAL)
          nrank[ND_rank(n)]++;
    }
    for (size_t ii = 0; ii < LIST_SIZE(&Tree_node); ii++) {
      node_t *const n = LIST_GET(&Tree_node, ii);
      if (ND_node_type(n) != NORMAL)
        continue;
      int inweight = 0;
      int outweight = 0;
      int low = 0;
      int high = Maxrank;
      for (size_t i = 0; (e = ND_in(n).list[i]); i++) {
        inweight += ED_weight(e);
        low = MAX(low, ND_rank(agtail(e)) + ED_minlen(e));
      }
      for (size_t i = 0; (e = ND_out(n).list[i]); i++) {
        outweight += ED_weight(e);
        high = MIN(high, ND_rank(aghead(e)) - ED_minlen(e));
      }
      if (low < 0)
        low = 0;		/* vnodes can have ranks < 0 */
      if (adj) {
        if (inweight == outweight)
            ND_rank(n) = adj == 1 ? low : high;
      }
      else {
                if (inweight == outweight) {
                    int choice = low;
                    for (int i = low + 1; i <= high; i++)
                        if (nrank[i] < nrank[choice])
                            choice = i;
                    nrank[ND_rank(n)]--;
                    nrank[choice]++;
                    ND_rank(n) = choice;
                }
      }
      free_list(ND_tree_in(n));
      free_list(ND_tree_out(n));
      ND_mark(n) = false;
    }
    LIST_FREE(&Tree_node);
    free(nrank);
}

static bool init_graph(network_simplex_ctx_t *ctx, graph_t *g) {
    edge_t *e;

    *ctx = (network_simplex_ctx_t){.G = g};
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	ND_mark(n) = false;
	ctx->N_nodes++;
	for (size_t i = 0; ND_out(n).list[i]; i++)
	    ctx->N_edges++;
    }

    LIST_RESERVE(&ctx->Tree_edge, ctx->N_nodes);

    bool feasible = true;
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	ND_priority(n) = 0;
	size_t i;
	for (i = 0; (e = ND_in(n).list[i]); i++) {
	    ND_priority(n)++;
	    ED_cutvalue(e) = 0;
	    ED_tree_index(e) = -1;
	    if (ND_rank(aghead(e)) - ND_rank(agtail(e)) < ED_minlen(e))
		feasible = false;
	}
	ND_tree_in(n).list = gv_calloc(i + 1, sizeof(edge_t *));
	ND_tree_in(n).size = 0;
	for (i = 0; ND_out(n).list[i]; i++);
	ND_tree_out(n).list = gv_calloc(i + 1, sizeof(edge_t *));
	ND_tree_out(n).size = 0;
    }
    return feasible;
}

/* graphSize:
 * Compute no. of nodes and edges in the graph
 */
static void graphSize(graph_t *g, size_t *nn, size_t *ne) {
    size_t nnodes = 0;
    size_t nedges = 0;
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	nnodes++;
	for (size_t i = 0; ND_out(n).list[i]; i++) {
	    nedges++;
	}
    }
    *nn = nnodes;
    *ne = nedges;
}

/* rank:
 * Apply network simplex to rank the nodes in a graph.
 * Uses ED_minlen as the internode constraint: if a->b with minlen=ml,
 * rank b - rank a >= ml.
 * Assumes the graph has the following additional structure:
 *   A list of all nodes, starting at GD_nlist, and linked using ND_next.
 *   Out and in edges lists stored in ND_out and ND_in, even if the node
 *  doesn't have any out or in edges.
 * The node rank values are stored in ND_rank.
 * Returns 0 if successful; returns 1 if the graph was not connected;
 * returns 2 if something seriously wrong;
 */
int rank2(graph_t * g, int balance, int maxiter, int search_size)
{
    int iter = 0;
    char ns[] = "network simplex: ";
    edge_t *e;
    network_simplex_ctx_t ctx = {0};

#ifdef DEBUG
    check_cycles(g);
#endif
    if (Verbose) {
	size_t nn, ne;
	graphSize (g, &nn, &ne);
	fprintf(stderr, "%s %" PRISIZE_T " nodes %" PRISIZE_T
	        " edges maxiter=%d balance=%d\n", ns, nn, ne, maxiter, balance);
	start_timer();
    }
    bool feasible = init_graph(&ctx, g);
    if (!feasible)
	init_rank(&ctx);

    if (search_size >= 0)
	ctx.Search_size = search_size;
    else
	ctx.Search_size = SEARCHSIZE;

    {
	const int err = feasible_tree(&ctx);
	if (err != 0) {
	    freeTreeList(&ctx, g);
	    return err;
	}
    }
    if (maxiter <= 0) {
	freeTreeList(&ctx, g);
	return 0;
    }

    while ((e = leave_edge(&ctx))) {
	edge_t *const f = enter_edge(e);
	const int err = update(&ctx, e, f);
	if (err != 0) {
	    freeTreeList(&ctx, g);
	    return err;
	}
	iter++;
	if (Verbose && iter % 100 == 0) {
	    if (iter % 1000 == 100)
		fputs(ns, stderr);
	    fprintf(stderr, "%d ", iter);
	    if (iter % 1000 == 0)
		fputc('\n', stderr);
	}
	if (iter >= maxiter)
	    break;
    }
    switch (balance) {
    case 1:
	TB_balance(&ctx);
	reset_lists(&ctx);
	break;
    case 2:
	LR_balance(&ctx);
	break;
    default:
	(void)scan_and_normalize(&ctx);
	freeTreeList (&ctx, ctx.G);
	break;
    }
    if (Verbose) {
	if (iter >= 100)
	    fputc('\n', stderr);
	fprintf(stderr, "%s%" PRISIZE_T " nodes %" PRISIZE_T " edges %d iter %.2f sec\n",
		ns, ctx.N_nodes, ctx.N_edges, iter, elapsed_sec());
    }
    return 0;
}

int rank(graph_t * g, int balance, int maxiter)
{
    char *s;
    int search_size;

    if ((s = agget(g, "searchsize")))
	search_size = atoi(s);
    else
	search_size = SEARCHSIZE;

    return rank2 (g, balance, maxiter, search_size);
}

/* set cut value of f, assuming values of edges on one side were already set */
static void x_cutval(edge_t * f)
{
    node_t *v;
    edge_t *e;
    int dir;

    /* set v to the node on the side of the edge already searched */
    if (ND_par(agtail(f)) == f) {
	v = agtail(f);
	dir = 1;
    } else {
	v = aghead(f);
	dir = -1;
    }

    int sum = 0;
    for (int i = 0; (e = ND_out(v).list[i]); i++)
	if (sadd_overflow(sum, x_val(e, v, dir), &sum)) {
	    agerrorf("overflow when computing edge weight sum\n");
	    graphviz_exit(EXIT_FAILURE);
	}
    for (int i = 0; (e = ND_in(v).list[i]); i++)
	if (sadd_overflow(sum, x_val(e, v, dir), &sum)) {
	    agerrorf("overflow when computing edge weight sum\n");
	    graphviz_exit(EXIT_FAILURE);
	}
    ED_cutvalue(f) = sum;
}

static int x_val(edge_t * e, node_t * v, int dir)
{
    node_t *other;
    int d, rv, f;

    if (agtail(e) == v)
	other = aghead(e);
    else
	other = agtail(e);
    if (!(SEQ(ND_low(v), ND_lim(other), ND_lim(v)))) {
	f = 1;
	rv = ED_weight(e);
    } else {
	f = 0;
	if (TREE_EDGE(e))
	    rv = ED_cutvalue(e);
	else
	    rv = 0;
	rv -= ED_weight(e);
    }
    if (dir > 0) {
	if (aghead(e) == v)
	    d = 1;
	else
	    d = -1;
    } else {
	if (agtail(e) == v)
	    d = 1;
	else
	    d = -1;
    }
    if (f)
	d = -d;
    if (d < 0)
	rv = -rv;
    return rv;
}

static void dfs_cutval(node_t * v, edge_t * par)
{
    // per-node state
    typedef struct {
	node_t *v;
	edge_t *par;
	int out_i; ///< counter for iterating through `ND_tree_out(v).list`
	int in_i;  ///< counter for iterating through `ND_tree_in(v).list`
    } state_t;

    LIST(state_t) todo = {0};
    LIST_PUSH_BACK(&todo, ((state_t){.v = v, .par = par}));

    while (!LIST_IS_EMPTY(&todo)) {
	state_t *const top = LIST_BACK(&todo);

	bool updated = false;
	edge_t *e;
	for (; (e = ND_tree_out(top->v).list[top->out_i]); ++top->out_i) {
	    if (e != top->par) {
	      ++top->out_i;
	      LIST_PUSH_BACK(&todo, ((state_t){.v = aghead(e), .par = e}));
	      updated = true;
	      break;
	    }
	}
	if (updated) {
	    continue;
	}

	for (; (e = ND_tree_in(top->v).list[top->in_i]); ++top->in_i) {
	    if (e != top->par) {
	      ++top->in_i;
	      LIST_PUSH_BACK(&todo, ((state_t){.v = agtail(e), .par = e}));
	      updated = true;
	      break;
	    }
	}
	if (updated) {
	    continue;
	}

	if (top->par) {
	    x_cutval(top->par);
	}
	LIST_DROP_BACK(&todo);
    }

    LIST_FREE(&todo);
}

/// local state used by `dfs_range*`
typedef struct {
  node_t *v;
  edge_t *par;
  int lim;
  int tree_out_i;
  int tree_in_i;
} dfs_state_t;

/*
* Initializes DFS range attributes (par, low, lim) over tree nodes such that:
* ND_par(n) - parent tree edge
* ND_low(n) - min DFS index for nodes in sub-tree (>= 1)
* ND_lim(n) - max DFS index for nodes in sub-tree
*/
static int dfs_range_init(node_t *v) {
    int lim = 0;

    LIST(dfs_state_t) todo = {0};

    ND_par(v) = NULL;
    ND_low(v) = 1;
    const dfs_state_t root = {.v = v, .par = NULL, .lim = 1};
    LIST_PUSH_BACK(&todo, root);

    while (!LIST_IS_EMPTY(&todo)) {
        bool pushed_new = false;
        dfs_state_t *const s = LIST_BACK(&todo);

        while (ND_tree_out(s->v).list[s->tree_out_i]) {
            edge_t *const e = ND_tree_out(s->v).list[s->tree_out_i];
            ++s->tree_out_i;
            if (e != s->par) {
                node_t *const n = aghead(e);
                ND_par(n) = e;
                ND_low(n) = s->lim;
                const dfs_state_t next = {.v = n, .par = e, .lim = s->lim};
                LIST_PUSH_BACK(&todo, next);
                pushed_new = true;
                break;
            }
        }
        if (pushed_new) {
            continue;
        }

        while (ND_tree_in(s->v).list[s->tree_in_i]) {
            edge_t *const e = ND_tree_in(s->v).list[s->tree_in_i];
            ++s->tree_in_i;
            if (e != s->par) {
                node_t *const n = agtail(e);
                ND_par(n) = e;
                ND_low(n) = s->lim;
                const dfs_state_t next = {.v = n, .par = e, .lim = s->lim};
                LIST_PUSH_BACK(&todo, next);
                pushed_new = true;
                break;
            }
        }
        if (pushed_new) {
            continue;
        }

        ND_lim(s->v) = s->lim;

        lim = s->lim;
        LIST_DROP_BACK(&todo);

        if (!LIST_IS_EMPTY(&todo)) {
            LIST_BACK(&todo)->lim = lim + 1;
        }
    }

    LIST_FREE(&todo);

    return lim + 1;
}

/*
 * Incrementally updates DFS range attributes
 */
static int dfs_range(node_t * v, edge_t * par, int low)
{
    int lim = 0;

    if (ND_par(v) == par && ND_low(v) == low) {
	return ND_lim(v) + 1;
    }

    LIST(dfs_state_t) todo = {0};

    ND_par(v) = par;
    ND_low(v) = low;
    const dfs_state_t root = {.v = v, .par = par, .lim = low};
    LIST_PUSH_BACK(&todo, root);

    while (!LIST_IS_EMPTY(&todo)) {
	bool processed_child = false;
	dfs_state_t *const s = LIST_BACK(&todo);

	while (ND_tree_out(s->v).list[s->tree_out_i]) {
	    edge_t *const e = ND_tree_out(s->v).list[s->tree_out_i];
	    ++s->tree_out_i;
	    if (e != s->par) {
		node_t *const n = aghead(e);
		if (ND_par(n) == e && ND_low(n) == s->lim) {
		    s->lim = ND_lim(n) + 1;
		} else {
		    ND_par(n) = e;
		    ND_low(n) = s->lim;
		    const dfs_state_t next = {.v = n, .par = e, .lim = s->lim};
		    LIST_PUSH_BACK(&todo, next);
		}
		processed_child = true;
		break;
	    }
	}
	if (processed_child) {
	    continue;
	}

	while (ND_tree_in(s->v).list[s->tree_in_i]) {
	    edge_t *const e = ND_tree_in(s->v).list[s->tree_in_i];
	    ++s->tree_in_i;
	    if (e != s->par) {
		node_t *const n = agtail(e);
		if (ND_par(n) == e && ND_low(n) == s->lim) {
		    s->lim = ND_lim(n) + 1;
		} else {
		    ND_par(n) = e;
		    ND_low(n) = s->lim;
		    const dfs_state_t next = {.v = n, .par = e, .lim = s->lim};
		    LIST_PUSH_BACK(&todo, next);
		}
		processed_child = true;
		break;
	    }
	}
	if (processed_child) {
	    continue;
	}

	ND_lim(s->v) = s->lim;

	lim = s->lim;
	LIST_DROP_BACK(&todo);

	if (!LIST_IS_EMPTY(&todo)) {
	    LIST_BACK(&todo)->lim = lim + 1;
	}
    }

    LIST_FREE(&todo);

    return lim + 1;
}

#ifdef DEBUG
void tchk(network_simplex_ctx_t *ctx)
{
    edge_t *e;

    size_t n_cnt = 0;
    size_t e_cnt = 0;
    for (node_t *n = agfstnode(ctx->G); n; n = agnxtnode(ctx->G, n)) {
	n_cnt++;
	for (int i = 0; (e = ND_tree_out(n).list[i]); i++) {
	    e_cnt++;
	    if (SLACK(e) > 0)
		fprintf(stderr, "not a tight tree %p", e);
	}
    }
    if (e_cnt != LIST_SIZE(&ctx->Tree_edge))
	fprintf(stderr, "something missing\n");
}

static void dump_node(FILE *sink, node_t *n) {
    if (ND_node_type(n)) {
      fprintf(sink, "%p", n);
    }
    else
      fputs(agnameof(n), sink);
}

static void dump_graph (graph_t* g)
{
    edge_t *e;
    FILE *const fp = fopen ("ns.gv", "w");
    fprintf (fp, "digraph \"%s\" {\n", agnameof(g));
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	fputs("  \"", fp);
	dump_node(fp, n);
	fputs("\"\n", fp);
    }
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	for (int i = 0; (e = ND_out(n).list[i]); i++) {
	    fputs("  \"", fp);
	    dump_node(fp, n);
	    fputs("\"", fp);
	    node_t *const w = aghead(e);
	    fputs(" -> \"", fp);
	    dump_node(fp, w);
	    fputs("\"\n", fp);
	}
    }

    fprintf (fp, "}\n");
    fclose (fp);
}

static node_t *checkdfs(graph_t* g, node_t * n)
{
    edge_t *e;

    if (ND_mark(n))
	return NULL;
    ND_mark(n) = true;
    ND_onstack(n) = true;
    for (int i = 0; (e = ND_out(n).list[i]); i++) {
	node_t *const w = aghead(e);
	if (ND_onstack(w)) {
	    dump_graph (g);
	    fprintf(stderr, "cycle: last edge %p %s(%p) %s(%p)\n", e, agnameof(n), n,
	            agnameof(w), w);
	    return w;
	}
	else {
	    if (!ND_mark(w)) {
		node_t *const x = checkdfs(g, w);
		if (x) {
		    fprintf(stderr,"unwind %p %s(%p)\n", e, agnameof(n), n);
		    if (x != n) return x;
		    fprintf(stderr,"unwound to root\n");
		    fflush(stderr);
		    abort();
		    return NULL;
		}
	    }
	}
    }
    ND_onstack(n) = false;
    return NULL;
}

void check_cycles(graph_t * g)
{
    for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
	ND_mark(n) = false;
	ND_onstack(n) = false;
    }
    for (node_t *n = GD_nlist(g); n; n = ND_next(n))
	checkdfs(g, n);
}
#endif				/* DEBUG */
