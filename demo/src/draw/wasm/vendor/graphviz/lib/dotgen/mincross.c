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
 * dot_mincross(g) takes a ranked graphs, and finds an ordering
 * that avoids edge crossings.  clusters are expanded.
 * N.B. the rank structure is global (not allocated per cluster)
 * because mincross may compare nodes in different clusters.
 */

#include "config.h"

#include <assert.h>
#include <cgraph/cgraph.h>
#include <dotgen/dot.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <util/alloc.h>
#include <util/bitarray.h>
#include <util/exit.h>
#include <util/gv_math.h>
#include <util/itos.h>
#include <util/list.h>
#include <util/streq.h>

struct adjmatrix_t {
  size_t nrows;  ///< how many rows have been allocated?
  size_t ncols;  ///< how many columns have been allocated?
  uint8_t *data; ///< bit-packed backing memory
};

/// get the value of a matrix cell
///
/// @param me Matrix to inspect
/// @param row Row coordinate
/// @param col Column coordinate
/// @return True if the given cell was set
static bool matrix_get(adjmatrix_t *me, size_t row, size_t col) {
  assert(me != NULL);

  // if this index is beyond anything allocated, infer it as unset
  if (row >= me->nrows) {
    return false;
  }
  if (col >= me->ncols) {
    return false;
  }

  const size_t index = row * me->ncols + col;
  const size_t byte_index = index / 8;
  const size_t bit_index = index % 8;
  return (me->data[byte_index] >> bit_index) & 1;
}

/// set the value of a matrix cell to true
///
/// @param me Matrix to update
/// @param row Row coordinate
/// @param col Column coordinate
static void matrix_set(adjmatrix_t *me, size_t row, size_t col) {
  assert(me != NULL);

  // if we are updating beyond allocated space, expand the backing store
  if (row >= me->nrows || col >= me->ncols) {
    // allocate an enlarged space
    const size_t nrows = zmax(me->nrows, row + 1);
    const size_t ncols = zmax(me->ncols, col + 1);
    const size_t bits = nrows * ncols;
    const size_t bytes = bits / 8 + (bits % 8 == 0 ? 0 : 1);
    uint8_t *const data = gv_alloc(bytes);

    // replicate set bits
    for (size_t r = 0; r < me->nrows; ++r) {
      for (size_t c = 0; c < me->ncols; ++c) {
        if (!matrix_get(me, r, c)) {
          continue;
        }
        const size_t index = r * ncols + c;
        const size_t byte_index = index / 8;
        const size_t bit_index = index % 8;
        data[byte_index] |= (uint8_t)(UINT8_C(1) << bit_index);
      }
    }

    // replace old matrix with newly expanded one
    free(me->data);
    *me = (adjmatrix_t){.nrows = nrows, .ncols = ncols, .data = data};
  }

  assert(row < me->nrows);
  assert(col < me->ncols);

  const size_t index = row * me->ncols + col;
  const size_t byte_index = index / 8;
  const size_t bit_index = index % 8;
  me->data[byte_index] |= (uint8_t)(UINT8_C(1) << bit_index);
}

/* #define DEBUG */
#define MARK(v)		(ND_mark(v))
#define saveorder(v)	(ND_coord(v)).x
#define flatindex(v)	((size_t)ND_low(v))

	/* forward declarations */
static bool medians(graph_t * g, int r0, int r1);
static int nodeposcmpf(const void *, const void *);
static int edgeidcmpf(const void *, const void *);
static void flat_breakcycles(graph_t * g);
static void flat_reorder(graph_t * g);
static void flat_search(graph_t * g, node_t * v);
static void init_mincross(graph_t * g);
static void merge2(graph_t * g);
static void init_mccomp(graph_t *g, size_t c);
static void cleanup2(graph_t *g, int64_t nc);
/// @return minimum crossings on success, negative value on failure
static int64_t mincross_clust(graph_t *g);
/// @return minimum crossings on success, negative value on failure
static int64_t mincross(graph_t *g, int startpass);
static void mincross_step(graph_t * g, int pass);
static void mincross_options(graph_t * g);
static void save_best(graph_t * g);
static void restore_best(graph_t * g);

/// create an adjacency matrix
///
/// These matrices dynamically expand on demand. So the initial dimensions
/// provided can be later safely exceeded at runtime.
///
/// @param initial_rows How many rows to allocate now
/// @param initial_columns How many columns to allocate now
/// @return A constructed matrix
static adjmatrix_t *new_matrix(size_t initial_rows, size_t initial_columns);

static void free_matrix(adjmatrix_t * p);
static int ordercmpf(const void *, const void *);
static int64_t ncross(void);
#ifdef DEBUG
void check_order(void);
void check_vlists(graph_t * g);
void node_in_root_vlist(node_t * n);
#endif


	/* mincross parameters */
static int MinQuit;
static const double Convergence = .995;

static graph_t *Root;
static int GlobalMinRank, GlobalMaxRank;
static edge_t **TE_list;
static int *TI_list;
static bool ReMincross;

typedef struct {
    Agrec_t h;
    int x, lo, hi;
    Agnode_t* np;
} info_t;

#define ND_x(n) (((info_t*)AGDATA(n))->x)
#define ND_lo(n) (((info_t*)AGDATA(n))->lo)
#define ND_hi(n) (((info_t*)AGDATA(n))->hi)
#define ND_np(n) (((info_t*)AGDATA(n))->np)
#define ND_idx(n) (ND_order(ND_np(n)))

static void
emptyComp (graph_t* sg)
{
    Agnode_t* n;
    Agnode_t* nxt;

    for (n = agfstnode(sg); n; n = nxt) {
	nxt = agnxtnode (sg, n);
	agdelnode(sg,n);
    }
}

#define isBackedge(e) (ND_idx(aghead(e)) > ND_idx(agtail(e)))

static Agnode_t*
findSource (Agraph_t* g, Agraph_t* sg)
{
    Agnode_t* n;

    for (n = agfstnode(sg); n; n = agnxtnode(sg, n))
	if (agdegree(g,n,1,0) == 0) return n;
    return NULL;
}

typedef LIST(node_t *) nodes_t;

static nodes_t topsort(Agraph_t *g, Agraph_t *sg) {
    Agnode_t* n;
    Agedge_t* e;
    Agedge_t* nxte;
    nodes_t arr = {0};

    while ((n = findSource(g, sg))) {
	LIST_APPEND(&arr, ND_np(n)); 
	agdelnode(sg, n);
	for (e = agfstout(g, n); e; e = nxte) {
	    nxte = agnxtout(g, e); 
	    agdeledge(g, e);
	}
    }
    return arr;
}

static int
getComp (graph_t* g, node_t* n, graph_t* comp, int* indices)
{
    int backedge = 0;
    Agedge_t* e;

    ND_x(n) = 1;
    indices[agnnodes(comp)] = ND_idx(n);
    agsubnode(comp, n, 1);
    for (e = agfstout(g,n); e; e = agnxtout(g,e)) {
	if (isBackedge(e)) backedge++;
	if (!ND_x(aghead(e)))
	    backedge += getComp(g, aghead(e), comp, indices);
    }
    for (e = agfstin(g,n); e; e = agnxtin(g,e)) {
	if (isBackedge(e)) backedge++;
	if (!ND_x(agtail(e)))
	    backedge += getComp(g, agtail(e), comp, indices);
    }
    return backedge;
}

/// for each pair of nodes (labels), we add an edge 
static void
fixLabelOrder (graph_t* g, rank_t* rk)
{
    bool haveBackedge = false;
    Agraph_t* sg;
    Agnode_t* n;
    Agnode_t* nxtp;
    Agnode_t* v;

    for (n = agfstnode(g); n; n = nxtp) {
	v = nxtp = agnxtnode(g, n);
	for (; v; v = agnxtnode(g, v)) {
	    if (ND_hi(v) <= ND_lo(n)) { 
		haveBackedge = true;
		agedge(g, v, n, NULL, 1);
	    }
	    else if (ND_hi(n) <= ND_lo(v)) { 
		agedge(g, n, v, NULL, 1);
	    }
	}
    }
    if (!haveBackedge) return;
    
    sg = agsubg(g, "comp", 1);
    int *indices = gv_calloc(agnnodes(g), sizeof(int));

    for (n = agfstnode(g); n; n = agnxtnode(g,n)) {
	if (ND_x(n) || agdegree(g,n,1,1) == 0) continue;
	if (getComp(g, n, sg, indices)) {
	    nodes_t arr = topsort(g, sg);
	    assert(LIST_SIZE(&arr) == (size_t)agnnodes(sg));
	    qsort(indices, LIST_SIZE(&arr), sizeof(int), ordercmpf);
	    for (size_t i = 0; i < LIST_SIZE(&arr); i++) {
		ND_order(LIST_GET(&arr, i)) = indices[i];
		rk->v[indices[i]] = LIST_GET(&arr, i);
	    }
	    LIST_FREE(&arr);
       }
       emptyComp(sg);
    }
    free(indices);
}

/* Check that the ordering of labels for flat edges is consistent. 
 * This is necessary because dot_position will attempt to force the label
 * to be between the edge's vertices. This can lead to an infeasible problem.
 *
 * We check each rank for any flat edge labels (as dummy nodes) and create a
 * graph with a node for each label. If the graph contains more than 1 node, we
 * call fixLabelOrder to see if there really is a problem and, if so, fix it.
 */ 
void
checkLabelOrder (graph_t* g)
{
    graph_t* lg = NULL;

    for (int r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	rank_t *const rk = GD_rank(g)+r;
	for (int j = 0; j < rk->n; j++) {
	    Agnode_t *const u = rk->v[j];
	    if (ND_alg(u)) {
		if (!lg) lg = agopen ("lg", Agstrictdirected, 0);
		Agnode_t *const n = agnode(lg, ITOS(j), 1);
		agbindrec(n, "info", sizeof(info_t), true);
		int lo = ND_order(aghead(ND_out(u).list[0]));
		int hi = ND_order(aghead(ND_out(u).list[1]));
		if (lo > hi) {
		    SWAP(&lo, &hi);
		}
		ND_lo(n) = lo;
		ND_hi(n) = hi;
		ND_np(n) = u;
	    }
	}
	if (lg) {
	    if (agnnodes(lg) > 1) fixLabelOrder (lg, rk);
	    agclose(lg);
	    lg = NULL;
	}
    }
}

/* Minimize edge crossings
 * Note that nodes are not placed into GD_rank(g) until mincross()
 * is called.
 */
int dot_mincross(graph_t *g) {
    int64_t nc;
    char *s;

    /* check whether malformed input has led to empty cluster that the crossing
     * functions will not anticipate
     */
    {
	size_t i;
	for (i = 1; i <= (size_t)GD_n_cluster(g); ) {
	    if (agfstnode(GD_clust(g)[i]) == NULL) {
	      agwarningf("removing empty cluster\n");
	      memmove(&GD_clust(g)[i], &GD_clust(g)[i + 1],
	        ((size_t)GD_n_cluster(g) - i) * sizeof(GD_clust(g)[0]));
	      --GD_n_cluster(g);
	    } else {
	      ++i;
	    }
	}
    }

    init_mincross(g);

    size_t comp;
    for (nc = 0, comp = 0; comp < GD_comp(g).size; comp++) {
	init_mccomp(g, comp);
	const int64_t mc = mincross(g, 0);
	if (mc < 0) {
	    return -1;
	}
	nc += mc;
    }

    merge2(g);

    /* run mincross on contents of each cluster */
    for (int c = 1; c <= GD_n_cluster(g); c++) {
	const int64_t mc = mincross_clust(GD_clust(g)[c]);
	if (mc < 0) {
	    return -1;
	}
	nc += mc;
#ifdef DEBUG
	check_vlists(GD_clust(g)[c]);
	check_order();
#endif
    }

    if (GD_n_cluster(g) > 0 && (!(s = agget(g, "remincross")) || mapbool(s))) {
	mark_lowclusters(g);
	ReMincross = true;
	const int64_t mc = mincross(g, 2);
	if (mc < 0) {
	    return -1;
	}
	nc = mc;
#ifdef DEBUG
	for (int c = 1; c <= GD_n_cluster(g); c++)
	    check_vlists(GD_clust(g)[c]);
#endif
    }
    cleanup2(g, nc);
    return 0;
}

static adjmatrix_t *new_matrix(size_t initial_rows, size_t initial_columns) {
    adjmatrix_t *rv = gv_alloc(sizeof(adjmatrix_t));
    const size_t bits = initial_rows * initial_columns;
    const size_t bytes = bits / 8 + (bits % 8 == 0 ? 0 : 1);
    uint8_t *const data = gv_alloc(bytes);
    *rv = (adjmatrix_t){.nrows = initial_rows, .ncols = initial_columns, .data = data};
    return rv;
}

static void free_matrix(adjmatrix_t * p)
{
    if (p) {
	free(p->data);
	free(p);
    }
}

static void init_mccomp(graph_t *g, size_t c) {
    int r;

    GD_nlist(g) = GD_comp(g).list[c];
    if (c > 0) {
	for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	    GD_rank(g)[r].v = GD_rank(g)[r].v + GD_rank(g)[r].n;
	    GD_rank(g)[r].n = 0;
	}
    }
}

static int betweenclust(edge_t * e)
{
    while (ED_to_orig(e))
	e = ED_to_orig(e);
    return ND_clust(agtail(e)) != ND_clust(aghead(e));
}

static void do_ordering_node(graph_t *g, node_t *n, bool outflag) {
    int i, ne;
    node_t *u, *v;
    edge_t *e, *f, *fe;
    edge_t **sortlist = TE_list;

    if (ND_clust(n))
	return;
    if (outflag) {
	for (i = ne = 0; (e = ND_out(n).list[i]); i++)
	    if (!betweenclust(e))
		sortlist[ne++] = e;
    } else {
	for (i = ne = 0; (e = ND_in(n).list[i]); i++)
	    if (!betweenclust(e))
		sortlist[ne++] = e;
    }
    if (ne <= 1)
	return;
    // Write null terminator at end of list. Requires +1 in TE_list allocation.
    sortlist[ne] = 0;
    qsort(sortlist, ne, sizeof(sortlist[0]), edgeidcmpf);
    for (ne = 1; (f = sortlist[ne]); ne++) {
	e = sortlist[ne - 1];
	if (outflag) {
	    u = aghead(e);
	    v = aghead(f);
	} else {
	    u = agtail(e);
	    v = agtail(f);
	}
	if (find_flat_edge(u, v))
	    return;
	fe = new_virtual_edge(u, v, NULL);
	ED_edge_type(fe) = FLATORDER;
	flat_edge(g, fe);
    }
}

static void do_ordering(graph_t *g, bool outflag) {
    /* Order all nodes in graph */
    node_t *n;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	do_ordering_node (g, n, outflag);
    }
}

static void do_ordering_for_nodes(graph_t * g)
{
    /* Order nodes which have the "ordered" attribute */
    node_t *n;
    const char *ordering;

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if ((ordering = late_string(n, N_ordering, NULL))) {
	    if (streq(ordering, "out"))
		do_ordering_node(g, n, true);
	    else if (streq(ordering, "in"))
		do_ordering_node(g, n, false);
	    else if (ordering[0])
		agerrorf("ordering '%s' not recognized for node '%s'.\n", ordering, agnameof(n));
	}
    }
}

/* handle case where graph specifies edge ordering
 * If the graph does not have an ordering attribute, we then
 * check for nodes having the attribute.
 * Note that, in this implementation, the value of G_ordering
 * dominates the value of N_ordering.
 */
static void ordered_edges(graph_t * g)
{
    char *ordering;

    if (!G_ordering && !N_ordering)
	return;
    if ((ordering = late_string(g, G_ordering, NULL))) {
	if (streq(ordering, "out"))
	    do_ordering(g, true);
	else if (streq(ordering, "in"))
	    do_ordering(g, false);
	else if (ordering[0])
	    agerrorf("ordering '%s' not recognized.\n", ordering);
    }
    else
    {
	graph_t *subg;

	for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	    /* clusters are processed by separate calls to ordered_edges */
	    if (!is_cluster(subg))
		ordered_edges(subg);
	}
	if (N_ordering) do_ordering_for_nodes (g);
    }
}

static int64_t mincross_clust(graph_t *g) {
    int c;

    if (expand_cluster(g) != 0) {
	return -1;
    }
    ordered_edges(g);
    flat_breakcycles(g);
    flat_reorder(g);
    int64_t nc = mincross(g, 2);
    if (nc < 0) {
	return nc;
    }

    for (c = 1; c <= GD_n_cluster(g); c++) {
	const int64_t mc = mincross_clust(GD_clust(g)[c]);
	if (mc < 0) {
	    return mc;
	}
	nc += mc;
    }

    save_vlist(g);
    return nc;
}

static bool left2right(graph_t *g, node_t *v, node_t *w) {
    /* CLUSTER indicates orig nodes of clusters, and vnodes of skeletons */
    if (!ReMincross) {
	if (ND_clust(v) != ND_clust(w) && ND_clust(v) && ND_clust(w)) {
	    /* the following allows cluster skeletons to be swapped */
	    if (ND_ranktype(v) == CLUSTER && ND_node_type(v) == VIRTUAL)
		return false;
	    if (ND_ranktype(w) == CLUSTER && ND_node_type(w) == VIRTUAL)
		return false;
	    return true;
	}
    } else {
	if (ND_clust(v) != ND_clust(w))
	    return true;
    }
    adjmatrix_t *const M = GD_rank(g)[ND_rank(v)].flat;
    if (M == NULL)
	return false;
    if (GD_flip(g)) {
	SWAP(&v, &w);
    }
    return matrix_get(M, (size_t)flatindex(v), (size_t)flatindex(w));
}

static int64_t in_cross(node_t *v, node_t *w) {
    edge_t **e1, **e2;
    int inv, t;
    int64_t cross = 0;

    for (e2 = ND_in(w).list; *e2; e2++) {
	int cnt = ED_xpenalty(*e2);		
		
	inv = ND_order(agtail(*e2));

	for (e1 = ND_in(v).list; *e1; e1++) {
	    t = ND_order(agtail(*e1)) - inv;
	    if (t > 0 || (t == 0 && ED_tail_port(*e1).p.x > ED_tail_port(*e2).p.x))
		cross += ED_xpenalty(*e1) * cnt;
	}
    }
    return cross;
}

static int out_cross(node_t * v, node_t * w)
{
    edge_t **e1, **e2;
    int inv, cross = 0, t;

    for (e2 = ND_out(w).list; *e2; e2++) {
	int cnt = ED_xpenalty(*e2);
	inv = ND_order(aghead(*e2));

	for (e1 = ND_out(v).list; *e1; e1++) {
	    t = ND_order(aghead(*e1)) - inv;
	    if (t > 0 || (t == 0 && ED_head_port(*e1).p.x > ED_head_port(*e2).p.x))
		cross += ED_xpenalty(*e1) * cnt;
	}
    }
    return cross;

}

static void exchange(node_t * v, node_t * w)
{
    int vi, wi, r;

    r = ND_rank(v);
    vi = ND_order(v);
    wi = ND_order(w);
    ND_order(v) = wi;
    GD_rank(Root)[r].v[wi] = v;
    ND_order(w) = vi;
    GD_rank(Root)[r].v[vi] = w;
}

static int64_t transpose_step(graph_t *g, int r, bool reverse) {
    int i;
    node_t *v, *w;

    int64_t rv = 0;
    GD_rank(g)[r].candidate = false;
    for (i = 0; i < GD_rank(g)[r].n - 1; i++) {
	v = GD_rank(g)[r].v[i];
	w = GD_rank(g)[r].v[i + 1];
	assert(ND_order(v) < ND_order(w));
	if (left2right(g, v, w))
	    continue;
	int64_t c0 = 0;
	int64_t c1 = 0;
	if (r > 0) {
	    c0 += in_cross(v, w);
	    c1 += in_cross(w, v);
	}
	if (GD_rank(g)[r + 1].n > 0) {
	    c0 += out_cross(v, w);
	    c1 += out_cross(w, v);
	}
	if (c1 < c0 || (c0 > 0 && reverse && c1 == c0)) {
	    exchange(v, w);
	    rv += c0 - c1;
	    GD_rank(Root)[r].valid = false;
	    GD_rank(g)[r].candidate = true;

	    if (r > GD_minrank(g)) {
		GD_rank(Root)[r - 1].valid = false;
		GD_rank(g)[r - 1].candidate = true;
	    }
	    if (r < GD_maxrank(g)) {
		GD_rank(Root)[r + 1].valid = false;
		GD_rank(g)[r + 1].candidate = true;
	    }
	}
    }
    return rv;
}

static void transpose(graph_t * g, bool reverse)
{
    int r;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++)
	GD_rank(g)[r].candidate = true;
    int64_t delta;
    do {
	delta = 0;
	for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	    if (GD_rank(g)[r].candidate) {
		delta += transpose_step(g, r, reverse);
	    }
	}
    } while (delta >= 1);
}

static int64_t mincross(graph_t *g, int startpass) {
    const int endpass = 2;
    int maxthispass = 0, iter, trying, pass;
    int64_t cur_cross, best_cross;

    if (startpass > 1) {
	cur_cross = best_cross = ncross();
	save_best(g);
    } else
	cur_cross = best_cross = INT64_MAX;
    for (pass = startpass; pass <= endpass; pass++) {
	if (pass <= 1) {
	    maxthispass = MIN(4, MaxIter);
	    if (g == dot_root(g))
		if (build_ranks(g, pass) != 0) {
		    return -1;
		}
	    if (pass == 0)
		flat_breakcycles(g);
	    flat_reorder(g);

	    if ((cur_cross = ncross()) <= best_cross) {
		save_best(g);
		best_cross = cur_cross;
	    }
	} else {
	    maxthispass = MaxIter;
	    if (cur_cross > best_cross)
		restore_best(g);
	    cur_cross = best_cross;
	}
	trying = 0;
	for (iter = 0; iter < maxthispass; iter++) {
	    if (Verbose)
		fprintf(stderr,
			"mincross: pass %d iter %d trying %d cur_cross %" PRId64 " best_cross %"
			PRId64 "\n",
			pass, iter, trying, cur_cross, best_cross);
	    if (trying++ >= MinQuit)
		break;
	    if (cur_cross == 0)
		break;
	    mincross_step(g, iter);
	    if ((cur_cross = ncross()) <= best_cross) {
		save_best(g);
		if (cur_cross < Convergence * (double)best_cross)
		    trying = 0;
		best_cross = cur_cross;
	    }
	}
	if (cur_cross == 0)
	    break;
    }
    if (cur_cross > best_cross)
	restore_best(g);
    if (best_cross > 0) {
	transpose(g, false);
	best_cross = ncross();
    }

    return best_cross;
}

static void restore_best(graph_t * g)
{
    node_t *n;
    int i, r;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    n = GD_rank(g)[r].v[i];
	    ND_order(n) = saveorder(n);
	}
    }
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	GD_rank(Root)[r].valid = false;
	qsort(GD_rank(g)[r].v, GD_rank(g)[r].n, sizeof(GD_rank(g)[0].v[0]),
	      nodeposcmpf);
    }
}

static void save_best(graph_t * g)
{
    node_t *n;
    int i, r;
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    n = GD_rank(g)[r].v[i];
	    saveorder(n) = ND_order(n);
	}
    }
}

/* merges the connected components of g */
static void merge_components(graph_t * g)
{
    node_t *u, *v;

    if (GD_comp(g).size <= 1)
	return;
    u = NULL;
    for (size_t c = 0; c < GD_comp(g).size; c++) {
	v = GD_comp(g).list[c];
	if (u)
	    ND_next(u) = v;
	ND_prev(v) = u;
	while (ND_next(v)) {
	    v = ND_next(v);
	}
	u = v;
    }
    GD_comp(g).size = 1;
    GD_nlist(g) = GD_comp(g).list[0];
    GD_minrank(g) = GlobalMinRank;
    GD_maxrank(g) = GlobalMaxRank;
}

/* merge connected components, create globally consistent rank lists */
static void merge2(graph_t * g)
{
    int i, r;
    node_t *v;

    /* merge the components and rank limits */
    merge_components(g);

    /* install complete ranks */
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	GD_rank(g)[r].n = GD_rank(g)[r].an;
	GD_rank(g)[r].v = GD_rank(g)[r].av;
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    v = GD_rank(g)[r].v[i];
	    if (v == NULL) {
		if (Verbose)
		    fprintf(stderr,
			    "merge2: graph %s, rank %d has only %d < %d nodes\n",
			    agnameof(g), r, i, GD_rank(g)[r].n);
		GD_rank(g)[r].n = i;
		break;
	    }
	    ND_order(v) = i;
	}
    }
}

static void cleanup2(graph_t *g, int64_t nc) {
    int i, j, r, c;
    node_t *v;
    edge_t *e;

    if (TI_list) {
	free(TI_list);
	TI_list = NULL;
    }
    if (TE_list) {
	free(TE_list);
	TE_list = NULL;
    }
    /* fix vlists of clusters */
    for (c = 1; c <= GD_n_cluster(g); c++)
	rec_reset_vlists(GD_clust(g)[c]);

    /* remove node temporary edges for ordering nodes */
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    v = GD_rank(g)[r].v[i];
	    ND_order(v) = i;
	    if (ND_flat_out(v).list) {
		for (j = 0; (e = ND_flat_out(v).list[j]); j++)
		    if (ED_edge_type(e) == FLATORDER) {
			delete_flat_edge(e);
			free(e->base.data);
			free(e);
			j--;
		    }
	    }
	}
	free_matrix(GD_rank(g)[r].flat);
    }
    if (Verbose)
	fprintf(stderr, "mincross %s: %" PRId64 " crossings, %.2f secs.\n",
		agnameof(g), nc, elapsed_sec());
}

static node_t *neighbor(node_t * v, int dir)
{
    node_t *rv = NULL;
assert(v);
    if (dir < 0) {
	if (ND_order(v) > 0)
	    rv = GD_rank(Root)[ND_rank(v)].v[ND_order(v) - 1];
    } else
	rv = GD_rank(Root)[ND_rank(v)].v[ND_order(v) + 1];
assert(rv == 0 || (ND_order(rv)-ND_order(v))*dir > 0);
    return rv;
}

static bool is_a_normal_node_of(graph_t *g, node_t *v) {
    return ND_node_type(v) == NORMAL && agcontains(g, v);
}

static bool is_a_vnode_of_an_edge_of(graph_t *g, node_t *v) {
    if (ND_node_type(v) == VIRTUAL
	&& ND_in(v).size == 1 && ND_out(v).size == 1) {
	edge_t *e = ND_out(v).list[0];
	while (ED_edge_type(e) != NORMAL)
	    e = ED_to_orig(e);
	if (agcontains(g, e))
	    return true;
    }
    return false;
}

static bool inside_cluster(graph_t *g, node_t *v) {
  return is_a_normal_node_of(g, v) || is_a_vnode_of_an_edge_of(g, v);
}

static node_t *furthestnode(graph_t * g, node_t * v, int dir)
{
    node_t *rv = v;
    for (node_t *u = v; (u = neighbor(u, dir)); ) {
	if (is_a_normal_node_of(g, u))
	    rv = u;
	else if (is_a_vnode_of_an_edge_of(g, u))
	    rv = u;
    }
    return rv;
}

void save_vlist(graph_t * g)
{
    int r;

    if (GD_rankleader(g))
	for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	    GD_rankleader(g)[r] = GD_rank(g)[r].v[0];
	}
}

void rec_save_vlists(graph_t * g)
{
    int c;

    save_vlist(g);
    for (c = 1; c <= GD_n_cluster(g); c++)
	rec_save_vlists(GD_clust(g)[c]);
}


void rec_reset_vlists(graph_t * g)
{
    // fix vlists of sub-clusters
    for (int c = 1; c <= GD_n_cluster(g); c++)
	rec_reset_vlists(GD_clust(g)[c]);

    if (GD_rankleader(g))
	for (int r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	    node_t *const v = GD_rankleader(g)[r];
	    if (v == NULL) {
	        continue;
	    }
#ifdef DEBUG
	    node_in_root_vlist(v);
#endif
	    node_t *const u = furthestnode(g, v, -1);
	    node_t *const w = furthestnode(g, v, 1);
	    GD_rankleader(g)[r] = u;
#ifdef DEBUG
	    assert(GD_rank(dot_root(g))[r].v[ND_order(u)] == u);
#endif
	    GD_rank(g)[r].v = GD_rank(dot_root(g))[r].v + ND_order(u);
	    GD_rank(g)[r].n = ND_order(w) - ND_order(u) + 1;
	}
}

/* The structures in crossing minimization and positioning require
 * that clusters have some node on each rank. This function recursively
 * guarantees this property. It takes into account nodes and edges in
 * a cluster, the latter causing dummy nodes for intervening ranks.
 * For any rank without node, we create a real node of small size. This
 * is stored in the subgraph sg, for easy removal later.
 *
 * I believe it is not necessary to do this for the root graph, as these
 * are laid out one component at a time and these will necessarily have a
 * node on each rank from source to sink levels.
 */
static Agraph_t *realFillRanks(Agraph_t *g, bitarray_t *ranks, Agraph_t* sg) {
    int i, c;
    Agedge_t* e;
    Agnode_t* n;

    for (c = 1; c <= GD_n_cluster(g); c++)
	sg = realFillRanks(GD_clust(g)[c], ranks, sg);

    if (dot_root(g) == g)
	return sg;
    bitarray_clear(ranks);
    for (n = agfstnode(g); n; n = agnxtnode(g,n)) {
	bitarray_set(ranks, ND_rank(n), true);
	for (e = agfstout(g,n); e; e = agnxtout(g,e)) {
	    for (i = ND_rank(n)+1; i <= ND_rank(aghead(e)); i++) 
		bitarray_set(ranks, i, true);
	}
    }
    for (i = GD_minrank(g); i <= GD_maxrank(g); i++) {
	if (!bitarray_get(*ranks, i)) {
	    if (!sg) {
		sg = agsubg (dot_root(g), "_new_rank", 1);
	    }
	    n = agnode (sg, NULL, 1);
	    agbindrec(n, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);
	    ND_rank(n) = i;
	    ND_lw(n) = ND_rw(n) = 0.5;
	    ND_ht(n) = 1;
	    ND_UF_size(n) = 1;
	    alloc_elist(4, ND_in(n));
	    alloc_elist(4, ND_out(n));
	    agsubnode (g, n, 1);
	}
    }
    return sg;
}

static void
fillRanks (Agraph_t* g)
{
    int rnks_sz = GD_maxrank(g) + 2;
    bitarray_t rnks = bitarray_new(rnks_sz);
    realFillRanks(g, &rnks, NULL);
    bitarray_reset(&rnks);
}

static void init_mincross(graph_t * g)
{
    int size;

    if (Verbose)
	start_timer();

    ReMincross = false;
    Root = g;
    /* alloc +1 for the null terminator usage in do_ordering() */
    size = agnedges(dot_root(g)) + 1;
    TE_list = gv_calloc(size, sizeof(edge_t*));
    TI_list = gv_calloc(size, sizeof(int));
    mincross_options(g);
    if (GD_flags(g) & NEW_RANK)
	fillRanks (g);
    class2(g);
    decompose(g, 1);
    allocate_ranks(g);
    ordered_edges(g);
    GlobalMinRank = GD_minrank(g);
    GlobalMaxRank = GD_maxrank(g);
}

static void flat_rev(Agraph_t * g, Agedge_t * e)
{
    int j;
    Agedge_t *rev;

    if (!ND_flat_out(aghead(e)).list)
	rev = NULL;
    else
	for (j = 0; (rev = ND_flat_out(aghead(e)).list[j]); j++)
	    if (aghead(rev) == agtail(e))
		break;
    if (rev) {
	merge_oneway(e, rev);
	if (ED_edge_type(rev) == FLATORDER && ED_to_orig(rev) == 0)
	    ED_to_orig(rev) = e;
	elist_append(e, ND_other(agtail(e)));
    } else {
	rev = new_virtual_edge(aghead(e), agtail(e), e);
	if (ED_edge_type(e) == FLATORDER)
	    ED_edge_type(rev) = FLATORDER;
	else
	    ED_edge_type(rev) = REVERSED;
	ED_label(rev) = ED_label(e);
	flat_edge(g, rev);
    }
}

static void flat_search(graph_t * g, node_t * v)
{
    int i;
    bool hascl;
    edge_t *e;
    adjmatrix_t *M = GD_rank(g)[ND_rank(v)].flat;

    ND_mark(v) = true;
    ND_onstack(v) = true;
    hascl = GD_n_cluster(dot_root(g)) > 0;
    if (ND_flat_out(v).list)
	for (i = 0; (e = ND_flat_out(v).list[i]); i++) {
	    if (hascl && !(agcontains(g, agtail(e)) && agcontains(g, aghead(e))))
		continue;
	    if (ED_weight(e) == 0)
		continue;
	    if (ND_onstack(aghead(e))) {
		matrix_set(M, (size_t)flatindex(aghead(e)), (size_t)flatindex(agtail(e)));
		delete_flat_edge(e);
		i--;
		if (ED_edge_type(e) == FLATORDER)
		    continue;
		flat_rev(g, e);
	    } else {
		matrix_set(M, (size_t)flatindex(agtail(e)), (size_t)flatindex(aghead(e)));
		if (!ND_mark(aghead(e)))
		    flat_search(g, aghead(e));
	    }
	}
    ND_onstack(v) = false;
}

static void flat_breakcycles(graph_t * g)
{
    int i, r;
    node_t *v;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	bool flat = false;
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    v = GD_rank(g)[r].v[i];
	    ND_mark(v) = false;
	    ND_onstack(v) = false;
	    ND_low(v) = i;
	    if (ND_flat_out(v).size > 0 && !flat) {
		GD_rank(g)[r].flat =
		    new_matrix((size_t)GD_rank(g)[r].n, (size_t)GD_rank(g)[r].n);
		flat = true;
	    }
	}
	if (flat) {
	    for (i = 0; i < GD_rank(g)[r].n; i++) {
		v = GD_rank(g)[r].v[i];
		if (!ND_mark(v))
		    flat_search(g, v);
	    }
	}
    }
}

/* Allocate rank structure, determining number of nodes per rank.
 * Note that no nodes are put into the structure yet.
 */
void allocate_ranks(graph_t * g)
{
    int r, low, high;
    node_t *n;
    edge_t *e;

    int *cn = gv_calloc(GD_maxrank(g) + 2, sizeof(int)); // must be 0 based, not GD_minrank
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	cn[ND_rank(n)]++;
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    low = ND_rank(agtail(e));
	    high = ND_rank(aghead(e));
	    if (low > high) {
		SWAP(&low, &high);
	    }
	    for (r = low + 1; r < high; r++)
		cn[r]++;
	}
    }
    GD_rank(g) = gv_calloc(GD_maxrank(g) + 2, sizeof(rank_t));
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	GD_rank(g)[r].an = GD_rank(g)[r].n = cn[r] + 1;
	GD_rank(g)[r].av = GD_rank(g)[r].v = gv_calloc(cn[r] + 1, sizeof(node_t*));
    }
    free(cn);
}

/* install a node at the current right end of its rank */
int install_in_rank(graph_t *g, node_t *n) {
    int i, r;

    r = ND_rank(n);
    i = GD_rank(g)[r].n;
    if (GD_rank(g)[r].an <= 0) {
	agerrorf("install_in_rank, line %d: %s %s rank %d i = %d an = 0\n",
	      __LINE__, agnameof(g), agnameof(n), r, i);
	return -1;
    }

    GD_rank(g)[r].v[i] = n;
    ND_order(n) = i;
    GD_rank(g)[r].n++;
    assert(GD_rank(g)[r].n <= GD_rank(g)[r].an);
#ifdef DEBUG
    {
	node_t *v;

	for (v = GD_nlist(g); v; v = ND_next(v))
	    if (v == n)
		break;
	assert(v != NULL);
    }
#endif
    if (ND_order(n) > GD_rank(Root)[r].an) {
	agerrorf("install_in_rank, line %d: ND_order(%s) [%d] > GD_rank(Root)[%d].an [%d]\n",
	      __LINE__, agnameof(n), ND_order(n), r, GD_rank(Root)[r].an);
	return -1;
    }
    if (r < GD_minrank(g) || r > GD_maxrank(g)) {
	agerrorf("install_in_rank, line %d: rank %d not in rank range [%d,%d]\n",
	      __LINE__, r, GD_minrank(g), GD_maxrank(g));
	return -1;
    }
    if (GD_rank(g)[r].v + ND_order(n) >
	GD_rank(g)[r].av + GD_rank(Root)[r].an) {
	agerrorf("install_in_rank, line %d: GD_rank(g)[%d].v + ND_order(%s) [%d] > GD_rank(g)[%d].av + GD_rank(Root)[%d].an [%d]\n",
	      __LINE__, r, agnameof(n),ND_order(n), r, r, GD_rank(Root)[r].an);
	return -1;
    }
    return 0;
}

/*	install nodes in ranks. the initial ordering ensure that series-parallel
 *	graphs such as trees are drawn with no crossings.  it tries searching
 *	in- and out-edges and takes the better of the two initial orderings.
 */
int build_ranks(graph_t *g, int pass) {
    int i, j;
    node_t *n, *ns;
    edge_t **otheredges;
    node_queue_t q = {0};
    for (n = GD_nlist(g); n; n = ND_next(n))
	MARK(n) = false;

#ifdef DEBUG
    {
	edge_t *e;
	for (n = GD_nlist(g); n; n = ND_next(n)) {
	    for (i = 0; (e = ND_out(n).list[i]); i++)
		assert(!MARK(aghead(e)));
	    for (i = 0; (e = ND_in(n).list[i]); i++)
		assert(!MARK(agtail(e)));
	}
    }
#endif

    for (i = GD_minrank(g); i <= GD_maxrank(g); i++)
	GD_rank(g)[i].n = 0;

    const bool walkbackwards = g != agroot(g); // if this is a cluster, need to
                                               // walk GD_nlist backward to
                                               // preserve input node order
    if (walkbackwards) {
	for (ns = GD_nlist(g); ND_next(ns); ns = ND_next(ns)) {
	    ;
	}
    } else {
	ns = GD_nlist(g);
    }
    for (n = ns; n; n = walkbackwards ? ND_prev(n) : ND_next(n)) {
	otheredges = pass == 0 ? ND_in(n).list : ND_out(n).list;
	if (otheredges[0] != NULL)
	    continue;
	if (!MARK(n)) {
	    MARK(n) = true;
	    LIST_PUSH_BACK(&q, n);
	    while (!LIST_IS_EMPTY(&q)) {
		node_t *n0 = LIST_POP_FRONT(&q);
		if (ND_ranktype(n0) != CLUSTER) {
		    if (install_in_rank(g, n0) != 0) {
		        LIST_FREE(&q);
		        return -1;
		    }
		    enqueue_neighbors(&q, n0, pass);
		} else {
		    const int rc = install_cluster(g, n0, pass, &q);
		    if (rc != 0) {
		        LIST_FREE(&q);
		        return rc;
		    }
		}
	    }
	}
    }
    assert(LIST_IS_EMPTY(&q));
    for (i = GD_minrank(g); i <= GD_maxrank(g); i++) {
	GD_rank(Root)[i].valid = false;
	if (GD_flip(g) && GD_rank(g)[i].n > 0) {
	    node_t **vlist = GD_rank(g)[i].v;
	    int num_nodes_1 = GD_rank(g)[i].n - 1;
	    int half_num_nodes_1 = num_nodes_1 / 2;
	    for (j = 0; j <= half_num_nodes_1; j++)
		exchange(vlist[j], vlist[num_nodes_1 - j]);
	}
    }

    if (g == dot_root(g) && ncross() > 0)
	transpose(g, false);
    LIST_FREE(&q);
    return 0;
}

void enqueue_neighbors(node_queue_t *q, node_t *n0, int pass) {
    edge_t *e;

    if (pass == 0) {
	for (size_t i = 0; i < ND_out(n0).size; i++) {
	    e = ND_out(n0).list[i];
	    if (!MARK(aghead(e))) {
		MARK(aghead(e)) = true;
		LIST_PUSH_BACK(q, aghead(e));
	    }
	}
    } else {
	for (size_t i = 0; i < ND_in(n0).size; i++) {
	    e = ND_in(n0).list[i];
	    if (!MARK(agtail(e))) {
		MARK(agtail(e)) = true;
		LIST_PUSH_BACK(q, agtail(e));
	    }
	}
    }
}

static bool constraining_flat_edge(Agraph_t *g, Agedge_t *e) {
  if (ED_weight(e) == 0)
    return false;
  if (!inside_cluster(g, agtail(e)))
    return false;
  if (!inside_cluster(g, aghead(e)))
    return false;
  return true;
}

/* construct nodes reachable from 'here' in post-order.
* This is the same as doing a topological sort in reverse order.
*/
static void postorder(graph_t *g, node_t *v, nodes_t *list, int r) {
    edge_t *e;
    int i;

    MARK(v) = true;
    if (ND_flat_out(v).size > 0) {
	for (i = 0; (e = ND_flat_out(v).list[i]); i++) {
	    if (!constraining_flat_edge(g, e)) continue;
	    if (!MARK(aghead(e)))
		postorder(g, aghead(e), list, r);
	}
    }
    assert(ND_rank(v) == r);
    LIST_APPEND(list, v);
}

static void flat_reorder(graph_t * g)
{
    int i, r, local_in_cnt, local_out_cnt, base_order;
    node_t *v;
    nodes_t temprank = {0};
    edge_t *flat_e, *e;

    if (!GD_has_flat_edges(g))
	return;
    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	if (GD_rank(g)[r].n == 0) continue;
	base_order = ND_order(GD_rank(g)[r].v[0]);
	for (i = 0; i < GD_rank(g)[r].n; i++)
	    MARK(GD_rank(g)[r].v[i]) = false;
	LIST_CLEAR(&temprank);

	/* construct reverse topological sort order in temprank */
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    if (GD_flip(g)) v = GD_rank(g)[r].v[i];
	    else v = GD_rank(g)[r].v[GD_rank(g)[r].n - i - 1];

	    local_in_cnt = local_out_cnt = 0;
	    for (size_t j = 0; j < ND_flat_in(v).size; j++) {
		flat_e = ND_flat_in(v).list[j];
		if (constraining_flat_edge(g, flat_e)) local_in_cnt++;
	    }
	    for (size_t j = 0; j < ND_flat_out(v).size; j++) {
		flat_e = ND_flat_out(v).list[j];
		if (constraining_flat_edge(g, flat_e)) local_out_cnt++;
	    }
	    if (local_in_cnt == 0 && local_out_cnt == 0)
		LIST_APPEND(&temprank, v);
	    else {
		if (!MARK(v) && local_in_cnt == 0) {
		    postorder(g, v, &temprank, r);
		}
	    }
	}

	if (!LIST_IS_EMPTY(&temprank)) {
	    if (!GD_flip(g)) {
		LIST_REVERSE(&temprank);
	    }
	    for (i = 0; i < GD_rank(g)[r].n; i++) {
		v = GD_rank(g)[r].v[i] = LIST_GET(&temprank, (size_t)i);
		ND_order(v) = i + base_order;
	    }

	    /* nonconstraint flat edges must be made LR */
	    for (i = 0; i < GD_rank(g)[r].n; i++) {
		v = GD_rank(g)[r].v[i];
		if (ND_flat_out(v).list) {
		    for (size_t j = 0; (e = ND_flat_out(v).list[j]); j++) {
			if ((!GD_flip(g) && ND_order(aghead(e)) < ND_order(agtail(e))) ||
				  (GD_flip(g) && ND_order(aghead(e)) > ND_order(agtail(e)))) {
			    assert(!constraining_flat_edge(g, e));
			    delete_flat_edge(e);
			    j--;
			    flat_rev(g, e);
			}
		    }
		}
	    }
	    /* postprocess to restore intended order */
	}
	/* else do no harm! */
	GD_rank(Root)[r].valid = false;
    }
    LIST_FREE(&temprank);
}

static void reorder(graph_t * g, int r, bool reverse, bool hasfixed)
{
    int changed = 0, nelt;
    node_t **vlist = GD_rank(g)[r].v;
    node_t **lp, **rp, **ep = vlist + GD_rank(g)[r].n;

    for (nelt = GD_rank(g)[r].n - 1; nelt >= 0; nelt--) {
	lp = vlist;
	while (lp < ep) {
	    /* find leftmost node that can be compared */
	    while (lp < ep && ND_mval(*lp) < 0)
		lp++;
	    if (lp >= ep)
		break;
	    /* find the node that can be compared */
	    bool sawclust = false;
	    bool muststay = false;
	    for (rp = lp + 1; rp < ep; rp++) {
		if (sawclust && ND_clust(*rp))
		    continue;	/* ### */
		if (left2right(g, *lp, *rp)) {
		    muststay = true;
		    break;
		}
		if (ND_mval(*rp) >= 0)
		    break;
		if (ND_clust(*rp))
		    sawclust = true;	/* ### */
	    }
	    if (rp >= ep)
		break;
	    if (!muststay) {
		const double p1 = ND_mval(*lp);
		const double p2 = ND_mval(*rp);
		if (p1 > p2 || (p1 >= p2 && reverse)) {
		    exchange(*lp, *rp);
		    changed++;
		}
	    }
	    lp = rp;
	}
	if (!hasfixed && !reverse)
	    ep--;
    }

    if (changed) {
	GD_rank(Root)[r].valid = false;
	if (r > 0)
	    GD_rank(Root)[r - 1].valid = false;
    }
}

static void mincross_step(graph_t * g, int pass)
{
    int r, other, first, last, dir;

    bool reverse = pass % 4 < 2;

    if (pass % 2 == 0) {	/* down pass */
	first = GD_minrank(g) + 1;
	if (GD_minrank(g) > GD_minrank(Root))
	    first--;
	last = GD_maxrank(g);
	dir = 1;
    } else {			/* up pass */
	first = GD_maxrank(g) - 1;
	last = GD_minrank(g);
	if (GD_maxrank(g) < GD_maxrank(Root))
	    first++;
	dir = -1;
    }

    for (r = first; r != last + dir; r += dir) {
	other = r - dir;
	bool hasfixed = medians(g, r, other);
	reorder(g, r, reverse, hasfixed);
    }
    transpose(g, !reverse);
}

static int local_cross(elist l, int dir)
{
    int i, j;
    int cross = 0;
    edge_t *e, *f;
    bool is_out = dir > 0;
    for (i = 0; (e = l.list[i]); i++) {
	if (is_out)
	    for (j = i + 1; (f = l.list[j]); j++) {
		if ((ND_order(aghead(f)) - ND_order(aghead(e)))
			 * (ED_tail_port(f).p.x - ED_tail_port(e).p.x) < 0)
		    cross += ED_xpenalty(e) * ED_xpenalty(f);
	} else
	    for (j = i + 1; (f = l.list[j]); j++) {
		if ((ND_order(agtail(f)) - ND_order(agtail(e)))
			* (ED_head_port(f).p.x - ED_head_port(e).p.x) < 0)
		    cross += ED_xpenalty(e) * ED_xpenalty(f);
	    }
    }
    return cross;
}

static int64_t rcross(graph_t *g, int r) {
    int top, bot, max, i, k;
    node_t **rtop, *v;

    int64_t cross = 0;
    max = 0;
    rtop = GD_rank(g)[r].v;

    int *Count = gv_calloc(GD_rank(Root)[r + 1].n + 1, sizeof(int));

    for (top = 0; top < GD_rank(g)[r].n; top++) {
	edge_t *e;
	if (max > 0) {
	    for (i = 0; (e = ND_out(rtop[top]).list[i]); i++) {
		for (k = ND_order(aghead(e)) + 1; k <= max; k++)
		    cross += Count[k] * ED_xpenalty(e);
	    }
	}
	for (i = 0; (e = ND_out(rtop[top]).list[i]); i++) {
	    int inv = ND_order(aghead(e));
	    if (inv > max)
		max = inv;
	    Count[inv] += ED_xpenalty(e);
	}
    }
    for (top = 0; top < GD_rank(g)[r].n; top++) {
	v = GD_rank(g)[r].v[top];
	if (ND_has_port(v))
	    cross += local_cross(ND_out(v), 1);
    }
    for (bot = 0; bot < GD_rank(g)[r + 1].n; bot++) {
	v = GD_rank(g)[r + 1].v[bot];
	if (ND_has_port(v))
	    cross += local_cross(ND_in(v), -1);
    }
    free(Count);
    return cross;
}

static int64_t ncross(void) {
    int r;

    graph_t *g = Root;
    int64_t count = 0;
    for (r = GD_minrank(g); r < GD_maxrank(g); r++) {
	if (GD_rank(g)[r].valid)
	    count += GD_rank(g)[r].cache_nc;
	else {
	    const int64_t nc = GD_rank(g)[r].cache_nc = rcross(g, r);
	    count += nc;
	    GD_rank(g)[r].valid = true;
	}
    }
    return count;
}

static int ordercmpf(const void *x, const void *y) {
  const int *i0 = x;
  const int *i1 = y;
  if (*i0 < *i1) {
    return -1;
  }
  if (*i0 > *i1) {
    return 1;
  }
  return 0;
}

/* Calculate a mval for nodes with no in or out non-flat edges.
 * Assume (ND_out(n).size == 0) && (ND_in(n).size == 0)
 * Find flat edge a->n where a has the largest order and set
 * n.mval = a.mval+1, assuming a.mval is defined (>=0).
 * If there are no flat in edges, find flat edge n->a where a 
 * has the smallest order and set * n.mval = a.mval-1, assuming 
 * a.mval is > 0.
 * Return true if n.mval is left -1, indicating a fixed node for sorting.
 */
static bool flat_mval(node_t * n)
{
    int i;
    edge_t *e, **fl;
    node_t *nn;

    if (ND_flat_in(n).size > 0) {
	fl = ND_flat_in(n).list;
	nn = agtail(fl[0]);
	for (i = 1; (e = fl[i]); i++)
	    if (ND_order(agtail(e)) > ND_order(nn))
		nn = agtail(e);
	if (ND_mval(nn) >= 0) {
	    ND_mval(n) = ND_mval(nn) + 1;
	    return false;
	}
    } else if (ND_flat_out(n).size > 0) {
	fl = ND_flat_out(n).list;
	nn = aghead(fl[0]);
	for (i = 1; (e = fl[i]); i++)
	    if (ND_order(aghead(e)) < ND_order(nn))
		nn = aghead(e);
	if (ND_mval(nn) > 0) {
	    ND_mval(n) = ND_mval(nn) - 1;
	    return false;
	}
    }
    return true;
}

#define VAL(node,port) (MC_SCALE * ND_order(node) + (port).order)

static bool medians(graph_t * g, int r0, int r1)
{
    int i, j0, lspan, rspan, *list;
    node_t *n, **v;
    edge_t *e;
    bool hasfixed = false;

    list = TI_list;
    v = GD_rank(g)[r0].v;
    for (i = 0; i < GD_rank(g)[r0].n; i++) {
	n = v[i];
	size_t j = 0;
	if (r1 > r0)
	    for (j0 = 0; (e = ND_out(n).list[j0]); j0++) {
		if (ED_xpenalty(e) > 0)
		    list[j++] = VAL(aghead(e), ED_head_port(e));
	} else
	    for (j0 = 0; (e = ND_in(n).list[j0]); j0++) {
		if (ED_xpenalty(e) > 0)
		    list[j++] = VAL(agtail(e), ED_tail_port(e));
	    }
	switch (j) {
	case 0:
	    ND_mval(n) = -1;
	    break;
	case 1:
	    ND_mval(n) = list[0];
	    break;
	case 2:
	    ND_mval(n) = (list[0] + list[1]) / 2;
	    break;
	default:
	    qsort(list, j, sizeof(int), ordercmpf);
	    if (j % 2)
		ND_mval(n) = list[j / 2];
	    else {
		/* weighted median */
		size_t rm = j / 2;
		size_t lm = rm - 1;
		rspan = list[j - 1] - list[rm];
		lspan = list[lm] - list[0];
		if (lspan == rspan)
		    ND_mval(n) = (list[lm] + list[rm]) / 2;
		else {
		    double w = list[lm] * (double)rspan + list[rm] * (double)lspan;
		    ND_mval(n) = w / (lspan + rspan);
		}
	    }
	}
    }
    for (i = 0; i < GD_rank(g)[r0].n; i++) {
	n = v[i];
	if (ND_out(n).size == 0 && ND_in(n).size == 0)
	    hasfixed |= flat_mval(n);
    }
    return hasfixed;
}

static int nodeposcmpf(const void *x, const void *y) {
  node_t *const *const n0 = x;
  node_t *const *const n1 = y;
  if (ND_order(*n0) < ND_order(*n1)) {
    return -1;
  }
  if (ND_order(*n0) > ND_order(*n1)) {
    return 1;
  }
  return 0;
}

static int edgeidcmpf(const void *x, const void *y) {
  edge_t *const *const e0 = x;
  edge_t *const *const e1 = y;
  if (AGSEQ(*e0) < AGSEQ(*e1)) {
    return -1;
  }
  if (AGSEQ(*e0) > AGSEQ(*e1)) {
    return 1;
  }
  return 0;
}

/* following code deals with weights of edges of "virtual" nodes */
#define ORDINARY	0
#define SINGLETON	1
#define	VIRTUALNODE	2
#define NTYPES		3

#define C_EE		1
#define C_VS		2
#define C_SS		2
#define C_VV		4

static const int table[NTYPES][NTYPES] = {
    /* ordinary */ {C_EE, C_EE, C_EE},
    /* singleton */ {C_EE, C_SS, C_VS},
    /* virtual */ {C_EE, C_VS, C_VV}
};

static int endpoint_class(node_t * n)
{
    if (ND_node_type(n) == VIRTUAL)
	return VIRTUALNODE;
    if (ND_weight_class(n) <= 1)
	return SINGLETON;
    return ORDINARY;
}

void virtual_weight(edge_t * e)
{
    int t;
    t = table[endpoint_class(agtail(e))][endpoint_class(aghead(e))];

    /* check whether the upcoming computation will overflow */
    assert(t >= 0);
    if (INT_MAX / t < ED_weight(e)) {
	agerrorf("overflow when calculating virtual weight of edge\n");
	graphviz_exit(EXIT_FAILURE);
    }

    ED_weight(e) *= t;
}

#ifdef DEBUG
void check_order(void)
{
    int i, r;
    node_t *v;
    graph_t *g = Root;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	assert(GD_rank(g)[r].v[GD_rank(g)[r].n] == NULL);
	for (i = 0; (v = GD_rank(g)[r].v[i]); i++) {
	    assert(ND_rank(v) == r);
	    assert(ND_order(v) == i);
	}
    }
}
#endif

static void mincross_options(graph_t * g)
{
    char *p;
    double f;

    /* set default values */
    MinQuit = 8;
    MaxIter = 24;

    p = agget(g, "mclimit");
    if (p && (f = atof(p)) > 0.0) {
	MinQuit = MAX(1, scale_clamp(MinQuit, f));
	MaxIter = MAX(1, scale_clamp(MaxIter, f));
    }
}

#ifdef DEBUG
void check_vlists(graph_t * g)
{
    int c, i, j, r;
    node_t *u;

    for (r = GD_minrank(g); r <= GD_maxrank(g); r++) {
	for (i = 0; i < GD_rank(g)[r].n; i++) {
	    u = GD_rank(g)[r].v[i];
	    j = ND_order(u);
	    assert(GD_rank(Root)[r].v[j] == u);
	}
	if (GD_rankleader(g)) {
	    u = GD_rankleader(g)[r];
	    j = ND_order(u);
	    assert(GD_rank(Root)[r].v[j] == u);
	}
    }
    for (c = 1; c <= GD_n_cluster(g); c++)
	check_vlists(GD_clust(g)[c]);
}

void node_in_root_vlist(node_t * n)
{
    node_t **vptr;

    for (vptr = GD_rank(Root)[ND_rank(n)].v; *vptr; vptr++)
	if (*vptr == n)
	    break;
    if (*vptr == 0)
	abort();
}
#endif				/* DEBUG code */
