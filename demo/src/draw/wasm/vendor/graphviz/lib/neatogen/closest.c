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
#include <cgraph/list.h>
#include <limits.h>
#include <neatogen/kkutils.h>
#include <neatogen/closest.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/sort.h>

/*****************************************
** This module contains functions that  **
** given a 1-D layout construct a graph **
** where close nodes in this layout are **
** adjacent                             **
*****************************************/

typedef struct {
    /* this structure represents two nodes in the 1-D layout */
    size_t left;  ///< the left node in the pair
    size_t right; ///< the right node in the pair
    double dist;		/* distance between the nodes in the layout */
} Pair;

#define LT(p,q) ((p).dist < (q).dist)
#define EQ(p,q) ((p).dist == (q).dist)

DEFINE_LIST(pairs, Pair *)

static void push(pairs_t *s, Pair x) {

  // copy the item to save it on the stack
  Pair *copy = gv_alloc(sizeof(x));
  *copy = x;

  pairs_push_back(s, copy);
}

static bool pop(pairs_t *s, Pair *x) {

  // is the stack empty?
  if (pairs_is_empty(s)) {
    return false;
  }

  // remove the top and pass it back to the caller
  Pair *top = pairs_pop_back(s);
  *x = *top;

  // discard the previously saved copy
  free(top);

  return true;
}

#define sub(h,i) (*pairs_get((h), (i)))

/* An auxulliary data structure (a heap) for 
 * finding the closest pair in the layout
 */
typedef struct {
    Pair *data;
    size_t heapSize;
    size_t maxSize;
} PairHeap;

#define left(i) (2*(i))
#define right(i) (2*(i)+1)
#define parent(i) ((i)/2)
#define insideHeap(h,i) ((i)<h->heapSize)
#define greaterPriority(h,i,j) \
  (LT(h->data[i],h->data[j]) || ((EQ(h->data[i],h->data[j])) && (rand()%2)))

#define exchange(h,i,j) {Pair temp; \
        temp=h->data[i]; \
        h->data[i]=h->data[j]; \
        h->data[j]=temp; \
}

static void heapify(PairHeap *h, size_t i) {
    size_t largest;
    while (1) {
	size_t l = left(i);
	size_t r = right(i);
	if (insideHeap(h, l) && greaterPriority(h, l, i))
	    largest = l;
	else
	    largest = i;
	if (insideHeap(h, r) && greaterPriority(h, r, largest))
	    largest = r;
	if (largest == i)
	    break;

	exchange(h, largest, i);
	i = largest;
    }
}

static void freeHeap(PairHeap * h)
{
    free(h->data);
}

static void initHeap(PairHeap *h, double *place, size_t *ordering, size_t n) {
    Pair edge;

    h->heapSize = n == 0 ? 0 : (n - 1);
    h->maxSize = h->heapSize;
    h->data = gv_calloc(h->maxSize, sizeof(Pair));

    for (size_t i = 0; n != 0 && i < n - 1; i++) {
	edge.left = ordering[i];
	edge.right = ordering[i + 1];
	edge.dist = place[ordering[i + 1]] - place[ordering[i]];
	h->data[i] = edge;
    }
    for (size_t j = (n - 1) / 2; n != 0 && j != SIZE_MAX; j--) {
	heapify(h, j);
    }
}

static bool extractMax(PairHeap * h, Pair * max)
{
    if (h->heapSize == 0)
	return false;

    *max = h->data[0];
    h->data[0] = h->data[h->heapSize - 1];
    h->heapSize--;
    heapify(h, 0);
    return true;
}

static void insert(PairHeap * h, Pair edge)
{
    size_t i = h->heapSize;
    if (h->heapSize == h->maxSize) {
	size_t newSize = h->maxSize * 2;
	h->data = gv_recalloc(h->data, h->maxSize, newSize, sizeof(Pair));
	h->maxSize = newSize;
    }
    h->heapSize++;
    h->data[i] = edge;
    while (i > 0 && greaterPriority(h, i, parent(i))) {
	exchange(h, i, parent(i));
	i = parent(i);
    }
}

static int cmp(const void *a, const void *b, void *context) {
  const size_t *x = a;
  const size_t *y = b;
  const double *place = context;

  if (place[*x] < place[*y]) {
    return -1;
  }
  if (place[*x] > place[*y]) {
    return 1;
  }
  return 0;
}

static void find_closest_pairs(double *place, size_t n, int num_pairs,
                               pairs_t* pairs_stack) {
    /* Fill the stack 'pairs_stack' with 'num_pairs' closest pairs int the 1-D layout 'place' */
    PairHeap heap;
    size_t *left = gv_calloc(n, sizeof(size_t));
    size_t *right = gv_calloc(n, sizeof(size_t));
    Pair pair = {0}, new_pair;

    /* Order the nodes according to their place */
    size_t *ordering = gv_calloc(n, sizeof(size_t));
    size_t *inv_ordering = gv_calloc(n, sizeof(size_t));

    for (size_t i = 0; i < n; i++) {
	ordering[i] = i;
    }
    gv_sort(ordering, n, sizeof(ordering[0]), cmp, place);
    for (size_t i = 0; i < n; i++) {
	inv_ordering[ordering[i]] = i;
    }

    /* Initialize heap with all consecutive pairs */
    initHeap(&heap, place, ordering, n);

    /* store the leftmost and rightmost neighbors of each node that were entered into heap */
    for (size_t i = 1; i < n; i++) {
	left[ordering[i]] = ordering[i - 1];
    }
    for (size_t i = 0; n != 0 && i < n - 1; i++) {
	right[ordering[i]] = ordering[i + 1];
    }

    /* extract the 'num_pairs' closest pairs */
    for (int i = 0; i < num_pairs; i++) {

	if (!extractMax(&heap, &pair)) {
	    break;		/* not enough pairs */
	}
	push(pairs_stack, pair);
	/* insert to heap "descendant" pairs */
	size_t left_index = inv_ordering[pair.left];
	size_t right_index = inv_ordering[pair.right];
	if (left_index > 0) {
	    size_t neighbor = ordering[left_index - 1];
	    if (inv_ordering[right[neighbor]] < right_index) {
		/* we have a new pair */
		new_pair.left = neighbor;
		new_pair.right = pair.right;
		new_pair.dist = place[pair.right] - place[neighbor];
		insert(&heap, new_pair);
		right[neighbor] = pair.right;
		left[pair.right] = neighbor;
	    }
	}
	if (right_index < n - 1) {
	    size_t neighbor = ordering[right_index + 1];
	    if (inv_ordering[left[neighbor]] > left_index) {
		/* we have a new pair */
		new_pair.left = pair.left;
		new_pair.right = neighbor;
		new_pair.dist = place[neighbor] - place[pair.left];
		insert(&heap, new_pair);
		left[neighbor] = pair.left;
		right[pair.left] = neighbor;
	    }
	}
    }
    free(left);
    free(right);
    free(ordering);
    free(inv_ordering);
    freeHeap(&heap);
}

static void add_edge(vtx_data * graph, int u, int v)
{
    for (size_t i = 0; i < graph[u].nedges; i++) {
	if (graph[u].edges[i] == v) {
	    /* edge already exist */
	    return;
	}
    }
    /* add the edge */
    graph[u].edges[graph[u].nedges++] = v;
    graph[v].edges[graph[v].nedges++] = u;
    if (graph[0].ewgts != NULL) {
	graph[u].ewgts[0]--;
	graph[v].ewgts[0]--;
    }
}

static void construct_graph(size_t n, pairs_t *edges_stack,
                            vtx_data **New_graph) {
    /* construct an unweighted graph using the edges 'edges_stack' */
    vtx_data *new_graph;

    /* first compute new degrees and nedges; */
    int *degrees = gv_calloc(n, sizeof(int));
    size_t top = pairs_size(edges_stack);
    size_t new_nedges = 2 * top + n;
    Pair pair;
    int *edges = gv_calloc(new_nedges, sizeof(int));
    float *weights = gv_calloc(new_nedges, sizeof(float));

    for (size_t i = 0; i < n; i++) {
	degrees[i] = 1;		/* save place for the self loop */
    }
    for (size_t i = 0; i < top; i++) {
	pair = sub(edges_stack, i);
	degrees[pair.left]++;
	degrees[pair.right]++;
    }

    /* copy graph into new_graph: */
    for (size_t i = 0; i < new_nedges; i++) {
	weights[i] = 1.0;
    }

    *New_graph = new_graph = gv_calloc(n, sizeof(vtx_data));
    for (size_t i = 0; i < n; i++) {
	new_graph[i].nedges = 1;
	new_graph[i].ewgts = weights;
	new_graph[i].edges = edges;
	assert(i <= INT_MAX);
	*edges = (int)i; // self loop for Lap
	*weights = 0;		/* self loop weight for Lap */
	weights += degrees[i];
	edges += degrees[i];	/* reserve space for possible more edges */
    }

    free(degrees);

    /* add all edges from stack */
    while (pop(edges_stack, &pair)) {
	assert(pair.left <= INT_MAX);
	assert(pair.right <= INT_MAX);
	add_edge(new_graph, (int)pair.left, (int)pair.right);
    }
}

void
closest_pairs2graph(double *place, int n, int num_pairs, vtx_data ** graph)
{
    /* build a graph with with edges between the 'num_pairs' closest pairs in the 1-D space: 'place' */
    pairs_t pairs_stack = {0};
    assert(n >= 0);
    find_closest_pairs(place, (size_t)n, num_pairs, &pairs_stack);
    construct_graph((size_t)n, &pairs_stack, graph);
    pairs_free(&pairs_stack);
}
