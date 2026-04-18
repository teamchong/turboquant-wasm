/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <neatogen/bfs.h>
#include <neatogen/dijkstra.h>
#include <neatogen/kkutils.h>
#include <stdlib.h>
#include <math.h>
#include <util/alloc.h>
#include <util/sort.h>

size_t common_neighbors(vtx_data *graph, int u, int *v_vector) {
    // count number of common neighbors of 'v_vector' and 'u'
    int neighbor;
    size_t num_shared_neighbors = 0;
    for (size_t j = 1; j < graph[u].nedges; j++) {
	neighbor = graph[u].edges[j];
	if (v_vector[neighbor] > 0) {
	    // a shared neighbor
	    num_shared_neighbors++;
	}
    }
    return num_shared_neighbors;
}
void fill_neighbors_vec_unweighted(vtx_data * graph, int vtx, int *vtx_vec)
{
    /* a node is NOT a neighbor of itself! */
    /* unlike the other version of this function */
    for (size_t j = 1; j < graph[vtx].nedges; j++) {
	vtx_vec[graph[vtx].edges[j]] = 1;
    }
}

void empty_neighbors_vec(vtx_data * graph, int vtx, int *vtx_vec)
{
    /* a node is NOT a neighbor of itself! */
    /* unlike the other version ofthis function */
    for (size_t j = 1; j < graph[vtx].nedges; j++) {
	vtx_vec[graph[vtx].edges[j]] = 0;
    }
}

/* compute_apsp_dijkstra:
 * Assumes the graph has weights
 */
static DistType **compute_apsp_dijkstra(vtx_data * graph, int n)
{
    int i;

    DistType *storage = gv_calloc((size_t)(n * n), sizeof(DistType));
    DistType **dij = gv_calloc(n, sizeof(DistType*));
    for (i = 0; i < n; i++)
	dij[i] = storage + i * n;

    for (i = 0; i < n; i++) {
	dijkstra(i, graph, n, dij[i]);
    }
    return dij;
}

static DistType **compute_apsp_simple(vtx_data * graph, int n)
{
    /* compute all pairs shortest path */
    /* for unweighted graph */
    int i;
    DistType *storage = gv_calloc((size_t)(n * n), sizeof(DistType));

    DistType **dij = gv_calloc(n, sizeof(DistType*));
    for (i = 0; i < n; i++) {
	dij[i] = storage + i * n;
    }
    for (i = 0; i < n; i++) {
	bfs(i, graph, n, dij[i]);
    }
    return dij;
}

DistType **compute_apsp(vtx_data * graph, int n)
{
    if (graph->ewgts)
	return compute_apsp_dijkstra(graph, n);
    else
	return compute_apsp_simple(graph, n);
}

DistType **compute_apsp_artificial_weights(vtx_data *graph, int n) {
    DistType **Dij;
    /* compute all-pairs-shortest-path-length while weighting the graph */
    /* so high-degree nodes are distantly located */

    float *old_weights = graph[0].ewgts;

    compute_new_weights(graph, n);
    Dij = compute_apsp_dijkstra(graph, n);
    restore_old_weights(graph, n, old_weights);
    return Dij;
}


/**********************/
/*		      */
/*  Quick Sort        */
/*		      */
/**********************/

double distance_kD(double **coords, int dim, int i, int j)
{
    /* compute a k-D Euclidean distance between 'coords[*][i]' and 'coords[*][j]' */
    double sum = 0;
    int k;
    for (k = 0; k < dim; k++) {
	sum +=
	    (coords[k][i] - coords[k][j]) * (coords[k][i] - coords[k][j]);
    }
    return sqrt(sum);
}

static int fcmpf(const void *a, const void *b, void *context) {
    const int *ip1 = a;
    const int *ip2 = b;
    float *fvals = context;
    float d1 = fvals[*ip1];
    float d2 = fvals[*ip2];
    if (d1 < d2) {
      return -1;
    }
    if (d1 > d2) {
      return 1;
    }
    return 0;
}

void quicksort_placef(float *place, int *ordering, int first, int last)
{
    if (first < last) {
	gv_sort(ordering + first, last - first + 1, sizeof(ordering[0]), fcmpf, place);
    }
}

static int cmp(const void *a, const void *b, void *context) {
  const int *x = a;
  const int *y = b;
  const double *place = context;

  if (place[*x] < place[*y]) {
    return -1;
  }
  if (place[*x] > place[*y]) {
    return 1;
  }
  return 0;
}

void quicksort_place(double *place, int *ordering, int size) {
  gv_sort(ordering, size, sizeof(ordering[0]), cmp, place);
}

void compute_new_weights(vtx_data * graph, int n)
{
    /* Reweight graph so that high degree nodes will be separated */

    int i;
    size_t nedges = 0;
    int *vtx_vec = gv_calloc(n, sizeof(int));
    size_t deg_i, deg_j;
    int neighbor;

    for (i = 0; i < n; i++) {
	nedges += graph[i].nedges;
    }
    float *weights = gv_calloc(nedges, sizeof(float));

    for (i = 0; i < n; i++) {
	graph[i].ewgts = weights;
	fill_neighbors_vec_unweighted(graph, i, vtx_vec);
	deg_i = graph[i].nedges - 1;
	for (size_t j = 1; j <= deg_i; j++) {
	    neighbor = graph[i].edges[j];
	    deg_j = graph[neighbor].nedges - 1;
	    weights[j] =
		(float)(deg_i + deg_j - 2 * common_neighbors(graph, neighbor, vtx_vec));
	}
	empty_neighbors_vec(graph, i, vtx_vec);
	weights += graph[i].nedges;
    }
    free(vtx_vec);
}

void restore_old_weights(vtx_data * graph, int n, float *old_weights)
{
    int i;
    free(graph[0].ewgts);
    graph[0].ewgts = NULL;
    if (old_weights != NULL) {
	for (i = 0; i < n; i++) {
	    graph[i].ewgts = old_weights;
	    old_weights += graph[i].nedges;
	}
    }
}
