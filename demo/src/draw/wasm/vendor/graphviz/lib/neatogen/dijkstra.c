/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/


/******************************************

	Dijkstra algorithm
	Computes single-source distances for
	weighted graphs

******************************************/

#include <assert.h>
#include <float.h>
#include <neatogen/bfs.h>
#include <neatogen/dijkstra.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/bitarray.h>

typedef DistType Word;

#define MAX_DIST ((DistType)INT_MAX)

/* This heap class is suited to the Dijkstra alg.
   data[i]=vertexNum <==> index[vertexNum]=i
*/

#define left(i) (2*(i))
#define right(i) (2*(i)+1)
#define parent(i) ((i)/2)
#define insideHeap(h,i) ((i)<h->heapSize)
#define greaterPriority(h,i,j,dist) (dist[h->data[i]]<dist[h->data[j]])
#define assign(h,i,j,index) {h->data[i]=h->data[j]; index[h->data[i]]=i;}
#define exchange(h,i,j,index) {int temp; \
		temp=h->data[i]; \
		h->data[i]=h->data[j]; \
		h->data[j]=temp; \
		index[h->data[i]]=i; \
		index[h->data[j]]=j; \
}

typedef struct {
    int *data;
    int heapSize;
} heap;

static void heapify(heap * h, int i, int index[], Word dist[])
{
    int l, r, largest;
    while (1) {
	l = left(i);
	r = right(i);
	if (insideHeap(h, l) && greaterPriority(h, l, i, dist))
	    largest = l;
	else
	    largest = i;
	if (insideHeap(h, r) && greaterPriority(h, r, largest, dist))
	    largest = r;

	if (largest == i)
	    break;

	exchange(h, largest, i, index);
	i = largest;
    }
}

static void freeHeap(heap * h)
{
    free(h->data);
}

static void
initHeap(heap * h, int startVertex, int index[], Word dist[], int n)
{
    int i, count;
    int j;    /* We cannot use an unsigned value in this loop */
    if (n == 1) h->data = NULL;
    else h->data = gv_calloc(n - 1, sizeof(int));
    h->heapSize = n - 1;

    for (count = 0, i = 0; i < n; i++)
	if (i != startVertex) {
	    h->data[count] = i;
	    index[i] = count;
	    count++;
	}

    for (j = (n - 1) / 2; j >= 0; j--)
	heapify(h, j, index, dist);
}

static bool extractMax(heap * h, int *max, int index[], Word dist[])
{
    if (h->heapSize == 0)
	return false;

    *max = h->data[0];
    h->data[0] = h->data[h->heapSize - 1];
    index[h->data[0]] = 0;
    h->heapSize--;
    heapify(h, 0, index, dist);

    return true;
}

static void
increaseKey(heap * h, int increasedVertex, Word newDist, int index[],
	    Word dist[])
{
    int placeInHeap;
    int i;

    if (dist[increasedVertex] <= newDist)
	return;

    placeInHeap = index[increasedVertex];

    dist[increasedVertex] = newDist;

    i = placeInHeap;
    while (i > 0 && dist[h->data[parent(i)]] > newDist) {	/* can write here: greaterPriority(i,parent(i),dist) */
	assign(h, i, parent(i), index);
	i = parent(i);
    }
    h->data[i] = increasedVertex;
    index[increasedVertex] = i;
}

void dijkstra(int vertex, vtx_data * graph, int n, DistType * dist)
{
    heap H;
    int closestVertex, neighbor;
    DistType closestDist, prevClosestDist = MAX_DIST;

    int *index = gv_calloc(n, sizeof(int));

    /* initial distances with edge weights: */
    for (int i = 0; i < n; i++)
	dist[i] = MAX_DIST;
    dist[vertex] = 0;
    for (size_t i = 1; i < graph[vertex].nedges; i++)
	dist[graph[vertex].edges[i]] = (DistType) graph[vertex].ewgts[i];

    initHeap(&H, vertex, index, dist, n);

    while (extractMax(&H, &closestVertex, index, dist)) {
	closestDist = dist[closestVertex];
	if (closestDist == MAX_DIST)
	    break;
	for (size_t i = 1; i < graph[closestVertex].nedges; i++) {
	    neighbor = graph[closestVertex].edges[i];
	    increaseKey(&H, neighbor, closestDist +
			(DistType)graph[closestVertex].ewgts[i], index, dist);
	}
	prevClosestDist = closestDist;
    }

    /* For dealing with disconnected graphs: */
    for (int i = 0; i < n; i++)
	if (dist[i] == MAX_DIST)	/* 'i' is not connected to 'vertex' */
	    dist[i] = prevClosestDist + 10;
    freeHeap(&H);
    free(index);
}

static void heapify_f(heap * h, int i, int index[], float dist[])
{
    int l, r, largest;
    while (1) {
	l = left(i);
	r = right(i);
	if (insideHeap(h, l) && greaterPriority(h, l, i, dist))
	    largest = l;
	else
	    largest = i;
	if (insideHeap(h, r) && greaterPriority(h, r, largest, dist))
	    largest = r;

	if (largest == i)
	    break;

	exchange(h, largest, i, index);
	i = largest;
    }
}

static void
initHeap_f(heap * h, int startVertex, int index[], float dist[], int n)
{
    int i, count;
    int j;			/* We cannot use an unsigned value in this loop */
    h->data = gv_calloc(n - 1, sizeof(int));
    h->heapSize = n - 1;

    for (count = 0, i = 0; i < n; i++)
	if (i != startVertex) {
	    h->data[count] = i;
	    index[i] = count;
	    count++;
	}

    for (j = (n - 1) / 2; j >= 0; j--)
	heapify_f(h, j, index, dist);
}

static bool extractMax_f(heap * h, int *max, int index[], float dist[])
{
    if (h->heapSize == 0)
	return false;

    *max = h->data[0];
    h->data[0] = h->data[h->heapSize - 1];
    index[h->data[0]] = 0;
    h->heapSize--;
    heapify_f(h, 0, index, dist);

    return true;
}

static void
increaseKey_f(heap * h, int increasedVertex, float newDist, int index[],
	      float dist[])
{
    int placeInHeap;
    int i;

    if (dist[increasedVertex] <= newDist)
	return;

    placeInHeap = index[increasedVertex];

    dist[increasedVertex] = newDist;

    i = placeInHeap;
    while (i > 0 && dist[h->data[parent(i)]] > newDist) {	/* can write here: greaterPriority(i,parent(i),dist) */
	assign(h, i, parent(i), index);
	i = parent(i);
    }
    h->data[i] = increasedVertex;
    index[increasedVertex] = i;
}

/* dijkstra_f:
 * Weighted shortest paths from vertex.
 * Assume graph is connected.
 */
void dijkstra_f(int vertex, vtx_data * graph, int n, float *dist)
{
    heap H;
    int closestVertex = 0, neighbor;
    float closestDist;
    int *index = gv_calloc(n, sizeof(int));

    /* initial distances with edge weights: */
    for (int i = 0; i < n; i++)
	dist[i] = FLT_MAX;
    dist[vertex] = 0;
    for (size_t i = 1; i < graph[vertex].nedges; i++)
	dist[graph[vertex].edges[i]] = graph[vertex].ewgts[i];

    initHeap_f(&H, vertex, index, dist, n);

    while (extractMax_f(&H, &closestVertex, index, dist)) {
	closestDist = dist[closestVertex];
	if (closestDist == FLT_MAX)
	    break;
	for (size_t i = 1; i < graph[closestVertex].nedges; i++) {
	    neighbor = graph[closestVertex].edges[i];
	    increaseKey_f(&H, neighbor, closestDist + graph[closestVertex].ewgts[i],
			  index, dist);
	}
    }

    freeHeap(&H);
    free(index);
}

// single source shortest paths that also builds terms as it goes
// mostly copied from dijkstra_f above
// returns the number of terms built
int dijkstra_sgd(graph_sgd *graph, int source, term_sgd *terms) {
    heap h;
    int *indices = gv_calloc(graph->n, sizeof(int));
    float *dists = gv_calloc(graph->n, sizeof(float));
    for (size_t i= 0; i < graph->n; i++) {
        dists[i] = FLT_MAX;
    }
    dists[source] = 0;
    for (size_t i = graph->sources[source]; i < graph->sources[source + 1];
         i++) {
        size_t target = graph->targets[i];
        dists[target] = graph->weights[i];
    }
    assert(graph->n <= INT_MAX);
    initHeap_f(&h, source, indices, dists, (int)graph->n);

    int closest = 0, offset = 0;
    while (extractMax_f(&h, &closest, indices, dists)) {
        float d = dists[closest];
        if (d == FLT_MAX) {
            break;
        }
        // if the target is fixed then always create a term as shortest paths are not calculated from there
        // if not fixed then only create a term if the target index is lower
        if (bitarray_get(graph->pinneds, closest) || closest<source) {
            terms[offset].i = source;
            terms[offset].j = closest;
            terms[offset].d = d;
            terms[offset].w = 1 / (d*d);
            offset++;
        }
        for (size_t i = graph->sources[closest]; i < graph->sources[closest + 1];
             i++) {
            size_t target = graph->targets[i];
            float weight = graph->weights[i];
            assert(target <= (size_t)INT_MAX);
            increaseKey_f(&h, (int)target, d+weight, indices, dists);
        }
    }
    freeHeap(&h);
    free(indices);
    free(dists);
    return offset;
}
