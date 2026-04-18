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

	Breadth First Search
	Computes single-source distances for
	unweighted graphs

******************************************/

#include <neatogen/bfs.h>
#include <stdbool.h>
#include <stdlib.h>
#include <util/alloc.h>

void bfs(int vertex, vtx_data *graph, int n, DistType *dist)
 /* compute vector 'dist' of distances of all nodes from 'vertex' */
{
    int closestVertex, neighbor;
    DistType closestDist = INT_MAX;

    /* initial distances with edge weights: */
    for (int i = 0; i < n; i++)
	dist[i] = -1;
    dist[vertex] = 0;

    Queue Q;
    mkQueue(&Q, n);
    initQueue(&Q, vertex);

    if (graph[0].ewgts == NULL) {
	while (deQueue(&Q, &closestVertex)) {
	    closestDist = dist[closestVertex];
	    for (size_t i = 1; i < graph[closestVertex].nedges; i++) {
		neighbor = graph[closestVertex].edges[i];
		if (dist[neighbor] < -0.5) {	/* first time to reach neighbor */
		    dist[neighbor] = closestDist + 1;
		    enQueue(&Q, neighbor);
		}
	    }
	}
    } else {
	while (deQueue(&Q, &closestVertex)) {
	    closestDist = dist[closestVertex];
	    for (size_t i = 1; i < graph[closestVertex].nedges; i++) {
		neighbor = graph[closestVertex].edges[i];
		if (dist[neighbor] < -0.5) {	/* first time to reach neighbor */
		    dist[neighbor] =
			closestDist +
			(DistType) graph[closestVertex].ewgts[i];
		    enQueue(&Q, neighbor);
		}
	    }
	}
    }

    /* For dealing with disconnected graphs: */
    for (int i = 0; i < n; i++)
	if (dist[i] < -0.5)	/* 'i' is not connected to 'vertex' */
	    dist[i] = closestDist + 10;

    freeQueue(&Q);
}

void mkQueue(Queue * qp, int size)
{
    qp->data = gv_calloc(size, sizeof(int));
    qp->queueSize = size;
    qp->start = qp->end = 0;
}

void freeQueue(Queue * qp)
{
    free(qp->data);
}

void initQueue(Queue * qp, int startVertex)
{
    qp->data[0] = startVertex;
    qp->start = 0;
    qp->end = 1;
}

bool deQueue(Queue * qp, int *vertex)
{
    if (qp->start >= qp->end)
	return false;		/* underflow */
    *vertex = qp->data[qp->start++];
    return true;
}

bool enQueue(Queue * qp, int vertex)
{
    if (qp->end >= qp->queueSize)
	return false;		/* overflow */
    qp->data[qp->end++] = vertex;
    return true;
}
