/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/


/************************************************

	Functions for computing the high-dimensional
	embedding and the PCA projection.

************************************************/

#include <neatogen/dijkstra.h>
#include <neatogen/bfs.h>
#include <neatogen/kkutils.h>
#include <neatogen/embed_graph.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <util/alloc.h>

void embed_graph(vtx_data * graph, int n, int dim, DistType *** Coords,
		 int reweight_graph)
{
/* Compute 'dim'-dimensional high-dimensional embedding (HDE) for the 'n' nodes
  The embedding is based on choosing 'dim' pivots, and associating each
  coordinate with a unique pivot, assigning it to the graph-theoretic distances 
  of all nodes from the pivots
*/

    int i, j;
    int node;
    DistType *storage = gv_calloc(n * dim, sizeof(DistType));
    DistType **coords = *Coords;
    DistType *dist = gv_calloc(n, sizeof(DistType)); // this vector stores  the
                                                     // distances of each nodes
                                                     // to the selected “pivots”
    float *old_weights = graph[0].ewgts;
    DistType max_dist = 0;

    /* this matrix stores the distance between each node and each "pivot" */
    *Coords = coords = gv_calloc(dim, sizeof(DistType *));
    for (i = 0; i < dim; i++)
	coords[i] = storage + i * n;

    if (reweight_graph) {
	compute_new_weights(graph, n);
    }

    /* select the first pivot */
    node = rand() % n;

    if (reweight_graph) {
	dijkstra(node, graph, n, coords[0]);
    } else {
	bfs(node, graph, n, coords[0]);
    }

    for (i = 0; i < n; i++) {
	dist[i] = coords[0][i];
	if (dist[i] > max_dist) {
	    node = i;
	    max_dist = dist[i];
	}
    }

    /* select other dim-1 nodes as pivots */
    for (i = 1; i < dim; i++) {
	if (reweight_graph) {
	    dijkstra(node, graph, n, coords[i]);
	} else {
	    bfs(node, graph, n, coords[i]);
	}
	max_dist = 0;
	for (j = 0; j < n; j++) {
	    dist[j] = MIN(dist[j], coords[i][j]);
	    if (dist[j] > max_dist) {
		node = j;
		max_dist = dist[j];
	    }
	}

    }

    free(dist);

    if (reweight_graph) {
	restore_old_weights(graph, n, old_weights);
    }

}

 /* Make each axis centered around 0 */
void center_coordinate(DistType ** coords, int n, int dim)
{
    int i, j;
    double sum, avg;
    for (i = 0; i < dim; i++) {
	sum = 0;
	for (j = 0; j < n; j++) {
	    sum += coords[i][j];
	}
	avg = sum / n;
	for (j = 0; j < n; j++) {
	    coords[i][j] -= (DistType) avg;
	}
    }
}
