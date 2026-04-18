/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Multilevel_Modularity_Clustering_struct *Multilevel_Modularity_Clustering;

struct Multilevel_Modularity_Clustering_struct {
  int level;/* 0, 1, ... */
  int n;
  SparseMatrix A; /* n x n matrix */
  SparseMatrix P; 
  Multilevel_Modularity_Clustering next;
  Multilevel_Modularity_Clustering prev;
  bool delete_top_level_A;
  int *matching; /* dimension n. matching[i] is the clustering assignment of node i */
  double modularity;
  double deg_total; /* total edge weights, including self-edges */
  double *deg;/* dimension n. deg[i] equal to the sum of edge weights connected to vertex i. I.e., sum of  row i */
  bool agglomerate_regardless; ///< whether to agglomerate nodes even if this
                               ///< causes modularity reduction. This is used if
                               ///< we want to force agglomeration so as to get
                               ///< fewer clusters
};

enum {CLUSTERING_MODULARITY = 0, CLUSTERING_MQ};

/* find a clustering of vertices by maximize modularity
   A: symmetric square matrix n x n. If real value, value will be used as edges weights, otherwise edge weights are considered as 1.
   inplace: whether A can e modified. If true, A will be modified by removing diagonal.

   maxcluster: used to specify the maximum number of cluster desired, e.g., maxcluster=10 means that a maximum of 10 clusters
   .   is desired. this may not always be realized, and modularity may be low when this is specified. Default: maxcluster = 0 (no limit)

   nclusters: on output the number of clusters
   assignment: dimension n. Node i is assigned to cluster "assignment[i]". 0 <= assignment < nclusters.
   .   If *assignment = NULL on entry, it will be allocated. Otherwise used.
   modularity: achieve modularity
*/
void modularity_clustering(SparseMatrix A, bool inplace, int maxcluster,
			   int *nclusters, int **assignment, double *modularity);

#ifdef __cplusplus
}
#endif
