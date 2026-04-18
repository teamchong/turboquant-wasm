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

typedef struct Multilevel_MQ_Clustering_struct *Multilevel_MQ_Clustering;

struct Multilevel_MQ_Clustering_struct {
  int level;/* 0, 1, ... */
  int n;
  SparseMatrix A; /* n x n matrix */
  SparseMatrix P; 
  Multilevel_MQ_Clustering next;
  Multilevel_MQ_Clustering prev;
  bool delete_top_level_A;
  int *matching; /* dimension n. matching[i] is the clustering assignment of node i */

  /*

   .                                 |E(i,i)|                                     |E(i,j)|
   MQ/2 = (1/k) * \sum_{i=1...k} ------------ - (1/(k*(k-1))) * \sum_{i<j} -------------------  
   .                                |V(i)|^2                                   |V(i)|*|V(j)|
   .   = mq_in/k - mq_out/(k*(k-1))
  */

  double mq;
  double mq_in, mq_out;/* mqs(A) = deg_in(A)/|A|^2 - deg_out(A)/|A|/(|V|-|A|) */
  int ncluster; /* number of clusters */

  double *deg_intra;/* dimension n. deg[i] equal to the sum of edge weights within cluster i */
  double *dout;/* dimension n, dout[i] = \sum_{j -- i} a(i,j)/|j| is the scaled sum of outdegree */
  double *wgt; /* total vertex weight each coarse grid vertex represent */
};

/* find a clustering of vertices by maximize modularity quality
   A: symmetric square matrix n x n. If real value, value will be used as edges weights, otherwise edge weights are considered as 1.

   maxcluster: used to specify the maximum number of cluster desired, e.g., maxcluster=10 means that a maximum of 10 clusters
   .   is desired. this may not always be realized, and modularity quality may be low when this is specified. Default: maxcluster = 0 (no limit)

   nclusters: on output the number of clusters
   assignment: dimension n. Node i is assigned to cluster "assignment[i]". 0 <= assignment < nclusters.
   .   If *assignment = NULL on entry, it will be allocated. Otherwise used.
   mq: achieve modularity
*/
void mq_clustering(SparseMatrix A, int maxcluster,
			   int *nclusters, int **assignment, double *mq);

#ifdef __cplusplus
}
#endif
