/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#define STANDALONE
#include <math.h>
#include <sparse/general.h>
#include <sparse/SparseMatrix.h>
#include <sparse/clustering.h>
#include <stdbool.h>
#include <util/alloc.h>

static Multilevel_Modularity_Clustering Multilevel_Modularity_Clustering_init(SparseMatrix A, int level){
  int n = A->n, i, j;

  assert(A->type == MATRIX_TYPE_REAL);
  assert(SparseMatrix_is_symmetric(A, false));

  if (!A) return NULL;
  assert(A->m == n);
  Multilevel_Modularity_Clustering grid = gv_alloc(sizeof(struct Multilevel_Modularity_Clustering_struct));
  grid->level = level;
  grid->n = n;
  grid->A = A;
  grid->P = NULL;
  grid->next = NULL;
  grid->prev = NULL;
  grid->delete_top_level_A = false;
  grid->matching = gv_calloc(n, sizeof(double));
  grid->deg = NULL;
  grid->agglomerate_regardless = false;

  if (level == 0){
    double modularity = 0;
    int *ia = A->ia, *ja = A->ja;
    double deg_total = 0;
    double *deg, *a = A->a;

    grid->deg_total = 0.;
    grid->deg = gv_calloc(n, sizeof(double));
    deg = grid->deg;

    double *indeg = gv_calloc(n, sizeof(double));
    for (i = 0; i < n; i++){
      deg[i] = 0;
      indeg[i] = 0.;
      for (j = ia[i]; j < ia[i+1]; j++){
	deg[i] += a[j];
	if (ja[j] == i) indeg[i] = a[j];
      }
      deg_total += deg[i];
    }
    deg_total = fmax(deg_total, 1);
    for (i = 0; i < n; i++){
      modularity += (indeg[i] - deg[i]*deg[i]/deg_total)/deg_total;
    }
    grid->deg_total = deg_total;
    grid->deg = deg;
    grid->modularity = modularity;
    free(indeg);
  }


  return grid;
} 

static void Multilevel_Modularity_Clustering_delete(Multilevel_Modularity_Clustering grid){
  if (!grid) return;
  if (grid->A){
    if (grid->level == 0) {
      if (grid->delete_top_level_A) SparseMatrix_delete(grid->A);
    } else {
      SparseMatrix_delete(grid->A);
    }
  }
  SparseMatrix_delete(grid->P);
  free(grid->matching);
  free(grid->deg);

  Multilevel_Modularity_Clustering_delete(grid->next);
  free(grid);
}

static Multilevel_Modularity_Clustering Multilevel_Modularity_Clustering_establish(Multilevel_Modularity_Clustering grid, int ncluster_target){
  int *matching = grid->matching;
  SparseMatrix A = grid->A;
  int n = grid->n, level = grid->level, nc = 0;
  double modularity = 0;
  int *ia = A->ia, *ja = A->ja;
  double *deg = grid->deg;
  int i, j, jj, jc, jmax;
  double inv_deg_total = 1./ grid->deg_total;
  double gain;
  double maxgain;
  double total_gain = 0;
  modularity = grid->modularity;

  double *deg_new = gv_calloc(n, sizeof(double));
  double *deg_inter = gv_calloc(n, sizeof(double));
  int *mask = gv_calloc(n, sizeof(int));
  for (i = 0; i < n; i++) mask[i] = -1;

  assert(n == A->n);
  for (i = 0; i < n; i++) matching[i] = UNMATCHED;

  /* gain in merging node i into cluster j is
     deg(i,j)/deg_total - 2*deg(i)*deg(j)/deg_total^2
  */
  double *a = A->a;
  for (i = 0; i < n; i++){
    if (matching[i] != UNMATCHED) continue;
    /* accumulate connections between i and clusters */
    for (j = ia[i]; j < ia[i+1]; j++){
      jj = ja[j];
      if (jj == i) continue;
      if ((jc=matching[jj]) != UNMATCHED){
	if (mask[jc] != i) {
	  mask[jc] = i;
	  deg_inter[jc] = a[j];
	} else {
	  deg_inter[jc] += a[j];
	}
      }
    }

    maxgain = 0;
    jmax = -1;
    for (j = ia[i]; j < ia[i+1]; j++){
      jj = ja[j];
      if (jj == i) continue;
      if ((jc=matching[jj]) == UNMATCHED){
	/* the first 2 is due to the fact that deg_iter gives edge weight from i to jj, but there are also edges from jj to i */
	gain = (2*a[j] - 2*deg[i]*deg[jj]*inv_deg_total)*inv_deg_total;
      } else {
	if (deg_inter[jc] > 0){
	  /* the first 2 is due to the fact that deg_iter gives edge weight from i to jc, but there are also edges from jc to i */
	  gain = (2*deg_inter[jc] - 2*deg[i]*deg_new[jc]*inv_deg_total)*inv_deg_total;
	  //	  printf("mod = %f deg_inter[jc] =%f, deg[i] = %f, deg_new[jc]=%f, gain = %f\n",modularity, deg_inter[jc],deg[i],deg_new[jc],gain);
	  deg_inter[jc] = -1; /* so that we do not redo the calulation when we hit another neighbor in cluster jc */
	} else {
	  gain = -1;
	}
      }
      if (jmax < 0 || gain > maxgain){
	maxgain = gain;
        jmax = jj;
      } 

    }

    /* now merge i and jmax */
    if (maxgain > 0 || grid->agglomerate_regardless){
      total_gain += maxgain;
      jc = matching[jmax];
      if (jc == UNMATCHED){
	//fprintf(stderr, "maxgain=%f, merge %d, %d\n",maxgain, i, jmax);
	matching[i] = matching[jmax] = nc;
	deg_new[nc] = deg[i] + deg[jmax];
	nc++;
      } else {
	//fprintf(stderr, "maxgain=%f, merge with existing cluster %d, %d\n",maxgain, i, jc);
	deg_new[jc] += deg[i];
	matching[i] = jc;
      }
    } else {
      assert(maxgain <= 0);
      matching[i] = nc;
      deg_new[nc] = deg[i];
      nc++;
    }

  }

  if (Verbose) fprintf(stderr,"modularity = %f new modularity = %f level = %d, n = %d, nc = %d, gain = %g\n", modularity, modularity + total_gain, 
		       level, n, nc, total_gain);

  if (ncluster_target > 0){
    if (nc <= ncluster_target && n >= ncluster_target){
      if (n - ncluster_target > ncluster_target - nc){/* ncluster = nc */

      } else if (n - ncluster_target <= ncluster_target - nc){/* ncluster_target close to n */
	fprintf(stderr,"ncluster_target = %d, close to n=%d\n", ncluster_target, n);
	for (i = 0; i < n; i++) matching[i] = i;
	free(deg_new);
	goto RETURN;
      }
    } else if (n < ncluster_target){
      fprintf(stderr,"n < target\n");
      for (i = 0; i < n; i++) matching[i] = i;
      free(deg_new);
      goto RETURN;
    }
  }

  if (nc >= 1 && (total_gain > 0 || nc < n)){
    /* now set up restriction and prolongation operator */
    SparseMatrix P, R, R0, B, cA;
    double one = 1.;
    Multilevel_Modularity_Clustering cgrid;

    R0 = SparseMatrix_new(nc, n, 1, MATRIX_TYPE_REAL, FORMAT_COORD);
    for (i = 0; i < n; i++){
      jj = matching[i];
      SparseMatrix_coordinate_form_add_entry(R0, jj, i, &one);
    }
    R = SparseMatrix_from_coordinate_format(R0);
    SparseMatrix_delete(R0);
    P = SparseMatrix_transpose(R);
    B = SparseMatrix_multiply(R, A);
    SparseMatrix_delete(R);
    if (!B) {
      free(deg_new);
      goto RETURN;
    }
    cA = SparseMatrix_multiply(B, P); 
    SparseMatrix_delete(B);
    if (!cA) {
      free(deg_new);
      goto RETURN;
    }
    grid->P = P;
    level++;
    cgrid = Multilevel_Modularity_Clustering_init(cA, level); 
    cgrid->deg = deg_new;
    cgrid->modularity = grid->modularity + total_gain;
    cgrid->deg_total = grid->deg_total;
    cgrid = Multilevel_Modularity_Clustering_establish(cgrid, ncluster_target);
    grid->next = cgrid;
    cgrid->prev = grid;
  } else {
    /* if we want a small number of cluster but right now we have too many, we will force agglomeration */
    if (ncluster_target > 0 && nc > ncluster_target && !(grid->agglomerate_regardless)){
      grid->agglomerate_regardless = true;
      free(deg_inter);
      free(mask);
      free(deg_new);
      return Multilevel_Modularity_Clustering_establish(grid, ncluster_target);
    }
    /* no more improvement, stop and final clustering found */
    for (i = 0; i < n; i++) matching[i] = i;
    free(deg_new);
  }

 RETURN:
  free(deg_inter);
  free(mask);
  return grid;
}

static Multilevel_Modularity_Clustering Multilevel_Modularity_Clustering_new(SparseMatrix A0, int ncluster_target){
  /* ncluster_target is used to specify the target number of cluster desired, e.g., ncluster_target=10 means that around 10 clusters
     is desired. The resulting clustering will give as close to this number as possible.
     If this number != the optimal number of clusters, the resulting modularity may be lower, or equal to, the optimal modularity.
     .  Agglomeration will be forced even if that reduces the modularity when there are too many clusters. It will stop when nc <= ncluster_target <= nc2,
     .  where nc and nc2 are the number of clusters in the current and next level of clustering. The final cluster number will be
     .  selected among nc or nc2, which ever is closer to ncluster_target.
     Default: ncluster_target <= 0 */

  Multilevel_Modularity_Clustering grid;
  SparseMatrix A = A0;

  if (!SparseMatrix_is_symmetric(A, false) || A->type != MATRIX_TYPE_REAL){
    A = SparseMatrix_get_real_adjacency_matrix_symmetrized(A);
  }
  grid = Multilevel_Modularity_Clustering_init(A, 0);

  grid = Multilevel_Modularity_Clustering_establish(grid, ncluster_target);

  if (A != A0) grid->delete_top_level_A = true; // be sure to clean up later
  return grid;
}


static void hierachical_modularity_clustering(SparseMatrix A, int ncluster_target,
					      int *nclusters, int **assignment, double *modularity){
  /* find a clustering of vertices by maximize modularity
     A: symmetric square matrix n x n. If real value, value will be used as edges weights, otherwise edge weights are considered as 1.

     ncluster_target: is used to specify the target number of cluster desired, e.g., ncluster_target=10 means that around 10 clusters 
     is desired. The resulting clustering will give as close to this number as possible.                                                                                                 
     If this number != the optimal number of clusters, the resulting modularity may be lower, or equal to, the optimal modularity.                                                      
     .  Agglomeration will be forced even if that reduces the modularity when there are too many clusters. It will stop when nc <= ncluster_target <= nc2,                                   
     .  where nc and nc2 are the number of clusters in the current and next level of clustering. The final cluster number will be                                                     
     .  selected among nc or nc2, which ever is closer to ncluster_target.                                                                                                                  
     Default: ncluster_target <= 0 

     nclusters: on output the number of clusters
     assignment: dimension n. Node i is assigned to cluster "assignment[i]". 0 <= assignment < nclusters
   */

  Multilevel_Modularity_Clustering grid, cgrid;
  int *matching, i;
  SparseMatrix P;
  assert(A->m == A->n);

  *modularity = 0.;

  grid = Multilevel_Modularity_Clustering_new(A, ncluster_target);

  /* find coarsest */
  cgrid = grid;
  while (cgrid->next){
    cgrid = cgrid->next;
  }

  /* project clustering up */
  double *u = gv_calloc(cgrid->n, sizeof(double));
  for (i = 0; i < cgrid->n; i++) u[i] = (double) (cgrid->matching)[i];
  *nclusters = cgrid->n;
  *modularity = cgrid->modularity;

  while (cgrid->prev){
    double *v = NULL;
    P = cgrid->prev->P;
    SparseMatrix_multiply_vector(P, u, &v);
    free(u);
    u = v;
    cgrid = cgrid->prev;
  }

  if (*assignment){
    matching = *assignment; 
  } else {
    matching = gv_calloc(grid->n, sizeof(int));
    *assignment = matching;
  }
  for (i = 0; i < grid->n; i++) (matching)[i] = (int) u[i];
  free(u);

  Multilevel_Modularity_Clustering_delete(grid);
}



void modularity_clustering(SparseMatrix A, bool inplace, int ncluster_target,
			   int *nclusters, int **assignment, double *modularity){
  /* find a clustering of vertices by maximize modularity
     A: symmetric square matrix n x n. If real value, value will be used as edges weights, otherwise edge weights are considered as 1.
     inplace: whether A can e modified. If true, A will be modified by removing diagonal.
     ncluster_target: is used to specify the target number of cluster desired, e.g., ncluster_target=10 means that around 10 clusters
     is desired. The resulting clustering will give as close to this number as possible.
     If this number != the optimal number of clusters, the resulting modularity may be lower, or equal to, the optimal modularity.
     .  Agglomeration will be forced even if that reduces the modularity when there are too many clusters. It will stop when nc <= ncluster_target <= nc2,
     .  where nc and nc2 are the number of clusters in the current and next level of clustering. The final cluster number will be
     .  selected among nc or nc2, which ever is closer to ncluster_target.
     Default: ncluster_target <= 0 
     nclusters: on output the number of clusters
     assignment: dimension n. Node i is assigned to cluster "assignment[i]". 0 <= assignment < nclusters
   */
  SparseMatrix B;

  assert(A->m == A->n);

  B = SparseMatrix_symmetrize(A, false);

  if (!inplace && B == A) {
    B = SparseMatrix_copy(A);
  }

  B = SparseMatrix_remove_diagonal(B);

  if (B->type != MATRIX_TYPE_REAL) B = SparseMatrix_set_entries_to_real_one(B);

  hierachical_modularity_clustering(B, ncluster_target, nclusters, assignment, modularity);

  if (B != A) SparseMatrix_delete(B);

}
