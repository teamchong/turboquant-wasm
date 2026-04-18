/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <sfdpgen/Multilevel.h>
#include <assert.h>
#include <common/arith.h>
#include <stddef.h>
#include <stdbool.h>
#include <util/alloc.h>

static const int minsize = 4;
static const double min_coarsen_factor = 0.75;

static Multilevel Multilevel_init(SparseMatrix A) {
  if (!A) return NULL;
  assert(A->m == A->n);
  Multilevel grid = gv_alloc(sizeof(struct Multilevel_struct));
  grid->level = 0;
  grid->n = A->n;
  grid->A = A;
  grid->P = NULL;
  grid->R = NULL;
  grid->next = NULL;
  grid->prev = NULL;
  grid->delete_top_level_A = false;
  return grid;
}

void Multilevel_delete(Multilevel grid){
  if (!grid) return;
  if (grid->A){
    if (grid->level == 0) {
      if (grid->delete_top_level_A) {
	SparseMatrix_delete(grid->A);
      }
    } else {
      SparseMatrix_delete(grid->A);
    }
  }
  SparseMatrix_delete(grid->P);
  SparseMatrix_delete(grid->R);
  Multilevel_delete(grid->next);
  free(grid);
}

static void maximal_independent_edge_set_heavest_edge_pernode_supernodes_first(SparseMatrix A, int **cluster, int **clusterp, int *ncluster){
  int i, ii, j, *ia, *ja, m, n, *p = NULL;
  (void)n;
  double *a, amax = 0;
  int jamax = 0;
  int *matched, nz, nz0;
  enum {MATCHED = -1};
  int  nsuper, *super = NULL, *superp = NULL;

  assert(A);
  assert(A->is_pattern_symmetric);
  ia = A->ia;
  ja = A->ja;
  m = A->m;
  n = A->n;
  assert(n == m);
  *cluster = gv_calloc(m, sizeof(int));
  *clusterp = gv_calloc(m + 1, sizeof(int));
  matched = gv_calloc(m, sizeof(int));

  for (i = 0; i < m; i++) matched[i] = i;

  assert(SparseMatrix_is_symmetric(A, false));
  assert(A->type == MATRIX_TYPE_REAL);

  SparseMatrix_decompose_to_supervariables(A, &nsuper, &super, &superp);

  *ncluster = 0;
  (*clusterp)[0] = 0;
  nz = 0;
  a = A->a;

  for (i = 0; i < nsuper; i++){
    if (superp[i+1] - superp[i] <= 1) continue;
    nz0 = (*clusterp)[*ncluster];
    for (j = superp[i]; j < superp[i+1]; j++){
      matched[super[j]] = MATCHED;
      (*cluster)[nz++] = super[j];
      if (nz - nz0 >= MAX_CLUSTER_SIZE){
	(*clusterp)[++(*ncluster)] = nz;
	nz0 = nz;
      }
    }
    if (nz > nz0) (*clusterp)[++(*ncluster)] = nz;
  }

  p = random_permutation(m);
  for (ii = 0; ii < m; ii++){
    i = p[ii];
    bool first = true;
    if (matched[i] == MATCHED) continue;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      if (matched[ja[j]] != MATCHED && matched[i] != MATCHED){
        if (first) {
          amax = a[j];
          jamax = ja[j];
          first = false;
        } else {
          if (a[j] > amax){
            amax = a[j];
            jamax = ja[j];
          }
        }
      }
    }
    if (!first){
        matched[jamax] = MATCHED;
        matched[i] = MATCHED;
        (*cluster)[nz++] = i;
        (*cluster)[nz++] = jamax;
        (*clusterp)[++(*ncluster)] = nz;
    }
  }

  for (i = 0; i < m; i++){
    if (matched[i] == i){
      (*cluster)[nz++] = i;
      (*clusterp)[++(*ncluster)] = nz;
    }
  }
  free(p);

  free(super);

  free(superp);

  free(matched);
}

static void Multilevel_coarsen_internal(SparseMatrix A, SparseMatrix *cA,
                                        SparseMatrix *P, SparseMatrix *R) {
  int nc, nzc, n, i;
  int *irn = NULL, *jcn = NULL;
  double *val = NULL;
  int j;
  int *cluster=NULL, *clusterp=NULL, ncluster;

  assert(A->m == A->n);
  *cA = NULL;
  *P = NULL;
  *R = NULL;
  n = A->m;

  maximal_independent_edge_set_heavest_edge_pernode_supernodes_first(A, &cluster, &clusterp, &ncluster);
  assert(ncluster <= n);
  nc = ncluster;
  if (nc == n || nc < minsize) {
#ifdef DEBUG_PRINT
    if (Verbose)
      fprintf(stderr, "nc = %d, nf = %d, minsz = %d, coarsen_factor = %f coarsening stops\n",nc, n, minsize, min_coarsen_factor);
#endif
    goto RETURN;
  }
  irn = gv_calloc(n, sizeof(int));
  jcn = gv_calloc(n, sizeof(int));
  val = gv_calloc(n, sizeof(double));
  nzc = 0; 
  for (i = 0; i < ncluster; i++){
    for (j = clusterp[i]; j < clusterp[i+1]; j++){
      assert(clusterp[i+1] > clusterp[i]);
      irn[nzc] = cluster[j];
      jcn[nzc] = i;
      val[nzc++] = 1.;
   }
  }
  assert(nzc == n);
  *P = SparseMatrix_from_coordinate_arrays(nzc, n, nc, irn, jcn, val,
                                           MATRIX_TYPE_REAL, sizeof(double));
  *R = SparseMatrix_transpose(*P);

  *cA = SparseMatrix_multiply3(*R, A, *P); 
  if (!*cA) goto RETURN;

  *R = SparseMatrix_divide_row_by_degree(*R);
  (*cA)->is_symmetric = true;
  (*cA)->is_pattern_symmetric = true;
  *cA = SparseMatrix_remove_diagonal(*cA);

 RETURN:
  free(irn);
  free(jcn);
  free(val);

  free(cluster);
  free(clusterp);
}

static void Multilevel_coarsen(SparseMatrix A, SparseMatrix *cA,
                               SparseMatrix *P, SparseMatrix *R) {
  SparseMatrix cA0 = A, P0 = NULL, R0 = NULL, M;
  int nc = 0, n;
  
  *P = NULL; *R = NULL; *cA = NULL;

  n = A->n;

  do {/* this loop force a sufficient reduction */
    Multilevel_coarsen_internal(A, &cA0, &P0, &R0);
    if (!cA0) return;
    nc = cA0->n;
#ifdef DEBUG_PRINT
    if (Verbose) fprintf(stderr,"nc=%d n = %d\n",nc,n);
#endif
    if (*P){
      assert(*R);
      M = SparseMatrix_multiply(*P, P0);
      SparseMatrix_delete(*P);
      SparseMatrix_delete(P0);
      *P = M;
      M = SparseMatrix_multiply(R0, *R);
      SparseMatrix_delete(*R);
      SparseMatrix_delete(R0);
      *R = M;
    } else {
      *P = P0;
      *R = R0;
    }

    if (*cA) SparseMatrix_delete(*cA);
    *cA = cA0;

    A = cA0;
  } while (nc > min_coarsen_factor*n);

}

void print_padding(int n){
  int i;
  for (i = 0; i < n; i++) fputs (" ", stderr);
}

static Multilevel Multilevel_establish(Multilevel grid,
                                       const Multilevel_control ctrl) {
  Multilevel cgrid;
  SparseMatrix P, R, A, cA;

#ifdef DEBUG_PRINT
  if (Verbose) {
    print_padding(grid->level);
    fprintf(stderr, "level -- %d, n = %d, nz = %d nz/n = %f\n", grid->level, grid->n, grid->A->nz, grid->A->nz/(double) grid->n);
  }
#endif
  A = grid->A;
  if (grid->level >= ctrl.maxlevel - 1) {
#ifdef DEBUG_PRINT
  if (Verbose) {
    print_padding(grid->level);
    fprintf(stderr, " maxlevel reached, coarsening stops\n");
  }
#endif
    return grid;
  }
  Multilevel_coarsen(A, &cA, &P, &R);
  if (!cA) return grid;

  cgrid = Multilevel_init(cA);
  grid->next = cgrid;
  cgrid->level = grid->level + 1;
  cgrid->n = cA->m;
  cgrid->P = P;
  grid->R = R;
  cgrid->prev = grid;
  cgrid = Multilevel_establish(cgrid, ctrl);
  return grid;
  
}

Multilevel Multilevel_new(SparseMatrix A0,
                          const Multilevel_control ctrl) {
  /* A: the weighting matrix. D: the distance matrix, could be NULL. If not null, the two matrices must have the same sparsity pattern */
  Multilevel grid;
  SparseMatrix A = A0;

  if (!SparseMatrix_is_symmetric(A, false) || A->type != MATRIX_TYPE_REAL){
    A = SparseMatrix_get_real_adjacency_matrix_symmetrized(A);
  }
  grid = Multilevel_init(A);
  grid = Multilevel_establish(grid, ctrl);
  if (A != A0) grid->delete_top_level_A = true; // be sure to clean up later
  return grid;
}


Multilevel Multilevel_get_coarsest(Multilevel grid){
  while (grid->next){
    grid = grid->next;
  }
  return grid;
}

