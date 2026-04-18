/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include "config.h"

#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <common/types.h>
#include <common/globals.h>

#include <sfdpgen/sparse_solve.h>
#include <sfdpgen/post_process.h>
#include <neatogen/overlap.h>
#include <sfdpgen/spring_electrical.h>
#include <neatogen/call_tri.h>
#include <sfdpgen/sfdp.h>
#include <util/alloc.h>
#include <util/exit.h>
#include <util/unused.h>

#define node_degree(i) (ia[(i)+1] - ia[(i)])

static SparseMatrix ideal_distance_matrix(SparseMatrix A, int dim, double *x){
  /* find the ideal distance between edges, either 1, or |N[i] \Union N[j]| - |N[i] \Intersection N[j]|
   */
  SparseMatrix D;
  int *ia, *ja, i, j, k, l, nz;
  double *d;
  double len, di, sum, sumd;

  assert(SparseMatrix_is_symmetric(A, false));

  D = SparseMatrix_copy(A);
  ia = D->ia;
  ja = D->ja;
  if (D->type != MATRIX_TYPE_REAL){
    free(D->a);
    D->type = MATRIX_TYPE_REAL;
    D->a = gv_calloc(D->nz, sizeof(double));
  }
  d = D->a;

  int *mask = gv_calloc(D->m, sizeof(int));
  for (i = 0; i < D->m; i++) mask[i] = -1;

  for (i = 0; i < D->m; i++){
    di = node_degree(i);
    mask[i] = i;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      mask[ja[j]] = i;
    }
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      if (i == k) continue;
      len = di + node_degree(k);
      for (l = ia[k]; l < ia[k+1]; l++){
	if (mask[ja[l]] == i) len--;
      }
      d[j] = len;
      assert(len > 0);
    }
    
  }

  sum = 0; sumd = 0;
  nz = 0;
  for (i = 0; i < D->m; i++){
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      nz++;
      sum += distance(x, dim, i, ja[j]);
      sumd += d[j];
    }
  }
  sum /= nz; sumd /= nz;
  sum = sum/sumd;

  for (i = 0; i < D->m; i++){
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      d[j] = sum*d[j];
    }
  }

  free(mask);
  return D;
}


StressMajorizationSmoother StressMajorizationSmoother2_new(SparseMatrix A, int dim, double lambda0, double *x, 
							  int ideal_dist_scheme){
  /* use up to dist 2 neighbor. This is used in overcoming pherical effect with ideal distance of
     2-neighbors equal graph distance etc.
   */
  int i, j, k, l, m = A->m, *ia = A->ia, *ja = A->ja, *iw, *jw, *id, *jd;
  int nz;
  double *d, *w, *lambda;
  double diag_d, diag_w, dist, s = 0, stop = 0, sbot = 0;
  SparseMatrix ID;

  assert(SparseMatrix_is_symmetric(A, false));

  ID = ideal_distance_matrix(A, dim, x);

  StressMajorizationSmoother sm = gv_alloc(sizeof(struct StressMajorizationSmoother_struct));
  sm->scaling = 1.;
  sm->data = NULL;
  sm->scheme = SM_SCHEME_NORMAL;
  sm->tol_cg = 0.01;
  sm->maxit_cg = floor(sqrt(A->m));

  lambda = sm->lambda = gv_calloc(m, sizeof(double));
  for (i = 0; i < m; i++) sm->lambda[i] = lambda0;
  int *mask = gv_calloc(m, sizeof(int));

  double *avg_dist = gv_calloc(m, sizeof(double));

  for (i = 0; i < m ;i++){
    avg_dist[i] = 0;
    nz = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      avg_dist[i] += distance(x, dim, i, ja[j]);
      nz++;
    }
    assert(nz > 0);
    avg_dist[i] /= nz;
  }


  for (i = 0; i < m; i++) mask[i] = -1;

  nz = 0;
  for (i = 0; i < m; i++){
    mask[i] = i;
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      if (mask[k] != i){
	mask[k] = i;
	nz++;
      }
    }
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      for (l = ia[k]; l < ia[k+1]; l++){
	if (mask[ja[l]] != i){
	  mask[ja[l]] = i;
	  nz++;
	}
      }
    }
  }

  sm->Lw = SparseMatrix_new(m, m, nz + m, MATRIX_TYPE_REAL, FORMAT_CSR);
  sm->Lwd = SparseMatrix_new(m, m, nz + m, MATRIX_TYPE_REAL, FORMAT_CSR);
  if (!(sm->Lw) || !(sm->Lwd)) {
    StressMajorizationSmoother_delete(sm);
    return NULL;
  }

  iw = sm->Lw->ia; jw = sm->Lw->ja;

  w = sm->Lw->a;
  d = sm->Lwd->a;

  id = sm->Lwd->ia; jd = sm->Lwd->ja;
  iw[0] = id[0] = 0;

  nz = 0;
  for (i = 0; i < m; i++){
    mask[i] = i+m;
    diag_d = diag_w = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      if (mask[k] != i+m){
	mask[k] = i+m;

	jw[nz] = k;
	if (ideal_dist_scheme == IDEAL_GRAPH_DIST){
	  dist = 1;
	} else if (ideal_dist_scheme == IDEAL_AVG_DIST){
	  dist = (avg_dist[i] + avg_dist[k])*0.5;
	} else if (ideal_dist_scheme == IDEAL_POWER_DIST){
	  dist = pow(distance_cropped(x,dim,i,k),.4);
	} else {
	  fprintf(stderr,"ideal_dist_scheme value wrong");
	  assert(0);
	  graphviz_exit(1);
	}

	/*	
	  w[nz] = -1./(ia[i+1]-ia[i]+ia[ja[j]+1]-ia[ja[j]]);
	  w[nz] = -2./(avg_dist[i]+avg_dist[k]);*/
	/* w[nz] = -1.;*//* use unit weight for now, later can try 1/(deg(i)+deg(k)) */
	w[nz] = -1/(dist*dist);

	diag_w += w[nz];

	jd[nz] = k;
	/*
	  d[nz] = w[nz]*distance(x,dim,i,k);
	  d[nz] = w[nz]*dd[j];
	  d[nz] = w[nz]*(avg_dist[i] + avg_dist[k])*0.5;
	*/
	d[nz] = w[nz]*dist;
	stop += d[nz]*distance(x,dim,i,k);
	sbot += d[nz]*dist;
	diag_d += d[nz];

	nz++;
      }
    }

    /* distance 2 neighbors */
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      for (l = ia[k]; l < ia[k+1]; l++){
	if (mask[ja[l]] != i+m){
	  mask[ja[l]] = i+m;

	  if (ideal_dist_scheme == IDEAL_GRAPH_DIST){
	    dist = 2;
	  } else if (ideal_dist_scheme == IDEAL_AVG_DIST){
	    dist = (avg_dist[i] + 2*avg_dist[k] + avg_dist[ja[l]])*0.5;
	  } else if (ideal_dist_scheme == IDEAL_POWER_DIST){
	    dist = pow(distance_cropped(x,dim,i,ja[l]),.4);
	  } else {
	    fprintf(stderr,"ideal_dist_scheme value wrong");
	    assert(0);
	    graphviz_exit(1);
	  }

	  jw[nz] = ja[l];
	  /*
	    w[nz] = -1/(ia[i+1]-ia[i]+ia[ja[l]+1]-ia[ja[l]]);
	    w[nz] = -2/(avg_dist[i] + 2*avg_dist[k] + avg_dist[ja[l]]);*/
	  /* w[nz] = -1.;*//* use unit weight for now, later can try 1/(deg(i)+deg(k)) */

	  w[nz] = -1/(dist*dist);

	  diag_w += w[nz];

	  jd[nz] = ja[l];
	  /*
	    d[nz] = w[nz]*(distance(x,dim,i,k)+distance(x,dim,k,ja[l]));
	    d[nz] = w[nz]*(dd[j]+dd[l]);
	    d[nz] = w[nz]*(avg_dist[i] + 2*avg_dist[k] + avg_dist[ja[l]])*0.5;
	  */
	  d[nz] = w[nz]*dist;
	  stop += d[nz]*distance(x,dim,ja[l],k);
	  sbot += d[nz]*dist;
	  diag_d += d[nz];

	  nz++;
	}
      }
    }
    jw[nz] = i;
    lambda[i] *= (-diag_w);/* alternatively don't do that then we have a constant penalty term scaled by lambda0 */

    w[nz] = -diag_w + lambda[i];
    jd[nz] = i;
    d[nz] = -diag_d;
    nz++;

    iw[i+1] = nz;
    id[i+1] = nz;
  }
  s = stop/sbot;
  for (i = 0; i < nz; i++) d[i] *= s;

  sm->scaling = s;
  sm->Lw->nz = nz;
  sm->Lwd->nz = nz;

  free(mask);
  free(avg_dist);
  SparseMatrix_delete(ID);
  return sm;
}
	
StressMajorizationSmoother SparseStressMajorizationSmoother_new(SparseMatrix A, int dim, double *x) {
  /* solve a stress model to achieve the ideal distance among a sparse set of edges recorded in A.
     A must be a real matrix.
   */
  int i, j, k, m = A->m, *ia, *ja, *iw, *jw, *id, *jd;
  int nz;
  double *d, *w, *lambda;
  double diag_d, diag_w, *a, dist, s = 0, stop = 0, sbot = 0;
  double xdot = 0;

  assert(SparseMatrix_is_symmetric(A, false) && A->type == MATRIX_TYPE_REAL);

  /* if x is all zero, make it random */
  for (i = 0; i < m*dim; i++) xdot += x[i]*x[i];
  if (xdot == 0){
    for (i = 0; i < m*dim; i++) x[i] = 72*drand();
  }

  ia = A->ia;
  ja = A->ja;
  a = A->a;

  StressMajorizationSmoother sm = gv_alloc(sizeof(struct StressMajorizationSmoother_struct));
  sm->scaling = 1.;
  sm->data = NULL;
  sm->scheme = SM_SCHEME_NORMAL;
  sm->D = A;
  sm->tol_cg = 0.01;
  sm->maxit_cg = floor(sqrt(A->m));

  lambda = sm->lambda = gv_calloc(m, sizeof(double));

  nz = A->nz;

  sm->Lw = SparseMatrix_new(m, m, nz + m, MATRIX_TYPE_REAL, FORMAT_CSR);
  sm->Lwd = SparseMatrix_new(m, m, nz + m, MATRIX_TYPE_REAL, FORMAT_CSR);
  if (!(sm->Lw) || !(sm->Lwd)) {
    StressMajorizationSmoother_delete(sm);
    return NULL;
  }

  iw = sm->Lw->ia; jw = sm->Lw->ja;
  id = sm->Lwd->ia; jd = sm->Lwd->ja;
  w = sm->Lw->a;
  d = sm->Lwd->a;
  iw[0] = id[0] = 0;

  nz = 0;
  for (i = 0; i < m; i++){
    diag_d = diag_w = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      if (k != i){

	jw[nz] = k;
	dist = a[j];
	w[nz] = -1;
	diag_w += w[nz];
	jd[nz] = k;
	d[nz] = w[nz]*dist;

	stop += d[nz]*distance(x,dim,i,k);
	sbot += d[nz]*dist;
	diag_d += d[nz];

	nz++;
      }
    }

    jw[nz] = i;
    lambda[i] *= (-diag_w);/* alternatively don't do that then we have a constant penalty term scaled by lambda0 */
    w[nz] = -diag_w + lambda[i];

    jd[nz] = i;
    d[nz] = -diag_d;
    nz++;

    iw[i+1] = nz;
    id[i+1] = nz;
  }
  s = stop/sbot;
  if (s == 0) {
    StressMajorizationSmoother_delete(sm);
    return NULL;
  }
  for (i = 0; i < nz; i++) d[i] *= s;


  sm->scaling = s;
  sm->Lw->nz = nz;
  sm->Lwd->nz = nz;

  return sm;
}

static double total_distance(int m, int dim, double* x, double* y){
  double total = 0, dist = 0;
  int i, j;

  for (i = 0; i < m; i++){
    dist = 0.;
    for (j = 0; j < dim; j++){
      dist += (y[i*dim+j] - x[i*dim+j])*(y[i*dim+j] - x[i*dim+j]);
    }
    total += sqrt(dist);
  }
  return total;

}



void SparseStressMajorizationSmoother_delete(SparseStressMajorizationSmoother sm){
  StressMajorizationSmoother_delete(sm);
}


double SparseStressMajorizationSmoother_smooth(SparseStressMajorizationSmoother sm, int dim, double *x, int maxit_sm){
  return StressMajorizationSmoother_smooth(sm, dim, x, maxit_sm);
}

static void get_edge_label_matrix(relative_position_constraints data, int m, int dim, double *x, SparseMatrix *LL, double **rhs){
  int edge_labeling_scheme = data->edge_labeling_scheme;
  int n_constr_nodes = data->n_constr_nodes;
  int *constr_nodes = data->constr_nodes;
  SparseMatrix A_constr = data->A_constr;
  int *ia = A_constr->ia, *ja = A_constr->ja, ii, jj, nz, l, ll, i, j;
  int *irn = data->irn, *jcn = data->jcn;
  double *val = data->val, dist, kk, k;
  double *x00 = NULL;
  SparseMatrix Lc = NULL;
  double constr_penalty = data->constr_penalty;

  if (edge_labeling_scheme == ELSCHEME_PENALTY || edge_labeling_scheme == ELSCHEME_STRAIGHTLINE_PENALTY){
    /* for an node with two neighbors j--i--k, and assume i needs to be between j and k, then the contribution to P is
       .   i    j     k
       i   1   -0.5  -0.5
       j  -0.5 0.25  0.25
       k  -0.5 0.25  0.25
       in general, if i has m neighbors j, k, ..., then
       p_ii = 1
       p_jj = 1/m^2
       p_ij = -1/m
       p jk = 1/m^2
       .    i     j    k
       i    1    -1/m  -1/m ...
       j   -1/m  1/m^2 1/m^2 ...
       k   -1/m  1/m^2 1/m^2 ...
    */
    if (!irn){
      assert((!jcn) && (!val));
      nz = 0;
      for (i = 0; i < n_constr_nodes; i++){
	ii = constr_nodes[i];
	k = ia[ii+1] - ia[ii];/*usually k = 2 */
	nz += (int)((k+1)*(k+1));
	
      }
      irn = data->irn = gv_calloc(nz, sizeof(int));
      jcn = data->jcn = gv_calloc(nz, sizeof(int));
      val = data->val = gv_calloc(nz, sizeof(double));
    }
    nz = 0;
    for (i = 0; i < n_constr_nodes; i++){
      ii = constr_nodes[i];
      jj = ja[ia[ii]]; ll = ja[ia[ii] + 1];
      if (jj == ll) continue; /* do not do loops */
      dist = distance_cropped(x, dim, jj, ll);
      dist *= dist;

      k = ia[ii+1] - ia[ii];/* usually k = 2 */
      kk = k*k;
      irn[nz] = ii; jcn[nz] = ii; val[nz++] = constr_penalty/(dist);
      k = constr_penalty/(k*dist); kk = constr_penalty/(kk*dist);
      for (j = ia[ii]; j < ia[ii+1]; j++){
	irn[nz] = ii; jcn[nz] = ja[j]; val[nz++] = -k;
      }
      for (j = ia[ii]; j < ia[ii+1]; j++){
	jj = ja[j];
	irn[nz] = jj; jcn[nz] = ii; val[nz++] = -k;
	for (l = ia[ii]; l < ia[ii+1]; l++){
	  ll = ja[l];
	  irn[nz] = jj; jcn[nz] = ll; val[nz++] = kk;
	}
      }
    }
    Lc = SparseMatrix_from_coordinate_arrays(nz, m, m, irn, jcn, val, MATRIX_TYPE_REAL, sizeof(double));
  } else if (edge_labeling_scheme == ELSCHEME_PENALTY2 || edge_labeling_scheme == ELSCHEME_STRAIGHTLINE_PENALTY2){
    /* for an node with two neighbors j--i--k, and assume i needs to be between the old position of j and k, then the contribution to P is
       1/d_jk, and to the right hand side: {0,...,average_position_of_i's neighbor if i is an edge node,...}
    */
    if (!irn){
      assert((!jcn) && (!val));
      nz = n_constr_nodes;
      irn = data->irn = gv_calloc(nz, sizeof(int));
      jcn = data->jcn = gv_calloc(nz, sizeof(int));
      val = data->val = gv_calloc(nz, sizeof(double));
    }
    x00 = gv_calloc(m * dim, sizeof(double));
    nz = 0;
    for (i = 0; i < n_constr_nodes; i++){
      ii = constr_nodes[i];
      jj = ja[ia[ii]]; ll = ja[ia[ii] + 1];
      dist = distance_cropped(x, dim, jj, ll);
      irn[nz] = ii; jcn[nz] = ii; val[nz++] = constr_penalty/(dist);
      for (j = ia[ii]; j < ia[ii+1]; j++){
	jj = ja[j];
	for (l = 0; l < dim; l++){
	  x00[ii*dim+l] += x[jj*dim+l];
	}
      }
      for (l = 0; l < dim; l++) {
	x00[ii*dim+l] *= constr_penalty/(dist)/(ia[ii+1] - ia[ii]);
      }
    }
    Lc = SparseMatrix_from_coordinate_arrays(nz, m, m, irn, jcn, val, MATRIX_TYPE_REAL, sizeof(double));
    
  }
  *LL = Lc;
  *rhs = x00;
}

static UNUSED double get_stress(int m, int dim, int *iw, int *jw, double *w,
                                double *d, double *x, double scaling) {
  int i, j;
  double res = 0., dist;
  /* we use the fact that d_ij = w_ij*graph_dist(i,j). Also, d_ij and x are scalinged by *scaling, so divide by it to get actual unscaled streee. */
  for (i = 0; i < m; i++){
    for (j = iw[i]; j < iw[i+1]; j++){
      if (i == jw[j]) {
	continue;
      }
      dist = d[j]/w[j];/* both negative*/
      res += -w[j]*(dist - distance(x, dim, i, jw[j]))*(dist - distance(x, dim, i, jw[j]));
    }
  }
  return 0.5*res/scaling/scaling;

}

double StressMajorizationSmoother_smooth(StressMajorizationSmoother sm, int dim, double *x, int maxit_sm) {
  SparseMatrix Lw = sm->Lw, Lwd = sm->Lwd, Lwdd = NULL;
  int i, j, k, m, *id, *jd, *iw, *jw, idiag, iter = 0;
  double *w, *dd, *d, *y = NULL, *x0 = NULL, *x00 = NULL, diag, diff = 1, *lambda = sm->lambda;
  SparseMatrix Lc = NULL;
  double dij, dist;

  const double tol = 0.001;


  Lwdd = SparseMatrix_copy(Lwd);
  m = Lw->m;
  x0 = calloc(dim*m, sizeof(double));
  if (!x0) goto RETURN;

  memcpy(x0, x, sizeof(double)*dim*m);
  y = calloc(dim*m, sizeof(double));
  if (!y) goto RETURN;

  id = Lwd->ia; jd = Lwd->ja;
  d = Lwd->a;
  dd = Lwdd->a;
  w = Lw->a;
  iw = Lw->ia; jw = Lw->ja;

#ifdef DEBUG_PRINT
  if (Verbose) fprintf(stderr, "initial stress = %f\n", get_stress(m, dim, iw, jw, w, d, x, sm->scaling));
#else
  (void)iw;
  (void)jw;
#endif
  /* for the additional matrix L due to the position constraints */
  if (sm->scheme == SM_SCHEME_NORMAL_ELABEL){
    get_edge_label_matrix(sm->data, m, dim, x, &Lc, &x00);
    if (Lc) Lw = SparseMatrix_add(Lw, Lc);
  }

  while (iter++ < maxit_sm && diff > tol){

    for (i = 0; i < m; i++){
      idiag = -1;
      diag = 0.;
      for (j = id[i]; j < id[i+1]; j++){
	if (i == jd[j]) {
	  idiag = j;
	  continue;
	}

	dist = distance(x, dim, i, jd[j]);
	if (d[j] == 0){
	  dd[j] = 0;
	} else {
	  if (dist == 0){
	    dij = d[j]/w[j];/* the ideal distance */
	    /* perturb so points do not sit at the same place */
	    for (k = 0; k < dim; k++) x[jd[j]*dim+k] += 0.0001*(drand()+.0001)*dij;
	    dist = distance(x, dim, i, jd[j]);	
	  }
	  dd[j] = d[j]/dist;
	}
	diag += dd[j];
      }
      assert(idiag >= 0);
      dd[idiag] = -diag;
    }
    /* solve (Lw+lambda*I) x = Lwdd y + lambda x0 */

    SparseMatrix_multiply_dense(Lwdd, x, y, dim);

    if (lambda){/* is there a penalty term? */
      for (i = 0; i < m; i++){
	for (j = 0; j < dim; j++){
	  y[i*dim+j] += lambda[i]*x0[i*dim+j];
	}
      }
    } 

    /* additional term added to the rhs */
    switch (sm->scheme){
    case SM_SCHEME_NORMAL_ELABEL: {
      for (i = 0; i < m; i++){
	for (j = 0; j < dim; j++){
	  y[i*dim+j] += x00[i*dim+j];
	}
      }
      break;
    }
    default:
      break;
  }

#ifdef DEBUG_PRINT
    if (Verbose) {
      fprintf(stderr, "stress1 = %g\n",get_stress(m, dim, iw, jw, w, d, x, sm->scaling));
    }
#endif

    SparseMatrix_solve(Lw, dim, x, y,  sm->tol_cg, sm->maxit_cg);

#ifdef DEBUG_PRINT
    if (Verbose) fprintf(stderr, "stress2 = %g\n",get_stress(m, dim, iw, jw, w, d, y, sm->scaling));
#endif
    diff = total_distance(m, dim, x, y)/sqrt(vector_product(m*dim, x, x));
#ifdef DEBUG_PRINT
    if (Verbose){
      fprintf(stderr, "Outer iter = %d, cg res = %g, ||x_{k+1}-x_k||/||x_k|| = %g\n",iter, res, diff);
    }
#endif


    memcpy(x, y, sizeof(double)*m*dim);
  }

#ifdef DEBUG
  _statistics[1] += iter-1;
#endif

#ifdef DEBUG_PRINT
  if (Verbose) fprintf(stderr, "iter = %d, final stress = %f\n", iter, get_stress(m, dim, iw, jw, w, d, x, sm->scaling));
#endif

 RETURN:
  SparseMatrix_delete(Lwdd);
  if (Lc) {
    SparseMatrix_delete(Lc);
    SparseMatrix_delete(Lw);
  }

  free(x0);
  free(y);
  free(x00);
  return diff;
  
}

void StressMajorizationSmoother_delete(StressMajorizationSmoother sm){
  if (!sm) return;
  if (sm->Lw) SparseMatrix_delete(sm->Lw);
  if (sm->Lwd) SparseMatrix_delete(sm->Lwd);
  free(sm->lambda);
  if (sm->data) sm->data_deallocator(sm->data);
  free(sm);
}

TriangleSmoother TriangleSmoother_new(SparseMatrix A, int dim, double *x,
                                      bool use_triangularization) {
  int i, j, k, m = A->m, *ia = A->ia, *ja = A->ja, *iw, *jw, jdiag, nz;
  SparseMatrix B;
  double *d, *w, diag_d, diag_w, dist;
  double s = 0, stop = 0, sbot = 0;

  assert(SparseMatrix_is_symmetric(A, false));

  double *avg_dist = gv_calloc(m, sizeof(double));

  for (i = 0; i < m ;i++){
    avg_dist[i] = 0;
    nz = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      avg_dist[i] += distance(x, dim, i, ja[j]);
      nz++;
    }
    assert(nz > 0);
    avg_dist[i] /= nz;
  }

  TriangleSmoother sm = gv_alloc(sizeof(struct TriangleSmoother_struct));
  sm->scaling = 1;
  sm->data = NULL;
  sm->scheme = SM_SCHEME_NORMAL;
  sm->tol_cg = 0.01;
  sm->maxit_cg = floor(sqrt(A->m));

  double *lambda = sm->lambda = gv_calloc(m, sizeof(double));
  
  if (m > 2){
    if (use_triangularization){
      B= call_tri(m, x);
    } else {
      B= call_tri2(m, dim, x);
    }
  } else {
    B = SparseMatrix_copy(A);
  }



  sm->Lw = SparseMatrix_add(A, B);

  SparseMatrix_delete(B);
  sm->Lwd = SparseMatrix_copy(sm->Lw);
  if (!(sm->Lw) || !(sm->Lwd)) {
    TriangleSmoother_delete(sm);
    return NULL;
  }

  iw = sm->Lw->ia; jw = sm->Lw->ja;

  w = sm->Lw->a;
  d = sm->Lwd->a;

  for (i = 0; i < m; i++){
    diag_d = diag_w = 0;
    jdiag = -1;
    for (j = iw[i]; j < iw[i+1]; j++){
      k = jw[j];
      if (k == i){
	jdiag = j;
	continue;
      }
      /*      w[j] = -1./(ia[i+1]-ia[i]+ia[ja[j]+1]-ia[ja[j]]);
	      w[j] = -2./(avg_dist[i]+avg_dist[k]);
	      w[j] = -1.*/;/* use unit weight for now, later can try 1/(deg(i)+deg(k)) */
      dist = pow(distance_cropped(x,dim,i,k),0.6);
      w[j] = 1/(dist*dist);
      diag_w += w[j];

      /*      d[j] = w[j]*distance(x,dim,i,k);
	      d[j] = w[j]*(avg_dist[i] + avg_dist[k])*0.5;*/
      d[j] = w[j]*dist;
      stop += d[j]*distance(x,dim,i,k);
      sbot += d[j]*dist;
      diag_d += d[j];

    }

    lambda[i] *= (-diag_w);

    assert(jdiag >= 0);
    w[jdiag] = -diag_w + lambda[i];
    d[jdiag] = -diag_d;
  }

  s = stop/sbot;
  for (i = 0; i < iw[m]; i++) d[i] *= s;
  sm->scaling = s;

  free(avg_dist);

  return sm;
}

void TriangleSmoother_delete(TriangleSmoother sm){

  StressMajorizationSmoother_delete(sm);

}

void TriangleSmoother_smooth(TriangleSmoother sm, int dim, double *x){
  StressMajorizationSmoother_smooth(sm, dim, x, 50);
}

/* ================================ spring and spring-electrical based smoother ================ */
SpringSmoother SpringSmoother_new(SparseMatrix A, int dim, spring_electrical_control ctrl, double *x){
  int i, j, k, l, m = A->m, *ia = A->ia, *ja = A->ja, *id, *jd;
  int nz;
  double *d, *dd;
  SparseMatrix ID = NULL;

  assert(SparseMatrix_is_symmetric(A, false));

  ID = ideal_distance_matrix(A, dim, x);
  dd = ID->a;

  SpringSmoother sm = gv_alloc(sizeof(struct SpringSmoother_struct));
  int *mask = gv_calloc(m, sizeof(int));

  double *avg_dist = gv_calloc(m, sizeof(double));

  for (i = 0; i < m ;i++){
    avg_dist[i] = 0;
    nz = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      avg_dist[i] += distance(x, dim, i, ja[j]);
      nz++;
    }
    assert(nz > 0);
    avg_dist[i] /= nz;
  }


  for (i = 0; i < m; i++) mask[i] = -1;

  nz = 0;
  for (i = 0; i < m; i++){
    mask[i] = i;
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      if (mask[k] != i){
	mask[k] = i;
	nz++;
      }
    }
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      for (l = ia[k]; l < ia[k+1]; l++){
	if (mask[ja[l]] != i){
	  mask[ja[l]] = i;
	  nz++;
	}
      }
    }
  }

  sm->D = SparseMatrix_new(m, m, nz, MATRIX_TYPE_REAL, FORMAT_CSR);
  if (!(sm->D)){
    SpringSmoother_delete(sm);
    return NULL;
  }

  id = sm->D->ia; jd = sm->D->ja;
  d = sm->D->a;
  id[0] = 0;

  nz = 0;
  for (i = 0; i < m; i++){
    mask[i] = i+m;
    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      if (mask[k] != i+m){
	mask[k] = i+m;
	jd[nz] = k;
	d[nz] = (avg_dist[i] + avg_dist[k])*0.5;
	d[nz] = dd[j];
	nz++;
      }
    }

    for (j = ia[i]; j < ia[i+1]; j++){
      k = ja[j];
      for (l = ia[k]; l < ia[k+1]; l++){
	if (mask[ja[l]] != i+m){
	  mask[ja[l]] = i+m;
	  jd[nz] = ja[l];
	  d[nz] = (avg_dist[i] + 2*avg_dist[k] + avg_dist[ja[l]])*0.5;
	  d[nz] = dd[j]+dd[l];
	  nz++;
	}
      }
    }
    id[i+1] = nz;
  }
  sm->D->nz = nz;
  sm->ctrl = spring_electrical_control_new();
  *(sm->ctrl) = *ctrl;
  sm->ctrl->random_start = false;
  sm->ctrl->multilevels = 1;
  sm->ctrl->step /= 2;
  sm->ctrl->maxiter = 20;

  free(mask);
  free(avg_dist);
  SparseMatrix_delete(ID);

  return sm;
}


void SpringSmoother_delete(SpringSmoother sm){
  if (!sm) return;
  if (sm->D) SparseMatrix_delete(sm->D);
  if (sm->ctrl) spring_electrical_control_delete(sm->ctrl);
}




void SpringSmoother_smooth(SpringSmoother sm, SparseMatrix A, int dim, double *x){
  int flag = 0;

  spring_electrical_spring_embedding(dim, A, sm->D, sm->ctrl, x, &flag);
  assert(!flag);

}

/*=============================== end of spring and spring-electrical based smoother =========== */

void post_process_smoothing(int dim, SparseMatrix A, spring_electrical_control ctrl, double *x) {
#ifdef TIME
  clock_t  cpu;
#endif

#ifdef TIME
  cpu = clock();
#endif

  switch (ctrl->smoothing){
  case SMOOTHING_RNG:
  case SMOOTHING_TRIANGLE:{
    TriangleSmoother sm;

    if (A->m > 2) {  /* triangles need at least 3 nodes */
      if (ctrl->smoothing == SMOOTHING_RNG){
        sm = TriangleSmoother_new(A, dim, x, false);
      } else {
        sm = TriangleSmoother_new(A, dim, x, true);
      }
      TriangleSmoother_smooth(sm, dim, x);
      TriangleSmoother_delete(sm);
    }
    break;
  }
  case SMOOTHING_STRESS_MAJORIZATION_GRAPH_DIST:
  case SMOOTHING_STRESS_MAJORIZATION_POWER_DIST:
  case SMOOTHING_STRESS_MAJORIZATION_AVG_DIST:
    {
      StressMajorizationSmoother sm;
      int dist_scheme = IDEAL_AVG_DIST;

      if (ctrl->smoothing == SMOOTHING_STRESS_MAJORIZATION_GRAPH_DIST){
	dist_scheme = IDEAL_GRAPH_DIST;
      } else if (ctrl->smoothing == SMOOTHING_STRESS_MAJORIZATION_AVG_DIST){
	dist_scheme = IDEAL_AVG_DIST;
      } else if (ctrl->smoothing == SMOOTHING_STRESS_MAJORIZATION_POWER_DIST){
	dist_scheme = IDEAL_POWER_DIST;
      }

      sm = StressMajorizationSmoother2_new(A, dim, 0.05, x, dist_scheme);
      StressMajorizationSmoother_smooth(sm, dim, x, 50);
      StressMajorizationSmoother_delete(sm);
      break;
    }
  case SMOOTHING_SPRING:{
    SpringSmoother sm = SpringSmoother_new(A, dim, ctrl, x);
    SpringSmoother_smooth(sm, A, dim, x);
    SpringSmoother_delete(sm);

    break;
  }

  }/* end switch between smoothing methods */

#ifdef TIME
  if (Verbose) fprintf(stderr, "post processing %f\n",((double) (clock() - cpu)) / CLOCKS_PER_SEC);
#endif
}
