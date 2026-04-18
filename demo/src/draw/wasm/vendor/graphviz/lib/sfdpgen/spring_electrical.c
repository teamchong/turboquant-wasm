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
#include <cgraph/list.h>
#include <sparse/SparseMatrix.h>
#include <sfdpgen/spring_electrical.h>
#include <sparse/QuadTree.h>
#include <sfdpgen/Multilevel.h>
#include <sfdpgen/post_process.h>
#include <neatogen/overlap.h>
#include <common/types.h>
#include <common/arith.h>
#include <math.h>
#include <common/globals.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/bitarray.h>

/// another parameter
/// fₐ(i, j) = C × dist(i , j)² ÷ K × dᵢⱼ, fᵣ(i, j) = K³⁻ᵖ ÷ dist(i, j)⁻ᵖ
static const double C = 0.2;

/// cut off size above which quadtree approximation is used
static const int quadtree_size = 45;

/// Barnes-Hutt constant,
/// if width(snode) ÷ dist[i, snode] < bh, treat snode as a supernode.
static const double bh = 0.6;

/// minimum different between two subsequence config before terminating.
/// ||x - xold||_∞ < tol ÷ K
static const double tol = 0.001;

static const double cool = 0.90;

spring_electrical_control spring_electrical_control_new(void){
  spring_electrical_control ctrl;
  ctrl = gv_alloc(sizeof(struct spring_electrical_control_struct));
  ctrl->p = AUTOP;/*a negativve number default to -1. repulsive force = dist^p */
  ctrl->random_start = true; // whether to apply SE from a random layout, or from existing layout
  ctrl->K = -1;/* the natural distance. If K < 0, K will be set to the average distance of an edge */
  ctrl->multilevels = 0;/* if <=1, single level */

  ctrl->max_qtree_level = 10;/* max level of quadtree */
  ctrl->maxiter = 500;
  ctrl->step = 0.1;
  ctrl->adaptive_cooling = true;
  ctrl->random_seed = 123;
  ctrl->beautify_leaves = false;
  ctrl->smoothing = SMOOTHING_NONE;
  ctrl->overlap = 0;
  ctrl->do_shrinking = true;
  ctrl->tscheme = QUAD_TREE_HYBRID;
  ctrl->initial_scaling = -4;
  ctrl->rotation = 0.;
  ctrl->edge_labeling_scheme = 0;
  return ctrl;
}

void spring_electrical_control_delete(spring_electrical_control ctrl){
  free(ctrl);
}

static char* smoothings[] = {
  "NONE", "STRESS_MAJORIZATION_GRAPH_DIST", "STRESS_MAJORIZATION_AVG_DIST", "STRESS_MAJORIZATION_POWER_DIST", "SPRING", "TRIANGLE", "RNG"
};

static char* tschemes[] = {
  "NONE", "NORMAL", "FAST", "HYBRID"
};

void spring_electrical_control_print(spring_electrical_control ctrl){
  fprintf (stderr, "spring_electrical_control:\n");
  fprintf (stderr, "  repulsive exponent: %.03f\n", ctrl->p);
  fprintf(stderr, "  random start %d seed %d\n", (int)ctrl->random_start,
          ctrl->random_seed);
  fprintf (stderr, "  K : %.03f C : %.03f\n", ctrl->K, C);
  fprintf (stderr, "  max levels %d\n", ctrl->multilevels);
  fprintf (stderr, "  quadtree size %d max_level %d\n", quadtree_size, ctrl->max_qtree_level);
  fprintf (stderr, "  Barnes-Hutt constant %.03f tolerance  %.03f maxiter %d\n", bh, tol, ctrl->maxiter);
  fprintf(stderr, "  cooling %.03f step size  %.03f adaptive %d\n", cool,
          ctrl->step, (int)ctrl->adaptive_cooling);
  fprintf (stderr, "  beautify_leaves %d node weights %d rotation %.03f\n",
           (int)ctrl->beautify_leaves, 0, ctrl->rotation);
  fprintf (stderr, "  smoothing %s overlap %d initial_scaling %.03f do_shrinking %d\n",
    smoothings[ctrl->smoothing], ctrl->overlap, ctrl->initial_scaling, (int)ctrl->do_shrinking);
  fprintf (stderr, "  octree scheme %s\n", tschemes[ctrl->tscheme]);
  fprintf (stderr, "  edge_labeling_scheme %d\n", ctrl->edge_labeling_scheme);
}

enum { MAX_I = 20, OPT_UP = 1, OPT_DOWN = -1, OPT_INIT = 0 };

typedef struct {
  int i;
  double work[MAX_I + 1];
  int direction;
} oned_optimizer;

static oned_optimizer oned_optimizer_new(int i){
  oned_optimizer opt = {0};
  opt.i = i;
  opt.direction = OPT_INIT;
  return opt;
}

static void oned_optimizer_train(oned_optimizer *opt, double work) {
  int i = opt->i;

  assert(i >= 0);
  opt->work[i] = work;
  if (opt->direction == OPT_INIT){
    if (opt->i == MAX_I){
      opt->direction = OPT_DOWN;
      opt->i = opt->i - 1;
    } else {
      opt->direction = OPT_UP;
      opt->i = MIN(MAX_I, opt->i + 1);
    }
  } else if (opt->direction == OPT_UP){
    assert(i >= 1);
    if (opt->work[i] < opt->work[i-1] && opt->i < MAX_I){
      opt->i = MIN(MAX_I, opt->i + 1);
    } else {
      (opt->i)--;
      opt->direction = OPT_DOWN;
    }
  } else {
    assert(i < MAX_I);
    if (opt->work[i] < opt->work[i+1] && opt->i > 0){
      opt->i = MAX(0, opt->i-1);
    } else {
      (opt->i)++;
      opt->direction = OPT_UP;
    }
  }
}

static int oned_optimizer_get(const oned_optimizer opt) {
  return opt.i;
}


double average_edge_length(SparseMatrix A, int dim, double *coord){
  double dist = 0, d;
  int *ia = A->ia, *ja = A->ja, i, j, k;
  assert(SparseMatrix_is_symmetric(A, true));

  if (ia[A->m] == 0) return 1;
  for (i = 0; i < A->m; i++){
    for (j = ia[i]; j < ia[i+1]; j++){
      d = 0;
      for (k = 0; k < dim; k++){
	d += (coord[dim*i+k] - coord[dim*ja[j]])*(coord[dim*i+k] - coord[dim*ja[j]]);
      }
      dist += sqrt(d);
    }
  }
  return dist/ia[A->m];
}

static double update_step(bool adaptive_cooling, double step, double Fnorm,
                          double Fnorm0) {

  if (!adaptive_cooling) {
    return cool*step;
  }
  if (Fnorm >= Fnorm0){
    step = cool*step;
  } else if (Fnorm > 0.95*Fnorm0){
    //    step = step;
  } else {
    step = 0.99*step/cool;
  }
  return step;
}


#define node_degree(i) (ia[(i)+1] - ia[(i)])

static void set_leaves(double *x, int dim, double dist, double ang, int i, int j){
  x[dim*j] = cos(ang)*dist + x[dim*i];
  x[dim*j+1] = sin(ang)*dist + x[dim*i+1];
}

DEFINE_LIST(ints, int)

static void beautify_leaves(int dim, SparseMatrix A, double *x){
  int m = A->m, i, j, *ia = A->ia, *ja = A->ja;
  int p;
  double dist;
  double step;

  assert(!SparseMatrix_has_diagonal(A));

  bitarray_t checked = bitarray_new(m);

  for (i = 0; i < m; i++){
    if (ia[i+1] - ia[i] != 1) continue;
    if (bitarray_get(checked, i)) continue;
    p = ja[ia[i]];
    if (!bitarray_get(checked, p)) {
      bitarray_set(&checked, p, true);
      dist = 0;
      ints_t leaves = {0};
      for (j = ia[p]; j < ia[p+1]; j++){
	if (node_degree(ja[j]) == 1){
	  bitarray_set(&checked, ja[j], true);
	  dist += distance(x, dim, p, ja[j]);
	  ints_append(&leaves, ja[j]);
	}
      }
      assert(!ints_is_empty(&leaves));
      dist /= (double)ints_size(&leaves);
      double ang1 = 0;
      double ang2 = 2 * M_PI;
      const double pad = 0.1; // fudge factor to account for the size and
                              // placement of the nodes themselves
      ang1 += pad;
      ang2 -= pad;
      assert(ang2 >= ang1);
      step = 0.;
      if (ints_size(&leaves) > 1) step = (ang2 - ang1) / (double)ints_size(&leaves);
      for (size_t k = 0; k < ints_size(&leaves); k++) {
	set_leaves(x, dim, dist, ang1, p, ints_get(&leaves, k));
	ang1 += step;
      }
      ints_free(&leaves);
    }
  }

  bitarray_reset(&checked);
}

void spring_electrical_embedding_fast(int dim, SparseMatrix A0, spring_electrical_control ctrl, double *x, int *flag){
  /* x is a point to a 1D array, x[i*dim+j] gives the coordinate of the i-th node at dimension j.  */
  SparseMatrix A = A0;
  int m, n;
  int i, j, k;
  double p = ctrl->p, K = ctrl->K, CRK, maxiter = ctrl->maxiter, step = ctrl->step, KP;
  int *ia = NULL, *ja = NULL;
  double *f = NULL, dist, F, Fnorm = 0, Fnorm0;
  int iter = 0;
  const bool adaptive_cooling = ctrl->adaptive_cooling;
  double counts[4], *force = NULL;
#ifdef TIME
  clock_t start, end, start0;
  double qtree_cpu = 0, qtree_new_cpu = 0;
  double total_cpu = 0;
  start0 = clock();
#endif
  int max_qtree_level = ctrl->max_qtree_level;

  if (!A || maxiter <= 0) return;

  m = A->m, n = A->n;
  if (n <= 0 || dim <= 0) return;

  oned_optimizer qtree_level_optimizer = oned_optimizer_new(max_qtree_level);

  *flag = 0;
  if (m != n) {
    *flag = ERROR_NOT_SQUARE_MATRIX;
    goto RETURN;
  }
  assert(A->format == FORMAT_CSR);
  A = SparseMatrix_symmetrize(A, true);
  ia = A->ia;
  ja = A->ja;

  if (ctrl->random_start){
    srand(ctrl->random_seed);
    for (i = 0; i < dim*n; i++) x[i] = drand();
  }
  if (K < 0){
    ctrl->K = K = average_edge_length(A, dim, x);
  }
  if (p >= 0) ctrl->p = p = -1;
  KP = pow(K, 1 - p);
  CRK = pow(C, (2.-p)/3.)/K;

  force = gv_calloc(dim * n, sizeof(double));

  do {
    iter++;
    Fnorm0 = Fnorm;
    Fnorm = 0.;

    max_qtree_level = oned_optimizer_get(qtree_level_optimizer);

#ifdef TIME
    start = clock();
#endif
    QuadTree qt = QuadTree_new_from_point_list(dim, n, max_qtree_level, x);

#ifdef TIME
    qtree_new_cpu += ((double) (clock() - start))/CLOCKS_PER_SEC;
#endif

    /* repulsive force */
#ifdef TIME
    start = clock();
#endif

    QuadTree_get_repulsive_force(qt, force, x, bh, p, KP, counts);

#ifdef TIME
    end = clock();
    qtree_cpu += ((double) (end - start)) / CLOCKS_PER_SEC;
#endif

    /* attractive force   C^((2-p)/3) ||x_i-x_j||/K * (x_j - x_i) */
    for (i = 0; i < n; i++){
      f = &(force[i*dim]);
      for (j = ia[i]; j < ia[i+1]; j++){
	if (ja[j] == i) continue;
	dist = distance(x, dim, i, ja[j]);
	for (k = 0; k < dim; k++){
	  f[k] -= CRK*(x[i*dim+k] - x[ja[j]*dim+k])*dist;
	}
      }
    }


    /* move */
    for (i = 0; i < n; i++){
      f = &(force[i*dim]);
      F = 0.;
      for (k = 0; k < dim; k++) F += f[k]*f[k];
      F = sqrt(F);
      Fnorm += F;
      if (F > 0) for (k = 0; k < dim; k++) f[k] /= F;
      for (k = 0; k < dim; k++) x[i*dim+k] += step*f[k];
    }/* done vertex i */



    if (qt) {
#ifdef TIME
      start = clock();
#endif
      QuadTree_delete(qt);
#ifdef TIME
      end = clock();
      qtree_new_cpu += ((double) (end - start)) / CLOCKS_PER_SEC;
#endif

      oned_optimizer_train(&qtree_level_optimizer,
                           counts[0] + 0.85 * counts[1] + 3.3 * counts[2]);
    } else {
      if (Verbose) {
        fprintf(stderr, "\r                iter = %d, step = %f Fnorm = %f nz = %d  K = %f                                  ",iter, step, Fnorm, A->nz,K);
      }
    }

    step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
  } while (step > tol && iter < maxiter);

#ifdef DEBUG_PRINT
    if (Verbose) {
      fprintf(stderr, "\n iter = %d, step = %f Fnorm = %f nz = %d  K = %f   ",iter, step, Fnorm, A->nz, K);
    }
#endif

  if (ctrl->beautify_leaves) beautify_leaves(dim, A, x);

#ifdef TIME
  total_cpu += ((double) (clock() - start0)) / CLOCKS_PER_SEC;
  if (Verbose) fprintf(stderr, "\n time for qtree = %f, qtree_force = %f, total cpu = %f\n",qtree_new_cpu, qtree_cpu, total_cpu);
#endif


 RETURN:
  ctrl->max_qtree_level = max_qtree_level;

  if (A != A0) SparseMatrix_delete(A);
  free(force);
}

static void spring_electrical_embedding_slow(int dim, SparseMatrix A0, spring_electrical_control ctrl, double *x, int *flag){
  /* a version that does vertex moves in one go, instead of one at a time, use for debugging the fast version. Quadtree is not used. */
  /* x is a point to a 1D array, x[i*dim+j] gives the coordinate of the i-th node at dimension j.  */
  SparseMatrix A = A0;
  int m, n;
  int i, j, k;
  double p = ctrl->p, K = ctrl->K, CRK, maxiter = ctrl->maxiter, step = ctrl->step, KP;
  int *ia = NULL, *ja = NULL;
  double *f = NULL, dist, F, Fnorm = 0, Fnorm0;
  int iter = 0;
  const bool adaptive_cooling = ctrl->adaptive_cooling;
  double *force;
#ifdef TIME
  clock_t start, end, start0, start2;
  double total_cpu = 0;
  start0 = clock();
#endif

  fprintf(stderr,"spring_electrical_embedding_slow");
  if (!A || maxiter <= 0) return;

  m = A->m, n = A->n;
  if (n <= 0 || dim <= 0) return;
  force = gv_calloc(n *dim, sizeof(double));

  *flag = 0;
  if (m != n) {
    *flag = ERROR_NOT_SQUARE_MATRIX;
    goto RETURN;
  }
  assert(A->format == FORMAT_CSR);
  A = SparseMatrix_symmetrize(A, true);
  ia = A->ia;
  ja = A->ja;

  if (ctrl->random_start){
    srand(ctrl->random_seed);
    for (i = 0; i < dim*n; i++) x[i] = drand();
  }
  if (K < 0){
    ctrl->K = K = average_edge_length(A, dim, x);
  }
  if (p >= 0) ctrl->p = p = -1;
  KP = pow(K, 1 - p);
  CRK = pow(C, (2.-p)/3.)/K;

  f = gv_calloc(dim, sizeof(double));
  do {
    for (i = 0; i < dim*n; i++) force[i] = 0;

    iter++;
    Fnorm0 = Fnorm;
    Fnorm = 0.;

#ifdef TIME
    start2 = clock();
#endif


    for (i = 0; i < n; i++){
      for (k = 0; k < dim; k++) f[k] = 0.;
     /* repulsive force K^(1 - p)/||x_i-x_j||^(1 - p) (x_i - x_j) */
      for (j = 0; j < n; j++){
	if (j == i) continue;
	dist = distance_cropped(x, dim, i, j);
	for (k = 0; k < dim; k++){
	  f[k] += KP*(x[i*dim+k] - x[j*dim+k])/pow(dist, 1.- p);
	}
      }
      for (k = 0; k < dim; k++) force[i*dim+k] += f[k];
    }



    for (i = 0; i < n; i++){
      for (k = 0; k < dim; k++) f[k] = 0.;
      /* attractive force   C^((2-p)/3) ||x_i-x_j||/K * (x_j - x_i) */
      for (j = ia[i]; j < ia[i+1]; j++){
	if (ja[j] == i) continue;
	dist = distance(x, dim, i, ja[j]);
	for (k = 0; k < dim; k++){
	  f[k] -= CRK*(x[i*dim+k] - x[ja[j]*dim+k])*dist;
	}
      }
      for (k = 0; k < dim; k++) force[i*dim+k] += f[k];
    }



    for (i = 0; i < n; i++){
     /* normalize force */
      for (k = 0; k < dim; k++) f[k] = force[i*dim+k];

      F = 0.;
      for (k = 0; k < dim; k++) F += f[k]*f[k];
      F = sqrt(F);
      Fnorm += F;

      if (F > 0) for (k = 0; k < dim; k++) f[k] /= F;

      for (k = 0; k < dim; k++) x[i*dim+k] += step*f[k];

    }/* done vertex i */

    step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
  } while (step > tol && iter < maxiter);

#ifdef DEBUG_PRINT
    if (Verbose) {
      fprintf(stderr, "iter = %d, step = %f Fnorm = %f nsuper = 0 nz = %d  K = %f   ",iter, step, Fnorm, A->nz,K);
    }
#endif

  if (ctrl->beautify_leaves) beautify_leaves(dim, A, x);

#ifdef TIME
  total_cpu += ((double) (clock() - start0)) / CLOCKS_PER_SEC;
  if (Verbose) fprintf(stderr, "time for supernode = 0, total cpu = %f\n", total_cpu);
#endif

 RETURN:
  if (A != A0) SparseMatrix_delete(A);
  free(f);
  free(force);
}



void spring_electrical_embedding(int dim, SparseMatrix A0, spring_electrical_control ctrl, double *x, int *flag){
  /* x is a point to a 1D array, x[i*dim+j] gives the coordinate of the i-th node at dimension j.  */
  SparseMatrix A = A0;
  int m, n;
  int i, j, k;
  double p = ctrl->p, K = ctrl->K, CRK, maxiter = ctrl->maxiter, step = ctrl->step, KP;
  int *ia = NULL, *ja = NULL;
  double *f = NULL, dist, F, Fnorm = 0, Fnorm0;
  int iter = 0;
  const bool adaptive_cooling = ctrl->adaptive_cooling;
  bool USE_QT = false;
  int nsuper = 0, nsupermax = 10;
  double *center = NULL, *supernode_wgts = NULL, *distances = NULL, nsuper_avg, counts = 0, counts_avg = 0;
#ifdef TIME
  clock_t start, end, start0, start2;
  double qtree_cpu = 0;
  double total_cpu = 0;
  start0 = clock();
#endif
  int max_qtree_level = ctrl->max_qtree_level;
  oned_optimizer qtree_level_optimizer = {0};

  if (!A || maxiter <= 0) return;

  m = A->m, n = A->n;
  if (n <= 0 || dim <= 0) return;

  if (n >= quadtree_size) {
    USE_QT = true;
    qtree_level_optimizer = oned_optimizer_new(max_qtree_level);
    center = gv_calloc(nsupermax * dim, sizeof(double));
    supernode_wgts = gv_calloc(nsupermax, sizeof(double));
    distances = gv_calloc(nsupermax, sizeof(double));
  }
  *flag = 0;
  if (m != n) {
    *flag = ERROR_NOT_SQUARE_MATRIX;
    goto RETURN;
  }
  assert(A->format == FORMAT_CSR);
  A = SparseMatrix_symmetrize(A, true);
  ia = A->ia;
  ja = A->ja;

  if (ctrl->random_start){
    srand(ctrl->random_seed);
    for (i = 0; i < dim*n; i++) x[i] = drand();
  }
  if (K < 0){
    ctrl->K = K = average_edge_length(A, dim, x);
  }
  if (p >= 0) ctrl->p = p = -1;
  KP = pow(K, 1 - p);
  CRK = pow(C, (2.-p)/3.)/K;

  f = gv_calloc(dim, sizeof(double));
  do {

    iter++;
    Fnorm0 = Fnorm;
    Fnorm = 0.;
    nsuper_avg = 0;
    counts_avg = 0;

    QuadTree qt = NULL;
    if (USE_QT) {

      max_qtree_level = oned_optimizer_get(qtree_level_optimizer);
      qt = QuadTree_new_from_point_list(dim, n, max_qtree_level, x);

	
    }
#ifdef TIME
    start2 = clock();
#endif

    for (i = 0; i < n; i++){
      for (k = 0; k < dim; k++) f[k] = 0.;
      /* attractive force   C^((2-p)/3) ||x_i-x_j||/K * (x_j - x_i) */
      for (j = ia[i]; j < ia[i+1]; j++){
	if (ja[j] == i) continue;
	dist = distance(x, dim, i, ja[j]);
	for (k = 0; k < dim; k++){
	  f[k] -= CRK*(x[i*dim+k] - x[ja[j]*dim+k])*dist;
	}
      }

      /* repulsive force K^(1 - p)/||x_i-x_j||^(1 - p) (x_i - x_j) */
      if (USE_QT){
#ifdef TIME
	start = clock();
#endif
	QuadTree_get_supernodes(qt, bh, &(x[dim*i]), i, &nsuper, &nsupermax,
				&center, &supernode_wgts, &distances, &counts);

#ifdef TIME
	end = clock();
	qtree_cpu += ((double) (end - start)) / CLOCKS_PER_SEC;
#endif
	counts_avg += counts;
	nsuper_avg += nsuper;
	for (j = 0; j < nsuper; j++){
	  dist = MAX(distances[j], MINDIST);
	  for (k = 0; k < dim; k++){
	    f[k] += supernode_wgts[j]*KP*(x[i*dim+k] - center[j*dim+k])/pow(dist, 1.- p);
	  }
	}
      } else {
	for (j = 0; j < n; j++){
	  if (j == i) continue;
	  dist = distance_cropped(x, dim, i, j);
	  for (k = 0; k < dim; k++){
	    f[k] += KP*(x[i*dim+k] - x[j*dim+k])/pow(dist, 1.- p);
	  }
	}
      }
	
      /* normalize force */
      F = 0.;
      for (k = 0; k < dim; k++) F += f[k]*f[k];
      F = sqrt(F);
      Fnorm += F;

      if (F > 0) for (k = 0; k < dim; k++) f[k] /= F;

      for (k = 0; k < dim; k++) x[i*dim+k] += step*f[k];

    }/* done vertex i */

    if (qt) {
      QuadTree_delete(qt);
      nsuper_avg /= n;
      counts_avg /= n;
      oned_optimizer_train(&qtree_level_optimizer, 5 * nsuper_avg + counts_avg);
    }

    step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
  } while (step > tol && iter < maxiter);

#ifdef DEBUG_PRINT
    if (Verbose) {
      if (USE_QT){
	fprintf(stderr, "iter = %d, step = %f Fnorm = %f qt_level = %d nsuper = %d nz = %d  K = %f   ",iter, step, Fnorm, max_qtree_level, (int) nsuper_avg,A->nz,K);
      } else {
	fprintf(stderr, "iter = %d, step = %f Fnorm = %f nsuper = %d nz = %d  K = %f   ",iter, step, Fnorm, (int) nsuper_avg,A->nz,K);
      }
    }
#endif

  if (ctrl->beautify_leaves) beautify_leaves(dim, A, x);

#ifdef TIME
  total_cpu += ((double) (clock() - start0)) / CLOCKS_PER_SEC;
  if (Verbose) fprintf(stderr, "time for supernode = %f, total cpu = %f\n",qtree_cpu, total_cpu);
#endif

 RETURN:
  if (USE_QT) {
    ctrl->max_qtree_level = max_qtree_level;
  }
  if (A != A0) SparseMatrix_delete(A);
  free(f);
  free(center);
  free(supernode_wgts);
  free(distances);
}

void spring_electrical_spring_embedding(int dim, SparseMatrix A0, SparseMatrix D, spring_electrical_control ctrl, double *x, int *flag){
  /* x is a point to a 1D array, x[i*dim+j] gives the coordinate of the i-th node at dimension j. Same as the spring-electrical except we also
     introduce force due to spring length
   */
  SparseMatrix A = A0;
  int m, n;
  int i, j, k;
  double p = ctrl->p, K = ctrl->K, CRK, maxiter = ctrl->maxiter, step = ctrl->step, KP;
  int *ia = NULL, *ja = NULL;
  int *id = NULL, *jd = NULL;
  double *d;
  double *xold = NULL;
  double *f = NULL, dist, F, Fnorm = 0, Fnorm0;
  int iter = 0;
  const bool adaptive_cooling = ctrl->adaptive_cooling;
  bool USE_QT = false;
  int nsuper = 0, nsupermax = 10;
  double *center = NULL, *supernode_wgts = NULL, *distances = NULL, counts = 0;
  int max_qtree_level = 10;

  if (!A  || maxiter <= 0) return;
  m = A->m, n = A->n;
  if (n <= 0 || dim <= 0) return;

  if (n >= quadtree_size) {
    USE_QT = true;
    center = gv_calloc(nsupermax * dim, sizeof(double));
    supernode_wgts = gv_calloc(nsupermax, sizeof(double));
    distances = gv_calloc(nsupermax, sizeof(double));
  }
  *flag = 0;
  if (m != n) {
    *flag = ERROR_NOT_SQUARE_MATRIX;
    goto RETURN;
  }
  assert(A->format == FORMAT_CSR);
  A = SparseMatrix_symmetrize(A, true);
  ia = A->ia;
  ja = A->ja;
  id = D->ia;
  jd = D->ja;
  d = D->a;

  if (ctrl->random_start){
    srand(ctrl->random_seed);
    for (i = 0; i < dim*n; i++) x[i] = drand();
  }
  if (K < 0){
    ctrl->K = K = average_edge_length(A, dim, x);
  }
  if (p >= 0) ctrl->p = p = -1;
  KP = pow(K, 1 - p);
  CRK = pow(C, (2.-p)/3.)/K;

  f = gv_calloc(dim, sizeof(double));
  xold = gv_calloc(dim * n, sizeof(double));
  do {
    iter++;
    memcpy(xold, x, sizeof(double)*dim*n);
    Fnorm0 = Fnorm;
    Fnorm = 0.;

    QuadTree qt = NULL;
    if (USE_QT) {
      qt = QuadTree_new_from_point_list(dim, n, max_qtree_level, x);
    }

    for (i = 0; i < n; i++){
      for (k = 0; k < dim; k++) f[k] = 0.;
      /* attractive force   C^((2-p)/3) ||x_i-x_j||/K * (x_j - x_i) */

      for (j = ia[i]; j < ia[i+1]; j++){
	if (ja[j] == i) continue;
	dist = distance(x, dim, i, ja[j]);
	for (k = 0; k < dim; k++){
	  f[k] -= CRK*(x[i*dim+k] - x[ja[j]*dim+k])*dist;
	}
      }

      for (j = id[i]; j < id[i+1]; j++){
	if (jd[j] == i) continue;
	dist = distance_cropped(x, dim, i, jd[j]);
	for (k = 0; k < dim; k++){
	  if (dist < d[j]){
	    f[k] += 0.2*CRK*(x[i*dim+k] - x[jd[j]*dim+k])*(dist - d[j])*(dist - d[j])/dist;
	  } else {
	    f[k] -= 0.2*CRK*(x[i*dim+k] - x[jd[j]*dim+k])*(dist - d[j])*(dist - d[j])/dist;
	  }
	  /* f[k] -= 0.2*CRK*(x[i*dim+k] - x[jd[j]*dim+k])*(dist - d[j]);*/
	}
      }

      /* repulsive force K^(1 - p)/||x_i-x_j||^(1 - p) (x_i - x_j) */
      if (USE_QT){
	QuadTree_get_supernodes(qt, bh, &(x[dim*i]), i, &nsuper, &nsupermax,
				&center, &supernode_wgts, &distances, &counts);
	for (j = 0; j < nsuper; j++){
	  dist = MAX(distances[j], MINDIST);
	  for (k = 0; k < dim; k++){
	    f[k] += supernode_wgts[j]*KP*(x[i*dim+k] - center[j*dim+k])/pow(dist, 1.- p);
	  }
	}
      } else {
	for (j = 0; j < n; j++){
	  if (j == i) continue;
	  dist = distance_cropped(x, dim, i, j);
	  for (k = 0; k < dim; k++){
	    f[k] += KP*(x[i*dim+k] - x[j*dim+k])/pow(dist, 1.- p);
	  }
	}
      }
	
      /* normalize force */
      F = 0.;
      for (k = 0; k < dim; k++) F += f[k]*f[k];
      F = sqrt(F);
      Fnorm += F;

      if (F > 0) for (k = 0; k < dim; k++) f[k] /= F;

      for (k = 0; k < dim; k++) x[i*dim+k] += step*f[k];

    }/* done vertex i */

    if (qt) QuadTree_delete(qt);

    step = update_step(adaptive_cooling, step, Fnorm, Fnorm0);
  } while (step > tol && iter < maxiter);

  if (ctrl->beautify_leaves) beautify_leaves(dim, A, x);

 RETURN:
  free(xold);
  if (A != A0) SparseMatrix_delete(A);
  free(f);
  free(center);
  free(supernode_wgts);
  free(distances);
}

static void interpolate_coord(int dim, SparseMatrix A, double *x) {
  int i, j, k, *ia = A->ia, *ja = A->ja, nz;
  double alpha = 0.5, beta;

  double *y = gv_calloc(dim, sizeof(double));
  for (i = 0; i < A->m; i++){
    for (k = 0; k < dim; k++) y[k] = 0;
    nz = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (ja[j] == i) continue;
      nz++;
      for (k = 0; k < dim; k++){
	y[k] += x[ja[j]*dim + k];
      }
    }
    if (nz > 0){
      beta = (1-alpha)/nz;
      for (k = 0; k < dim; k++) x[i*dim+k] = alpha*x[i*dim+k] +  beta*y[k];
    }
  }

  free(y);
}
static void prolongate(int dim, SparseMatrix A, SparseMatrix P, SparseMatrix R, double *x, double *y, double delta){
  int nc, *ia, *ja, i, j, k;
  SparseMatrix_multiply_dense(P, x, y, dim);

  interpolate_coord(dim, A, y);
  nc = R->m;
  ia = R->ia;
  ja = R->ja;
  for (i = 0; i < nc; i++){
    for (j = ia[i]+1; j < ia[i+1]; j++){
      for (k = 0; k < dim; k++){
        y[ja[j]*dim + k] += delta*(drand() - 0.5);
      }
    }
  }
}

static bool power_law_graph(SparseMatrix A) {
  int m, max = 0, i, *ia = A->ia, *ja = A->ja, j, deg;
  bool res = false;
  m = A->m;
  int *mask = gv_calloc(m + 1, sizeof(int));

  for (i = 0; i < m + 1; i++){
    mask[i] = 0;
  }

  for (i = 0; i < m; i++){
    deg = 0;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (i == ja[j]) continue;
      deg++;
    }
    mask[deg]++;
    max = MAX(max, mask[deg]);
  }
  if (mask[1] > 0.8*max && mask[1] > 0.3*m) res = true;
  free(mask);
  return res;
}

void pcp_rotate(int n, int dim, double *x){
  int i, k,l;
  double y[4], axis[2], center[2], dist, x0, x1;

  assert(dim == 2);
  for (i = 0; i < dim*dim; i++) y[i] = 0;
  for (i = 0; i < dim; i++) center[i] = 0;
  for (i = 0; i < n; i++){
    for (k = 0; k < dim; k++){
      center[k] += x[i*dim+k];
    }
  }
  for (i = 0; i < dim; i++) center[i] /= n;
  for (i = 0; i < n; i++){
    for (k = 0; k < dim; k++){
      x[dim*i+k] =  x[dim*i+k] - center[k];
    }
  }

  for (i = 0; i < n; i++){
    for (k = 0; k < dim; k++){
      for (l = 0; l < dim; l++){
	y[dim*k+l] += x[i*dim+k]*x[i*dim+l];
      }
    }
  }
  if (y[1] == 0) {
    axis[0] = 0; axis[1] = 1;
  } else {
    /*         Eigensystem[{{x0, x1}, {x1, x3}}] =
	       {{(x0 + x3 - Sqrt[x0^2 + 4*x1^2 - 2*x0*x3 + x3^2])/2,
	       (x0 + x3 + Sqrt[x0^2 + 4*x1^2 - 2*x0*x3 + x3^2])/2},
	       {{-(-x0 + x3 + Sqrt[x0^2 + 4*x1^2 - 2*x0*x3 + x3^2])/(2*x1), 1},
	       {-(-x0 + x3 - Sqrt[x0^2 + 4*x1^2 - 2*x0*x3 + x3^2])/(2*x1), 1}}}
    */
    axis[0] = -(-y[0] + y[3] - sqrt(y[0]*y[0]+4*y[1]*y[1]-2*y[0]*y[3]+y[3]*y[3]))/(2*y[1]);
    axis[1] = 1;
  }
  dist = sqrt(1+axis[0]*axis[0]);
  axis[0] = axis[0]/dist;
  axis[1] = axis[1]/dist;
  for (i = 0; i < n; i++){
    x0 = x[dim*i]*axis[0]+x[dim*i+1]*axis[1];
    x1 = -x[dim*i]*axis[1]+x[dim*i+1]*axis[0];
    x[dim*i] = x0;
    x[dim*i + 1] = x1;

  }


}

static void rotate(int n, int dim, double *x, double angle){
  int i, k;
  double axis[2], center[2], x0, x1;
  double radian = 3.14159/180;

  assert(dim == 2);
  for (i = 0; i < dim; i++) center[i] = 0;
  for (i = 0; i < n; i++){
    for (k = 0; k < dim; k++){
      center[k] += x[i*dim+k];
    }
  }
  for (i = 0; i < dim; i++) center[i] /= n;
  for (i = 0; i < n; i++){
    for (k = 0; k < dim; k++){
      x[dim*i+k] =  x[dim*i+k] - center[k];
    }
  }
  axis[0] = cos(-angle*radian);
  axis[1] = sin(-angle*radian);
  for (i = 0; i < n; i++){
    x0 = x[dim*i]*axis[0]+x[dim*i+1]*axis[1];
    x1 = -x[dim*i]*axis[1]+x[dim*i+1]*axis[0];
    x[dim*i] = x0;
    x[dim*i + 1] = x1;
  }


}

static void attach_edge_label_coordinates(int dim, SparseMatrix A, int n_edge_label_nodes, int *edge_label_nodes, double *x, double *x2){
  int i, ii, j, k;
  int nnodes = 0;
  double len;

  int *mask = gv_calloc(A->m, sizeof(int));

  for (i = 0; i < A->m; i++) mask[i] = 1;
  for (i = 0; i < n_edge_label_nodes; i++) {
    if (edge_label_nodes[i] >= 0 && edge_label_nodes[i] < A->m) mask[edge_label_nodes[i]] = -1;
  }

  for (i = 0; i < A->m; i++) {
    if (mask[i] >= 0) mask[i] = nnodes++;
  }


  for (i = 0; i < A->m; i++){
    if (mask[i] >= 0){
      for (k = 0; k < dim; k++) x[i*dim+k] = x2[mask[i]*dim+k];
    }
  }

  for (i = 0; i < n_edge_label_nodes; i++){
    ii = edge_label_nodes[i];
    len = A->ia[ii+1] - A->ia[ii];
    assert(len >= 2); /* should just be 2 */
    assert(mask[ii] < 0);
    for (k = 0; k < dim; k++) {
      x[ii*dim+k] = 0;
    }
    for (j = A->ia[ii]; j < A->ia[ii+1]; j++){
      for (k = 0; k < dim; k++){
	x[ii*dim+k] += x[(A->ja[j])*dim+k];
      }
    }
    for (k = 0; k < dim; k++) {
      x[ii*dim+k] /= len;
    }
  }

  free(mask);
}

static SparseMatrix shorting_edge_label_nodes(SparseMatrix A, int n_edge_label_nodes, int *edge_label_nodes){
  int i, id = 0, nz, j, jj, ii;
  int *ia = A->ia, *ja = A->ja, *irn = NULL, *jcn = NULL;
  SparseMatrix B;

  int *mask = gv_calloc(A->m, sizeof(int));

  for (i = 0; i < A->m; i++) mask[i] = 1;

  for (i = 0; i < n_edge_label_nodes; i++){
    mask[edge_label_nodes[i]] = -1;
  }

  for (i = 0; i < A->m; i++) {
    if (mask[i] > 0) mask[i] = id++;
  }

  nz  = 0;
  for (i = 0; i < A->m; i++){
    if (mask[i] < 0) continue;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (mask[ja[j]] >= 0) {
	nz++;
	continue;
      }
      ii = ja[j];
      for (jj = ia[ii]; jj < ia[ii+1]; jj++){
	if (ja[jj] != i && mask[ja[jj]] >= 0) nz++;
      }
    }
  }

  if (nz > 0) {
    irn = gv_calloc(nz, sizeof(int));
    jcn = gv_calloc(nz, sizeof(int));
  }

  nz = 0;
  for (i = 0; i < A->m; i++){
    if (mask[i] < 0) continue;
    for (j = ia[i]; j < ia[i+1]; j++){
      if (mask[ja[j]] >= 0) {
	irn[nz] = mask[i];
	jcn[nz++] = mask[ja[j]];
	continue;
      }
      ii = ja[j];
      for (jj = ia[ii]; jj < ia[ii+1]; jj++){
	if (ja[jj] != i && mask[ja[jj]] >= 0) {
	    irn[nz] = mask[i];
	    jcn[nz++] = mask[ja[jj]];
	}
      }
    }
  }

  B = SparseMatrix_from_coordinate_arrays(nz, id, id, irn, jcn, NULL, MATRIX_TYPE_PATTERN, sizeof(double));

  free(irn);
  free(jcn);
  free(mask);
  return B;

}

void multilevel_spring_electrical_embedding(int dim, SparseMatrix A0,
                                            spring_electrical_control ctrl,
                                            double *label_sizes, double *x,
                                            int n_edge_label_nodes,
                                            int *edge_label_nodes, int *flag) {

  int n;
  SparseMatrix A = A0, P = NULL;
  Multilevel grid, grid0;
  double *xc = NULL, *xf = NULL;
  struct spring_electrical_control_struct ctrl0;
#ifdef TIME
  clock_t  cpu;
#endif

  ctrl0 = *ctrl;

#ifdef TIME
  cpu = clock();
#endif

  *flag = 0;
  if (!A) return;
  n = A->n;
  if (n <= 0 || dim <= 0) return;

  if (!SparseMatrix_is_symmetric(A, false) || A->type != MATRIX_TYPE_REAL){
    A = SparseMatrix_get_real_adjacency_matrix_symmetrized(A);
  } else {
    A = SparseMatrix_remove_diagonal(A);
  }

  /* we first generate a layout discarding (shorting) the edge labels nodes, then assign the edge label nodes at the average of their neighbors */
  if ((ctrl->edge_labeling_scheme == ELSCHEME_STRAIGHTLINE_PENALTY || ctrl->edge_labeling_scheme == ELSCHEME_STRAIGHTLINE_PENALTY2)
      && n_edge_label_nodes > 0){
    SparseMatrix A2;

    double *x2 = gv_calloc(A->m * dim, sizeof(double));
    A2 = shorting_edge_label_nodes(A, n_edge_label_nodes, edge_label_nodes);
    multilevel_spring_electrical_embedding(dim, A2, ctrl, NULL, x2, 0, NULL, flag);

    assert(!(*flag));
    attach_edge_label_coordinates(dim, A, n_edge_label_nodes, edge_label_nodes, x, x2);
    remove_overlap(dim, A, x, label_sizes, ctrl->overlap, ctrl->initial_scaling,
		   ctrl->edge_labeling_scheme, n_edge_label_nodes, edge_label_nodes, A, ctrl->do_shrinking);
    SparseMatrix_delete(A2);
    free(x2);
    if (A != A0) SparseMatrix_delete(A);

    return;
  }

  Multilevel_control mctrl = {.maxlevel = ctrl->multilevels};
  grid0 = Multilevel_new(A, mctrl);

  grid = Multilevel_get_coarsest(grid0);
  if (Multilevel_is_finest(grid)){
    xc = x;
  } else {
    xc = gv_calloc(grid->n * dim, sizeof(double));
  }

  const bool plg = power_law_graph(A);
  if (ctrl->p == AUTOP){
    ctrl->p = -1;
    if (plg) ctrl->p = -1.8;
  }

  do {
#ifdef DEBUG_PRINT
    if (Verbose) {
      print_padding(grid->level);
      if (Multilevel_is_coarsest(grid)){
	fprintf(stderr, "coarsest level -- %d, n = %d\n", grid->level, grid->n);
      } else {
	fprintf(stderr, "level -- %d, n = %d\n", grid->level, grid->n);
      }
    }
#endif
    if (ctrl->tscheme == QUAD_TREE_NONE){
      spring_electrical_embedding_slow(dim, grid->A, ctrl, xc, flag);
    } else if (ctrl->tscheme == QUAD_TREE_FAST || (ctrl->tscheme == QUAD_TREE_HYBRID && grid->A->m > QUAD_TREE_HYBRID_SIZE)){
      if (ctrl->tscheme == QUAD_TREE_HYBRID && grid->A->m > 10 && Verbose){
	fprintf(stderr, "QUAD_TREE_HYBRID, size larger than %d, switch to fast quadtree", QUAD_TREE_HYBRID_SIZE);
      }
      spring_electrical_embedding_fast(dim, grid->A, ctrl, xc, flag);
    } else {
      spring_electrical_embedding(dim, grid->A, ctrl, xc, flag);
    }
    if (Multilevel_is_finest(grid)) break;
    if (*flag) {
      free(xc);
      goto RETURN;
    }
    P = grid->P;
    grid = grid->prev;
    if (Multilevel_is_finest(grid)){
      xf = x;
    } else {
      xf = gv_calloc(grid->n * dim, sizeof(double));
    }
    prolongate(dim, grid->A, P, grid->R, xc, xf, (ctrl->K)*0.001);
    free(xc);
    xc = xf;
    ctrl->random_start = false;
    ctrl->K = ctrl->K * 0.75;
    ctrl->adaptive_cooling = false;
    ctrl->step = .1;
  } while (grid);

#ifdef TIME
  if (Verbose)
    fprintf(stderr, "layout time %f\n",((double) (clock() - cpu)) / CLOCKS_PER_SEC);
  cpu = clock();
#endif

  post_process_smoothing(dim, A, ctrl, x);

  if (Verbose) fprintf(stderr, "ctrl->overlap=%d\n",ctrl->overlap);

  /* rotation has to be done before overlap removal, since rotation could induce overlaps */
  if (dim == 2){
    pcp_rotate(n, dim, x);
  }
  if (ctrl->rotation != 0) rotate(n, dim, x, ctrl->rotation);


  remove_overlap(dim, A, x, label_sizes, ctrl->overlap, ctrl->initial_scaling,
		 ctrl->edge_labeling_scheme, n_edge_label_nodes, edge_label_nodes, A, ctrl->do_shrinking);

 RETURN:
  *ctrl = ctrl0;
  if (A != A0) SparseMatrix_delete(A);
  Multilevel_delete(grid0);
 }
