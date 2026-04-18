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
#include <neatogen/overlap.h>
#include <util/alloc.h>

#if ((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))

#include <sparse/SparseMatrix.h>
#include <neatogen/call_tri.h>
#include <rbtree/red_black_tree.h>
#include <common/types.h>
#include <math.h>
#include <common/globals.h>
#include <stdbool.h>
#include <time.h>

static void ideal_distance_avoid_overlap(int dim, SparseMatrix A, double *x, double *width, double *ideal_distance, double *tmax, double *tmin){
  /*  if (x1>x2 && y1 > y2) we want either x1 + t (x1-x2) - x2 > (width1+width2), or y1 + t (y1-y2) - y2 > (height1+height2),
      hence t = MAX(expandmin, MIN(expandmax, (width1+width2)/(x1-x2) - 1, (height1+height2)/(y1-y2) - 1)), and
      new ideal distance = (1+t) old_distance. t can be negative sometimes.
      The result ideal distance is set to negative if the edge needs shrinking
  */
  int i, j, jj;
  int *ia = A->ia, *ja = A->ja;
  double dist, dx, dy, wx, wy, t;
  double expandmax = 1.5, expandmin = 1;

  *tmax = 0;
  *tmin = 1.e10;
  assert(SparseMatrix_is_symmetric(A, false));
  for (i = 0; i < A->m; i++){
    for (j = ia[i]; j < ia[i+1]; j++){
      jj = ja[j];
      if (jj == i) continue;
      dist = distance(x, dim, i, jj);
      dx = fabs(x[i*dim] - x[jj*dim]);
      dy = fabs(x[i*dim+1] - x[jj*dim+1]);
      wx = width[i*dim]+width[jj*dim];
      wy = width[i*dim+1]+width[jj*dim+1];
      if (dx < MACHINEACC*wx && dy < MACHINEACC*wy){
	ideal_distance[j] = hypot(wx, wy);
	*tmax = 2;
      } else {
	if (dx < MACHINEACC*wx){
	  t = wy/dy;
	} else if (dy < MACHINEACC*wy){
	  t = wx/dx;
	} else {
	  t = fmin(wx / dx, wy / dy);
	}
	if (t > 1) t = fmax(t, 1.001);/* no point in things like t = 1.00000001 as this slow down convergence */
	*tmax = fmax(*tmax, t);
	*tmin = fmin(*tmin, t);
	t = fmin(expandmax, t);
	t = fmax(expandmin, t);
	if (t > 1) {
	  ideal_distance[j] = t*dist;
	} else {
	  ideal_distance[j] = -t*dist;
	}
      }

    }
  }
}

enum {INTV_OPEN, INTV_CLOSE};

typedef struct {
  int node;
  double x;
  int status;
} scan_point;

static int comp_scan_points(const void *p, const void *q){
  const scan_point *pp = p;
  const scan_point *qq = q;
  if (pp->x > qq->x){
    return 1;
  } else if (pp->x < qq->x){
    return -1;
  } else {
    if (pp->node > qq->node){
      return 1;
    } else if (pp->node < qq->node){
      return -1;
    }
    return 0;
  }
  return 0;
}

static void NodeDest(void* a) {
  (void)a;
  /*  free((int*)a);*/
}

static SparseMatrix get_overlap_graph(int dim, int n, double *x, double *width, int check_overlap_only){
  /* if check_overlap_only = TRUE, we only check whether there is one overlap */
  int i, k, neighbor;
  SparseMatrix A = NULL, B = NULL;
  rb_red_blk_node *newNode, *newNode0;
  rb_red_blk_tree* treey;
  double one = 1;

  A = SparseMatrix_new(n, n, 1, MATRIX_TYPE_REAL, FORMAT_COORD);

  scan_point *scanpointsx = gv_calloc(2 * n, sizeof(scan_point));
  for (i = 0; i < n; i++){
    scanpointsx[2*i].node = i;
    scanpointsx[2*i].x = x[i*dim] - width[i*dim];
    scanpointsx[2*i].status = INTV_OPEN;
    scanpointsx[2*i+1].node = i+n;
    scanpointsx[2*i+1].x = x[i*dim] + width[i*dim];
    scanpointsx[2*i+1].status = INTV_CLOSE;
  }
  qsort(scanpointsx, 2*n, sizeof(scan_point), comp_scan_points);

  scan_point *scanpointsy = gv_calloc(2 * n, sizeof(scan_point));
  for (i = 0; i < n; i++){
    scanpointsy[i].node = i;
    scanpointsy[i].x = x[i*dim+1] - width[i*dim+1];
    scanpointsy[i].status = INTV_OPEN;
    scanpointsy[i+n].node = i;
    scanpointsy[i+n].x = x[i*dim+1] + width[i*dim+1];
    scanpointsy[i+n].status = INTV_CLOSE;
  }

  treey = RBTreeCreate(comp_scan_points, NodeDest);

  for (i = 0; i < 2*n; i++){
#ifdef DEBUG_RBTREE
    fprintf(stderr," k = %d node = %d x====%f\n",(scanpointsx[i].node)%n, (scanpointsx[i].node), (scanpointsx[i].x));
#endif

    k = (scanpointsx[i].node)%n;


    if (scanpointsx[i].status == INTV_OPEN){
#ifdef DEBUG_RBTREE
      fprintf(stderr, "inserting...");
      treey->PrintKey(&(scanpointsy[k]));
#endif

      RBTreeInsert(treey, &scanpointsy[k]); // add both open and close int for y

#ifdef DEBUG_RBTREE
      fprintf(stderr, "inserting2...");
      treey->PrintKey(&(scanpointsy[k+n]));
#endif

      RBTreeInsert(treey, &scanpointsy[k + n]);
    } else {
      double bsta, bbsta, bsto, bbsto; int ii; 

      assert(scanpointsx[i].node >= n);

      newNode = newNode0 = RBExactQuery(treey, &(scanpointsy[k + n]));
      ii = ((scan_point *)newNode->key)->node;
      assert(ii < n);
      bsta = scanpointsy[ii].x; bsto = scanpointsy[ii+n].x;

#ifdef DEBUG_RBTREE
      fprintf(stderr, "popping..%d....yinterval={%f,%f}\n", scanpointsy[k + n].node, bsta, bsto);
      treey->PrintKey(newNode->key);
#endif

     assert(treey->nil != newNode);
      while ((newNode) && ((newNode = TreePredecessor(treey, newNode)) != treey->nil)){
	neighbor = (((scan_point *)newNode->key)->node)%n;
	bbsta = scanpointsy[neighbor].x; bbsto = scanpointsy[neighbor+n].x;/* the y-interval of the node that has one end of the interval lower than the top of the leaving interval (bsto) */
#ifdef DEBUG_RBTREE
	fprintf(stderr," predecessor is node %d y = %f\n", ((scan_point *)newNode->key)->node, ((scan_point *)newNode->key)->x);
#endif
	if (neighbor != k){
	  if (fabs(0.5*(bsta+bsto) - 0.5*(bbsta+bbsto)) < 0.5*(bsto-bsta) + 0.5*(bbsto-bbsta)){/* if the distance of the centers of the interval is less than sum of width, we have overlap */
	    A = SparseMatrix_coordinate_form_add_entry(A, neighbor, k, &one);
#ifdef DEBUG_RBTREE
	    fprintf(stderr,"======================================  %d %d\n",k,neighbor);
#endif
	    if (check_overlap_only) goto check_overlap_RETURN;
	  }
	}
      }

#ifdef DEBUG_RBTREE
      fprintf(stderr, "deleting...");
      treey->PrintKey(newNode0->key);
#endif

      if (newNode0) RBDelete(treey,newNode0);
    }
  }

check_overlap_RETURN:
   free(scanpointsx);
  free(scanpointsy);
  RBTreeDestroy(treey);

  B = SparseMatrix_from_coordinate_format(A);
  SparseMatrix_delete(A);
  A = SparseMatrix_symmetrize(B, false);
  SparseMatrix_delete(B);
  if (Verbose) fprintf(stderr, "found %d clashes\n", A->nz);
  return A;
}



/* ============================== label overlap smoother ==================*/


static void relative_position_constraints_delete(void *d){
  if (!d) return;
  relative_position_constraints data = d;
  free(data->irn);
  free(data->jcn);
  free(data->val);
  /* other stuff inside relative_position_constraints is passed back to the user hence no need to deallocator*/
  free(d);
}

static relative_position_constraints relative_position_constraints_new(SparseMatrix A_constr, int edge_labeling_scheme, int n_constr_nodes, int *constr_nodes){
    assert(A_constr);
    relative_position_constraints data = gv_alloc(sizeof(struct relative_position_constraints_struct));
    data->constr_penalty = 1;
    data->edge_labeling_scheme = edge_labeling_scheme;
    data->n_constr_nodes = n_constr_nodes;
    data->constr_nodes = constr_nodes;
    data->A_constr = A_constr;
    data->irn = NULL;
    data->jcn = NULL;
    data->val = NULL;

    return data;
}
static void scale_coord(int dim, int m, double *x, double scale){
  int i;
  for (i = 0; i < dim*m; i++) {
    x[i] *= scale;
  }
}

static double overlap_scaling(int dim, int m, double *x, double *width,
                              double scale_sta, double scale_sto,
                              double epsilon, int maxiter) {
  /* do a bisection between scale_sta and scale_sto, up to maxiter iterations or till interval <= epsilon, to find the best scaling to avoid overlap
     m: number of points
     x: the coordinates
     width: label size
     scale_sta: starting bracket. If <= 0, assumed 0. If > 0, we will test this first and if no overlap, return.
     scale_sto: stopping bracket. This must be overlap free if positive. If <= 0, we will find automatically by doubling from scale_sta, or epsilon if scale_sta <= 0.
     typically usage: 
     - for shrinking down a layout to reduce white space, we will assume scale_sta and scale_sto are both given and positive, and scale_sta is the current guess.
     - for scaling up, we assume scale_sta, scale_sto <= 0
   */
  double scale = -1, scale_best = -1;
  SparseMatrix C = NULL;
  int check_overlap_only = 1;
  int overlap = 0;
  double two = 2;
  int iter = 0;

  assert(epsilon > 0);

  if (scale_sta <= 0) {
    scale_sta = 0;
  } else {
    scale_coord(dim, m, x, scale_sta);
    C = get_overlap_graph(dim, m, x, width, check_overlap_only);
    if (!C || C->nz == 0) {
      if (Verbose) fprintf(stderr," shrinking with %f works\n", scale_sta);
      SparseMatrix_delete(C);
      return scale_sta;
    }
    scale_coord(dim, m, x, 1./scale_sta);
    SparseMatrix_delete(C);
  }

  if (scale_sto < 0){
    if (scale_sta == 0) {
      scale_sto = epsilon;
    } else {
      scale_sto = scale_sta;
    }
    scale_coord(dim, m, x, scale_sto);
    do {
      scale_sto *= two;
      scale_coord(dim, m, x, two);
      C = get_overlap_graph(dim, m, x, width, check_overlap_only);
      overlap = (C && C->nz > 0);
      SparseMatrix_delete(C);
    } while (overlap);
    scale_coord(dim, m, x, 1/scale_sto);/* unscale */
  }

  scale_best = scale_sto;
  while (iter++ < maxiter && scale_sto - scale_sta > epsilon){

    if (Verbose) fprintf(stderr,"in overlap_scaling iter=%d, maxiter=%d, scaling bracket: {%f,%f}\n", iter, maxiter, scale_sta, scale_sto);

    scale = 0.5*(scale_sta + scale_sto);
    scale_coord(dim, m, x, scale);
    C = get_overlap_graph(dim, m, x, width, check_overlap_only);
    scale_coord(dim, m, x, 1./scale);/* unscale */
    overlap = (C && C->nz > 0);
    SparseMatrix_delete(C);
    if (overlap){
      scale_sta = scale;
    } else {
      scale_best = scale_sto = scale;
    }
  }

  /* final scaling */
  scale_coord(dim, m, x, scale_best);
  return scale_best;
}
 
OverlapSmoother OverlapSmoother_new(SparseMatrix A, int m, 
				    int dim, double *x, double *width, bool neighborhood_only,
				    double *max_overlap, double *min_overlap,
				    int edge_labeling_scheme, int n_constr_nodes, int *constr_nodes, SparseMatrix A_constr, int shrink
				    ){
  int i, j, k, *iw, *jw, jdiag;
  SparseMatrix B;
  double *d, *w, diag_d, diag_w, dist;

  assert((!A) || SparseMatrix_is_symmetric(A, false));

  OverlapSmoother sm = gv_alloc(sizeof(struct OverlapSmoother_struct));
  sm->scheme = SM_SCHEME_NORMAL;
  if (constr_nodes && n_constr_nodes > 0 && edge_labeling_scheme != ELSCHEME_NONE){
    sm->scheme = SM_SCHEME_NORMAL_ELABEL;
    sm->data = relative_position_constraints_new(A_constr, edge_labeling_scheme, n_constr_nodes, constr_nodes);
    sm->data_deallocator = relative_position_constraints_delete;
  } else {
    sm->data = NULL;
  }

  sm->tol_cg = 0.01;
  sm->maxit_cg = floor(sqrt(A->m));

  sm->lambda = gv_calloc(m, sizeof(double));
  
  B= call_tri(m, x);

  if (!neighborhood_only){
    SparseMatrix C, D;
    C = get_overlap_graph(dim, m, x, width, 0);
    D = SparseMatrix_add(B, C);
    SparseMatrix_delete(B);
    SparseMatrix_delete(C);
    B = D;
  }
  sm->Lw = B;
  sm->Lwd = SparseMatrix_copy(sm->Lw);

  if (!(sm->Lw) || !(sm->Lwd)) {
    OverlapSmoother_delete(sm);
    return NULL;
  }

  assert((sm->Lwd)->type == MATRIX_TYPE_REAL);
  
  ideal_distance_avoid_overlap(dim, sm->Lwd, x, width, sm->Lwd->a, max_overlap, min_overlap);

  /* no overlap at all! */
  if (*max_overlap < 1 && shrink){
    double scale_sta = fmin(1, *max_overlap * 1.0001);
    const double scale_sto = 1;

    if (Verbose) fprintf(stderr," no overlap (overlap = %f), rescale to shrink\n", *max_overlap - 1);

    scale_sta = overlap_scaling(dim, m, x, width, scale_sta, scale_sto, 0.0001, 15);

    *max_overlap = 1;
    goto RETURN;
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
      if (d[j] > 0){/* those edges that needs expansion */
	w[j] = -100/d[j]/d[j];
      } else {/* those that needs shrinking is set to negative in ideal_distance_avoid_overlap */
	w[j] = -1/d[j]/d[j];
	d[j] = -d[j];
      }
      dist = d[j];
      diag_w += w[j];
      d[j] = w[j]*dist;
      diag_d += d[j];

    }

    assert(jdiag >= 0);
    w[jdiag] = -diag_w;
    d[jdiag] = -diag_d;
  }
 RETURN:
  return sm;
}

void OverlapSmoother_delete(OverlapSmoother sm){

  StressMajorizationSmoother_delete(sm);

}

double OverlapSmoother_smooth(OverlapSmoother sm, int dim, double *x){
  int maxit_sm = 1;/* only using 1 iteration of stress majorization 
		      is found to give better results and save time! */
  return StressMajorizationSmoother_smooth(sm, dim, x, maxit_sm);
}

/*================================= end OverlapSmoother =============*/

static void scale_to_edge_length(int dim, SparseMatrix A, double *x, double avg_label_size){
  double dist;
  int i;

  if (!A) return;
  dist = average_edge_length(A, dim, x);
  if (Verbose) fprintf(stderr,"avg edge len=%f avg_label-size= %f\n", dist, avg_label_size);


  dist = avg_label_size / fmax(dist, MACHINEACC);

  for (i = 0; i < dim*A->m; i++) x[i] *= dist;
}

static void print_bounding_box(int n, int dim, double *x){
  int i, k;

  double *xmin = gv_calloc(dim, sizeof(double));
  double *xmax = gv_calloc(dim, sizeof(double));

  for (i = 0; i < dim; i++) xmin[i]=xmax[i] = x[i];

  for (i = 0; i < n; i++){
    for (k = 0; k < dim; k++){
      xmin[k] = fmin(xmin[k], x[i * dim + k]);
      xmax[k] = fmax(xmax[k], x[i * dim + k]);
    }
  }
  fprintf(stderr,"bounding box = \n");
  for (i = 0; i < dim; i++) fprintf(stderr,"{%f,%f}, ",xmin[i], xmax[i]);
  fprintf(stderr,"\n");

  free(xmin);
  free(xmax);
}

static int check_convergence(double max_overlap, double res,
                             bool has_penalty_terms, double epsilon) {
  if (!has_penalty_terms)
    return (max_overlap <= 1);
  return res < epsilon;
}

void remove_overlap(int dim, SparseMatrix A, double *x, double *label_sizes, int ntry, double initial_scaling, 
		    int edge_labeling_scheme, int n_constr_nodes, int *constr_nodes, SparseMatrix A_constr, bool do_shrinking) {
  /* 
     edge_labeling_scheme: if ELSCHEME_NONE, n_constr_nodes/constr_nodes/A_constr are not used

     n_constr_nodes: number of nodes that has constraints, these are nodes that is
     .               constrained to be close to the average of its neighbors.
     constr_nodes: a list of nodes that need to be constrained. If NULL, unused.
     A_constr: neighbors of node i are in the row i of this matrix. i needs to sit
     .         in between these neighbors as much as possible. this must not be NULL
     .         if constr_nodes != NULL.

  */

  OverlapSmoother sm;
  int i;
  double LARGE = 100000;
  double avg_label_size, res = LARGE;
  double max_overlap = 0, min_overlap = 999;
  bool neighborhood_only = true;
  double epsilon = 0.005;
  int shrink = 0;

#ifdef TIME
  clock_t  cpu;
#endif

#ifdef TIME
  cpu = clock();
#endif

  if (!label_sizes) return;

  if (initial_scaling < 0) {
    avg_label_size = 0;
    for (i = 0; i < A->m; i++) avg_label_size += label_sizes[i*dim]+label_sizes[i*dim+1];
    avg_label_size /= A->m;
    scale_to_edge_length(dim, A, x, -initial_scaling*avg_label_size);
  } else if (initial_scaling > 0){
    scale_to_edge_length(dim, A, x, initial_scaling);
  }

  if (!ntry) return;

#ifdef DEBUG
  _statistics[0] = _statistics[1] = 0.;
#endif

  bool has_penalty_terms =
      edge_labeling_scheme != ELSCHEME_NONE && n_constr_nodes > 0;
  for (i = 0; i < ntry; i++){
    if (Verbose) print_bounding_box(A->m, dim, x);
    sm = OverlapSmoother_new(A, A->m, dim, x, label_sizes, neighborhood_only,
			     &max_overlap, &min_overlap, edge_labeling_scheme, n_constr_nodes, constr_nodes, A_constr, shrink); 
    if (Verbose) {
      fprintf(stderr,
              "overlap removal neighbors only?= %d iter -- %d, overlap factor = %g underlap factor = %g\n",
              (int)neighborhood_only, i, max_overlap - 1, min_overlap);
    }
    if (check_convergence(max_overlap, res, has_penalty_terms, epsilon)) {
    
      OverlapSmoother_delete(sm);
      if (!neighborhood_only){
	break;
      } else {
	res = LARGE;
	neighborhood_only = false;
        if (do_shrinking) {
          shrink = 1;
        }
	continue;
      }
    }
    
    res = OverlapSmoother_smooth(sm, dim, x);
    if (Verbose) fprintf(stderr,"res = %f\n",res);
    OverlapSmoother_delete(sm);
  }
  if (Verbose) {
    fprintf(stderr,
            "overlap removal neighbors only?= %d iter -- %d, overlap factor = %g underlap factor = %g\n",
            (int)neighborhood_only, i, max_overlap - 1, min_overlap);
  }

  if (has_penalty_terms){
    /* now do without penalty */
    remove_overlap(dim, A, x, label_sizes, ntry, 0.,
		   ELSCHEME_NONE, 0, NULL, NULL, do_shrinking);
  }

#ifdef DEBUG
  fprintf(stderr," number of cg iter = %f, number of stress majorization iter = %f number of overlap removal try = %d\n",
	  _statistics[0], _statistics[1], i - 1);
#endif

#ifdef TIME
  fprintf(stderr, "post processing %f\n",((double) (clock() - cpu)) / CLOCKS_PER_SEC);
#endif
}

#else
#include <common/types.h>
#include <sparse/SparseMatrix.h>
void remove_overlap(int dim, SparseMatrix A, double *x, double *label_sizes, int ntry, double initial_scaling,
		    int edge_labeling_scheme, int n_constr_nodes, int *constr_nodes, SparseMatrix A_constr, bool do_shrinking)
{
    static int once;

    (void)dim;
    (void)A;
    (void)x;
    (void)label_sizes;
    (void)ntry;
    (void)initial_scaling;
    (void)edge_labeling_scheme;
    (void)n_constr_nodes;
    (void)constr_nodes;
    (void)A_constr;
    (void)do_shrinking;

    if (once == 0) {
	once = 1;
	agerrorf("remove_overlap: Graphviz not built with triangulation library\n");
    }
}
#endif
