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

#include <sfdpgen/spring_electrical.h>
#include <stdbool.h>

enum {SM_SCHEME_NORMAL, SM_SCHEME_NORMAL_ELABEL, SM_SCHEME_STRESS};

struct StressMajorizationSmoother_struct {
  SparseMatrix D;/* distance matrix. The diagonal is removed hence the ia, ja structure is different from Lw and Lwd!! */
  SparseMatrix Lw;/* the weighted laplacian. with offdiag = -1/w_ij */
  SparseMatrix Lwd;/* the laplacian like matrix with offdiag = -scaling*d_ij/w_ij. RHS in stress majorization = Lwd.x */
  double* lambda;
  void (*data_deallocator)(void*);
  void *data;
  int scheme;
  double scaling;/* scaling. It is multiplied to Lwd. need to divide coordinate x at the end of the stress majorization process */
  double tol_cg;/* tolerance and maxit for conjugate gradient that solves the Laplacian system.
		 typically the Laplacian only needs to be solved very crudely as it is part of an
		 outer iteration.*/
  double maxit_cg;
};

typedef struct StressMajorizationSmoother_struct *StressMajorizationSmoother;

void StressMajorizationSmoother_delete(StressMajorizationSmoother sm);

enum {IDEAL_GRAPH_DIST, IDEAL_AVG_DIST, IDEAL_POWER_DIST};
StressMajorizationSmoother StressMajorizationSmoother2_new(SparseMatrix A, int dim, double lambda, double *x, int ideal_dist_scheme);

double StressMajorizationSmoother_smooth(StressMajorizationSmoother sm, int dim, double *x, int maxit);
/*-------------------- triangle/neirhborhood graph based smoother ------------------- */
typedef  StressMajorizationSmoother TriangleSmoother;

#define TriangleSmoother_struct StressMajorizationSmoother_struct

void TriangleSmoother_delete(TriangleSmoother sm);

TriangleSmoother TriangleSmoother_new(SparseMatrix A, int dim, double *x,
                                      bool use_triangularization);

void TriangleSmoother_smooth(TriangleSmoother sm, int dim, double *x);



/*------------------ spring and spring-electrical based smoother */

struct SpringSmoother_struct {
  SparseMatrix D;
  spring_electrical_control ctrl;
};

typedef struct SpringSmoother_struct *SpringSmoother;

SpringSmoother SpringSmoother_new(SparseMatrix A, int dim, spring_electrical_control ctrl, double *x);

void SpringSmoother_delete(SpringSmoother sm);

void SpringSmoother_smooth(SpringSmoother sm, SparseMatrix A, int dim, double *x);
/*------------------------------------------------------------------*/

void post_process_smoothing(int dim, SparseMatrix A, spring_electrical_control ctrl, double *x);

/*-------------------- sparse stress majorizationp ------------------- */
typedef  StressMajorizationSmoother SparseStressMajorizationSmoother;

#define SparseStressMajorizationSmoother_struct StressMajorizationSmoother_struct

void SparseStressMajorizationSmoother_delete(SparseStressMajorizationSmoother sm);

SparseStressMajorizationSmoother
SparseStressMajorizationSmoother_new(SparseMatrix A, int dim, double *x);

double SparseStressMajorizationSmoother_smooth(SparseStressMajorizationSmoother sm, int dim, double *x, int maxit_sm);

/*--------------------------------------------------------------*/
