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

#include <sparse/general.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYMMETRY_EPSILON 0.0000001
enum {FORMAT_CSR, FORMAT_COORD};
enum {UNMASKED = -10, MASKED = 1};
enum {BIPARTITE_RECT = 0, BIPARTITE_PATTERN_UNSYM, BIPARTITE_UNSYM, BIPARTITE_ALWAYS};


struct SparseMatrix_struct {
  int m; /* row dimension */
  int n; /* column dimension */
  int nz;/* The actual length used is nz, for CSR/CSC matrix this is the same as ia[n] */
  int nzmax; /* the current length of ja and a (if exists) allocated.*/
  int type; /* whether it is real/complex matrix, or pattern only */
  int *ia; /* row pointer for CSR format, or row indices for coordinate format. 0-based */
  int *ja; /* column indices. 0-based */
  void *a; /* entry values. If NULL, pattern matrix */
  int format;/* whether it is CSR, CSC, COORD. By default it is in CSR format */
  bool is_pattern_symmetric :1;
  bool is_symmetric :1;
  bool is_undirected :1;
  size_t size;/* size of each entry. This allows for general matrix where each entry is, say, a matrix itself */
};

typedef struct SparseMatrix_struct* SparseMatrix;

enum {MATRIX_TYPE_REAL = 1<<0, MATRIX_TYPE_COMPLEX = 1<<1, MATRIX_TYPE_INTEGER = 1<<2, MATRIX_TYPE_PATTERN = 1<<3, MATRIX_TYPE_UNKNOWN = 1<<4};

/* SparseMatrix_general is more general and allow elements to be 
   any data structure, not just real/int/complex etc */
SparseMatrix SparseMatrix_new(int m, int n, int nz, int type, int format);
SparseMatrix SparseMatrix_general_new(int m, int n, int nz, int type, size_t sz, int format);

/* this version sum repeated entries */
SparseMatrix SparseMatrix_from_coordinate_format(SparseMatrix A);
SparseMatrix SparseMatrix_from_coordinate_format_not_compacted(SparseMatrix A);

SparseMatrix SparseMatrix_from_coordinate_arrays(int nz, int m, int n, int *irn, int *jcn, void *val, int type, size_t sz);
SparseMatrix SparseMatrix_from_coordinate_arrays_not_compacted(int nz, int m, int n, int *irn, int *jcn, void *val, int type, size_t sz);

void SparseMatrix_export(FILE *f, SparseMatrix A);/* export into MM format except the header */

void SparseMatrix_delete(SparseMatrix A);

SparseMatrix SparseMatrix_add(SparseMatrix A, SparseMatrix B);
SparseMatrix SparseMatrix_multiply(SparseMatrix A, SparseMatrix B);
SparseMatrix SparseMatrix_multiply3(SparseMatrix A, SparseMatrix B, SparseMatrix C);

enum {SUM_REPEATED_NONE = 0, SUM_REPEATED_ALL, };
SparseMatrix SparseMatrix_sum_repeat_entries(SparseMatrix A);
SparseMatrix SparseMatrix_coordinate_form_add_entry(SparseMatrix A, int irn,
                                                    int jcn, const void *val);
bool SparseMatrix_is_symmetric(SparseMatrix A, bool test_pattern_symmetry_only);
SparseMatrix SparseMatrix_transpose(SparseMatrix A);
SparseMatrix SparseMatrix_symmetrize(SparseMatrix A,
                                     bool pattern_symmetric_only);
void SparseMatrix_multiply_vector(SparseMatrix A, double *v, double **res);/* if v = NULL, v is assumed to be {1,1,...,1}*/
SparseMatrix SparseMatrix_remove_diagonal(SparseMatrix A);
SparseMatrix SparseMatrix_remove_upper(SparseMatrix A);/* remove diag and upper diag */
SparseMatrix SparseMatrix_divide_row_by_degree(SparseMatrix A);
SparseMatrix SparseMatrix_get_real_adjacency_matrix_symmetrized(SparseMatrix A);  /* symmetric, all entries to 1, diaginal removed */
void SparseMatrix_multiply_dense(SparseMatrix A, const double *v, double *res,
                                 int dim);
SparseMatrix SparseMatrix_apply_fun(SparseMatrix A, double (*fun)(double x));/* for real only! */
SparseMatrix SparseMatrix_copy(SparseMatrix A);
bool SparseMatrix_has_diagonal(SparseMatrix A);
SparseMatrix SparseMatrix_make_undirected(SparseMatrix A);/* make it strictly low diag only, and set flag to undirected */
int *SparseMatrix_weakly_connected_components(SparseMatrix A0, int *ncomp,
                                              int **comps);
void SparseMatrix_decompose_to_supervariables(SparseMatrix A, int *ncluster, int **cluster, int **clusterp);
SparseMatrix SparseMatrix_get_submatrix(SparseMatrix A, int nrow, int ncol, int *rindices, int *cindices);

SparseMatrix SparseMatrix_get_augmented(SparseMatrix A);

/* bipartite_options:
   BIPARTITE_RECT -- turn rectangular matrix into square), 
   BIPARTITE_PATTERN_UNSYM -- pattern unsummetric as bipartite
   BIPARTITE_UNSYM -- unsymmetric as square
   BIPARTITE_ALWAYS -- always as square
*/
SparseMatrix SparseMatrix_to_square_matrix(SparseMatrix A, int bipartite_options);

SparseMatrix SparseMatrix_sort(SparseMatrix A);

SparseMatrix SparseMatrix_set_entries_to_real_one(SparseMatrix A);

void SparseMatrix_distance_matrix(SparseMatrix A, double **dist_matrix);

SparseMatrix SparseMatrix_from_dense(int m, int n, double *x);

#ifdef __cplusplus
}
#endif
