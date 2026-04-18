#include <sparse/general.h>
#include <sparse/SparseMatrix.h>
#include <sfdpgen/spring_electrical.h>
#include <sfdpgen/post_process.h>
#include <sfdpgen/stress_model.h>
#include <stdbool.h>
#include <util/alloc.h>

void stress_model(int dim, SparseMatrix B, double **x, int maxit_sm, int *flag) {
  int m;
  int i;
  SparseMatrix A = B;

  if (!SparseMatrix_is_symmetric(A, false) || A->type != MATRIX_TYPE_REAL){
    if (A->type == MATRIX_TYPE_REAL){
      A = SparseMatrix_symmetrize(A, false);
      A = SparseMatrix_remove_diagonal(A);
    } else {
      A = SparseMatrix_get_real_adjacency_matrix_symmetrized(A);
    } 
  }
  A = SparseMatrix_remove_diagonal(A);

  *flag = 0;
  m = A->m;
  if (!x) {
    *x = gv_calloc(m * dim, sizeof(double));
    srand(123);
    for (i = 0; i < dim*m; i++) (*x)[i] = drand();
  }

  SparseStressMajorizationSmoother sm =
    SparseStressMajorizationSmoother_new(A, dim, *x);/* weight the long distances */

  if (!sm) {
    *flag = -1;
    goto RETURN;
  }


  sm->tol_cg = 0.1; /* we found that there is no need to solve the Laplacian accurately */
  sm->scheme = SM_SCHEME_STRESS;
  SparseStressMajorizationSmoother_smooth(sm, dim, *x, maxit_sm);
  for (i = 0; i < dim*m; i++) {
    (*x)[i] /= sm->scaling;
  }
  SparseStressMajorizationSmoother_delete(sm);

 RETURN:
  if (A != B) SparseMatrix_delete(A);
}
