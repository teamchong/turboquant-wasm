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

#include <neatogen/sparsegraph.h>

    extern void scadd(double *, int, double, double *);
    extern double norm(double *, int);

    extern void orthog1(int n, double *vec);
    extern void init_vec_orth1(int n, double *vec);
    extern void right_mult_with_vector(vtx_data *, int, double *,
				       double *);
    extern void right_mult_with_vector_f(float **, int, double *,
					 double *);
    extern void vectors_subtraction(int, double *, double *, double *);
    extern void vectors_addition(int, double *, double *, double *);
    extern void vectors_scalar_mult(int, const double *, double, double *);
    extern void copy_vector(int n, const double *source, double *dest);
    extern double vectors_inner_product(int n, const double *vector1,
					const double *vector2);
    extern double max_abs(int n, double *vector);

    /* sparse matrix extensions: */

    extern void right_mult_with_vector_transpose
	(double **, int, int, double *, double *);
    extern void right_mult_with_vector_d(double **, int, int, double *,
					 double *);
    extern void mult_dense_mat(double **, float **, int, int, int,
			       float ***C);
    extern void mult_dense_mat_d(double **, float **, int, int, int,
				 double ***CC);
    extern void mult_sparse_dense_mat_transpose(vtx_data *, double **, int,
						int, float ***);
    extern bool power_iteration(double **, int, int, double **, double *);


/*****************************
** Single precision (float) **
** version                  **
*****************************/

    extern void orthog1f(int n, float *vec);
    extern void right_mult_with_vector_ff(float *, int, float *, float *);
    extern void vectors_subtractionf(int, float *, float *, float *);
    extern void vectors_additionf(int n, float *vector1, float *vector2,
				  float *result);
    extern void vectors_mult_additionf(int n, float *vector1, float alpha,
				       float *vector2);
    extern void copy_vectorf(int n, float *source, float *dest);
    extern double vectors_inner_productf(int n, float *vector1,
					 float *vector2);
    extern void set_vector_val(int n, double val, double *result);
    extern void set_vector_valf(int n, float val, float * result);
    extern double max_absf(int n, float *vector);
    extern void square_vec(int n, float *vec);
    extern void invert_vec(int n, float *vec);
    extern void sqrt_vecf(int n, float *source, float *target);
    extern void invert_sqrt_vec(int n, float *vec);

#ifdef __cplusplus
}
#endif
