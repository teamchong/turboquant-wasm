/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <neatogen/matrix_ops.h>
#include <neatogen/pca.h>
#include <neatogen/closest.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <util/alloc.h>

static int num_pairs = 4;

void
PCA_alloc(DistType ** coords, int dim, int n, double **new_coords,
	  int new_dim)
{
    double sum;
    int i, j, k;

    double **eigs = gv_calloc(new_dim, sizeof(double *));
    for (i = 0; i < new_dim; i++)
	eigs[i] = gv_calloc(dim, sizeof(double));
    double *evals = gv_calloc(new_dim, sizeof(double));

    double **DD = gv_calloc(dim, sizeof(double *)); // dim×dim matrix: coords×coordsᵀ
    double *storage_ptr = gv_calloc(dim * dim, sizeof(double));
    for (i = 0; i < dim; i++) {
	DD[i] = storage_ptr;
	storage_ptr += dim;
    }

    for (i = 0; i < dim; i++) {
	for (j = 0; j <= i; j++) {
	    /* compute coords[i]*coords[j] */
	    sum = 0;
	    for (k = 0; k < n; k++) {
		sum += coords[i][k] * coords[j][k];
	    }
	    DD[i][j] = DD[j][i] = sum;
	}
    }

    power_iteration(DD, dim, new_dim, eigs, evals);

    for (j = 0; j < new_dim; j++) {
	for (i = 0; i < n; i++) {
	    sum = 0;
	    for (k = 0; k < dim; k++) {
		sum += coords[k][i] * eigs[j][k];
	    }
	    new_coords[j][i] = sum;
	}
    }

    for (i = 0; i < new_dim; i++)
	free(eigs[i]);
    free(eigs);
    free(evals);
    free(DD[0]);
    free(DD);
}

bool iterativePCA_1D(double **coords, int dim, int n, double *new_direction) {
    vtx_data *laplacian;
    float **mat1 = NULL;
    double **mat = NULL;
    double eval;

    /* Given that first projection of 'coords' is 'coords[0]'
       compute another projection direction 'new_direction'
       that scatters points that are close in 'coords[0]'
     */

    /* find the nodes that were close in 'coords[0]' */
    /* and construct appropriate Laplacian */
    closest_pairs2graph(coords[0], n, num_pairs * n, &laplacian);

    /* Compute coords*Lap*coords^T */
    mult_sparse_dense_mat_transpose(laplacian, coords, n, dim, &mat1);
    mult_dense_mat_d(coords, mat1, dim, n, dim, &mat);
    free(mat1[0]);
    free(mat1);

    /* Compute direction */
    return power_iteration(mat, dim, 1, &new_direction, &eval);
/* ?? When is mat freed? */
}
