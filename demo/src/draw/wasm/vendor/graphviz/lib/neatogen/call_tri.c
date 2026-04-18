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
#include <sparse/SparseMatrix.h>
#include <neatogen/call_tri.h>
#include <neatogen/delaunay.h>
#include <stddef.h>
#include <stdbool.h>
#include <util/alloc.h>

SparseMatrix call_tri(int n, double *x) {
    double one = 1;
    int i, ii, jj;
    SparseMatrix A;
    SparseMatrix B;
    int* edgelist = NULL;
    double* xv = gv_calloc(n, sizeof(double));
    double* yv = gv_calloc(n, sizeof(double));
    int numberofedges = 0;

    for (i = 0; i < n; i++) {
	xv[i] = x[i * 2];
	yv[i] = x[i * 2 + 1];
    }

    if (n > 2) {
	edgelist = delaunay_tri (xv, yv, n, &numberofedges);
    }

    A = SparseMatrix_new(n, n, 1, MATRIX_TYPE_REAL, FORMAT_COORD);
    for (i = 0; i < numberofedges; i++) {
	ii = edgelist[i * 2];
	jj = edgelist[i * 2 + 1];
	SparseMatrix_coordinate_form_add_entry(A, ii, jj, &one);
    }
    if (n == 2) {		/* if two points, add edge i->j */
	ii = 0;
	jj = 1;
	SparseMatrix_coordinate_form_add_entry(A, ii, jj, &one);
    }
    for (i = 0; i < n; i++) {
	SparseMatrix_coordinate_form_add_entry(A, i, i, &one);
    }
    B = SparseMatrix_from_coordinate_format(A);
    SparseMatrix_delete(A);
    A = SparseMatrix_symmetrize(B, false);
    SparseMatrix_delete(B);
    B = A;

    free (edgelist);
    free (xv);
    free (yv);
    return B;
}

SparseMatrix call_tri2(int n, int dim, double * xx)
{
    v_data *delaunay;
    int i, j;
    SparseMatrix A;
    SparseMatrix B;
    double one = 1;
    double *x = gv_calloc(n, sizeof(double));
    double *y = gv_calloc(n, sizeof(double));

    for (i = 0; i < n; i++) {
	x[i] = xx[dim * i];
	y[i] = xx[dim * i + 1];
    }

    delaunay = UG_graph(x, y, n);

    A = SparseMatrix_new(n, n, 1, MATRIX_TYPE_REAL, FORMAT_COORD);

    for (i = 0; i < n; i++) {
	for (j = 1; j < delaunay[i].nedges; j++) {
	    SparseMatrix_coordinate_form_add_entry(A, i, delaunay[i].edges[j], &one);
	}
    }
    for (i = 0; i < n; i++) {
	SparseMatrix_coordinate_form_add_entry(A, i, i, &one);
    }
    B = SparseMatrix_from_coordinate_format(A);
    {
	SparseMatrix tmp = SparseMatrix_symmetrize(B, false);
	SparseMatrix_delete(B);
	B = tmp;
    }
    SparseMatrix_delete(A);

    free (x);
    free (y);
    freeGraph (delaunay);
    
    return B;

}
 
