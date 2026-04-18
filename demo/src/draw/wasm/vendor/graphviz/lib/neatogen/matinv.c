/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/*
 * This code was (mostly) written by Ken Turkowski, who said:
 *
 * Oh, that. I wrote it in college the first time. It's open source - I think I
 * posted it after seeing so many people solve equations by inverting matrices
 * by computing minors naïvely.
 * -Ken
 *
 * The views represented here are mine and are not necessarily shared by
 * my employer.
   	Ken Turkowski			turk@apple.com
	Immersive Media Technologist 	http://www.worldserver.com/turk/
	Apple Computer, Inc.
	1 Infinite Loop, MS 302-3VR
	Cupertino, CA 95014
 */

/* Matinv() inverts the matrix A using LU decomposition.  Arguments:
 *	A    - the (n x n) matrix to be inverted
 *	Ainv - the (n x n) inverted matrix
 *	n    - the order of the matrices A and Ainv
 */

#include <stdlib.h>
#include <common/render.h>
#include <neatogen/neato.h>
#include <util/alloc.h>

int matinv(double **A, double **Ainv, int n)
{
    int i, j;
    double temp;

    /* Decompose matrix into L and U triangular matrices */
    if (lu_decompose(A, n) == 0)
	return (0);		/* Singular */

    /* Invert matrix by solving n simultaneous equations n times */
    double *b = gv_calloc(n, sizeof(double));
    for (i = 0; i < n; i++) {
	for (j = 0; j < n; j++)
	    b[j] = 0.0;
	b[i] = 1.0;
	lu_solve(Ainv[i], b, n);	/* Into a row of Ainv: fix later */
    }
    free(b);

    /* Transpose matrix */
    for (i = 0; i < n; i++) {
	for (j = 0; j < i; j++) {
	    temp = Ainv[i][j];
	    Ainv[i][j] = Ainv[j][i];
	    Ainv[j][i] = temp;
	}
    }

    return (1);
}
