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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <util/exit.h>
/* Applications that do not use the common library can define STANDALONE
 * to get definitions/definitions that are normally provided there.
 * In particular, note that Verbose is declared but undefined.
 */
#ifndef STANDALONE
#include "cgraph.h"
#include "globals.h"
#include "arith.h"
#endif  /* STANDALONE */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef STANDALONE

#define MAX(a,b) ((a)>(b)?(a):b)
#define MIN(a,b) ((a)<(b)?(a):b)

#define POINTS(inch) 72*(inch)

#ifdef GVDLL
__declspec(dllimport) extern unsigned char Verbose;
#else
extern unsigned char Verbose;
#endif

#endif    /* STANDALONE */

#ifdef DEBUG
extern double _statistics[10];
#endif


extern int irand(int n);
extern double drand(void);
extern int *random_permutation(int n);/* random permutation of 0 to n-1 */


double* vector_subtract_to(int n, double *x, double *y);/* y = x-y */

double vector_product(int n, double *x, double *y);

double* vector_saxpy(int n, double *x, double *y, double beta); /* y = x+beta*y */


double* vector_saxpy2(int n, double *x, double *y, double beta);/* x = x+beta*y */

/* take m elements v[p[i]]],i=1,...,m and oput in u. u will be assigned if *u = NULL */
void vector_float_take(int n, float *v, int m, int *p, float **u);

/* give the position of the smallest, second smallest etc in vector v.
   results in p. If *p == NULL, p is assigned.
*/
void vector_ordering(int n, double *v, int **p);
void vector_sort_int(int n, int *v);

#define MACHINEACC 1.0e-16
#define SQRT_MACHINEACC 1.0e-8

#define MINDIST 1.e-15

enum {UNMATCHED = -1};


double distance(double *x, int dim, int i, int j);
double distance_cropped(double *x, int dim, int i, int j);

double point_distance(double *p1, double *p2, int dim);

char *strip_dir(char *s);

#ifdef __cplusplus
}
#endif
