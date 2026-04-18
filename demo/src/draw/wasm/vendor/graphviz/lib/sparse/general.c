/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <sparse/general.h>
#include <errno.h>
#include <util/alloc.h>

#ifdef DEBUG
double _statistics[10];
#endif

double drand(void){
  return rand()/(double) RAND_MAX;
}

int irand(int n){
  /* 0, 1, ..., n-1 */
  assert(n > 1);
  /*return (int) MIN(floor(drand()*n),n-1);*/
  return rand()%n;
}

int *random_permutation(int n){
  int i, j, pp, len;
  if (n <= 0) return NULL;
  int *p = gv_calloc(n, sizeof(int));
  for (i = 0; i < n; i++) p[i] = i;

  len = n;
  while (len > 1){
    j = irand(len);
    pp = p[len-1];
    p[len-1] = p[j];
    p[j] = pp;
    len--;
  }
  return p;
}


double* vector_subtract_to(int n, double *x, double *y){
  /* y = x-y */
  int i;
  for (i = 0; i < n; i++) y[i] = x[i] - y[i];
  return y;
}
double vector_product(int n, double *x, double *y){
  double res = 0;
  int i;
  for (i = 0; i < n; i++) res += x[i]*y[i];
  return res;
}

double* vector_saxpy(int n, double *x, double *y, double beta){
  /* y = x+beta*y */
  int i;
  for (i = 0; i < n; i++) y[i] = x[i] + beta*y[i];
  return y;
}

double* vector_saxpy2(int n, double *x, double *y, double beta){
  /* x = x+beta*y */
  int i;
  for (i = 0; i < n; i++) x[i] = x[i] + beta*y[i];
  return x;
}

void vector_float_take(int n, float *v, int m, int *p, float **u){
  /* take m elements v[p[i]]],i=1,...,m and oput in u */
  int i;

  if (!*u) *u = gv_calloc(m, sizeof(float));

  for (i = 0; i < m; i++) {
    assert(p[i] < n && p[i] >= 0);
    (*u)[i] = v[p[i]];
  }
  
}

static int comp_ascend(const void *s1, const void *s2){
  const double *ss1 = s1;
  const double *ss2 = s2;

  if (ss1[0] > ss2[0]){
    return 1;
  } else if (ss1[0] < ss2[0]){
    return -1;
  }
  return 0;
}

static int comp_ascend_int(const void *s1, const void *s2){
  const int *ss1 = s1;
  const int *ss2 = s2;

  if (ss1[0] > ss2[0]){
    return 1;
  } else if (ss1[0] < ss2[0]){
    return -1;
  }
  return 0;
}

void vector_ordering(int n, double *v, int **p){
  /* give the position of the smallest, second smallest etc in vector v.
     results in p. If *p == NULL, p is assigned.
  */

  int i;

  if (!*p) *p = gv_calloc(n, sizeof(int));
  double *u = gv_calloc(2 * n, sizeof(double));

  for (i = 0; i < n; i++) {
    u[2*i+1] = i;
    u[2*i] = v[i];
  }

  qsort(u, n, sizeof(double)*2, comp_ascend);

  for (i = 0; i < n; i++) (*p)[i] = (int) u[2*i+1];
  free(u);
}

void vector_sort_int(int n, int *v){
  qsort(v, n, sizeof(int), comp_ascend_int);
}

double distance_cropped(double *x, int dim, int i, int j){
  double dist = distance(x, dim, i, j);
  return fmax(dist, MINDIST);
}

double distance(double *x, int dim, int i, int j){
  int k;
  double dist = 0.;
  for (k = 0; k < dim; k++) dist += (x[i*dim+k] - x[j*dim + k])*(x[i*dim+k] - x[j*dim + k]);
  dist = sqrt(dist);
  return dist;
}

double point_distance(double *p1, double *p2, int dim){
  int i;
  double dist;
  dist = 0;
  for (i = 0; i < dim; i++) dist += (p1[i] - p2[i])*(p1[i] - p2[i]);
  return sqrt(dist);
}

char *strip_dir(char *s){
  bool first = true;
  if (!s) return s;
  for (size_t i = strlen(s); ; i--) {
    if (first && s[i] == '.') {/* get rid of .mtx */
      s[i] = '\0';
      first = false;
    }
    if (s[i] == '/') return &s[i+1];
    if (i == 0) {
      break;
    }
  }
  return s;
}
