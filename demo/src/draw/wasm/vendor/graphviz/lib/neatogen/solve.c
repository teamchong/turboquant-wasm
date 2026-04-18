/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/* solves the system ab=c using gauss reduction */
#include <assert.h>
#include <common/render.h>
#include <math.h>
#include <neatogen/neatoprocs.h>
#include <stdio.h>
#include <stdlib.h>
#include <util/alloc.h>

static void swap(double *x, double *y) {
  const double temp = *x;
  *x = *y;
  *y = temp;
}

void solve(double *a, double *b, double *c, size_t n) { // a[n][n],b[n],c[n]

  assert(n >= 2);

  const size_t nsq = n * n;
  double *asave = gv_calloc(nsq, sizeof(double));
  double *csave = gv_calloc(n, sizeof(double));

  for (size_t i = 0; i < n; i++)
    csave[i] = c[i];
  for (size_t i = 0; i < nsq; i++)
    asave[i] = a[i];
  /* eliminate ith unknown */
  const size_t nm = n - 1;
  for (size_t i = 0; i < nm; i++) {
    /* find largest pivot */
    double amax = 0.;
    size_t istar = 0;
    for (size_t ii = i; ii < n; ii++) {
      const double dum = fabs(a[ii * n + i]);
      if (dum < amax)
        continue;
      istar = ii;
      amax = dum;
    }
    /* return if pivot is too small */
    if (amax < 1.e-10)
      goto bad;
    /* switch rows */
    for (size_t j = i; j < n; j++) {
      const size_t t = istar * n + j;
      swap(&a[t], &a[i * n + j]);
    }
    swap(&c[istar], &c[i]);
    /*pivot */
    const size_t ip = i + 1;
    for (size_t ii = ip; ii < n; ii++) {
      const double pivot = a[ii * n + i] / a[i * n + i];
      c[ii] -= pivot * c[i];
      for (size_t j = 0; j < n; j++)
        a[ii * n + j] = a[ii * n + j] - pivot * a[i * n + j];
    }
  }
  /* return if last pivot is too small */
  if (fabs(a[n * n - 1]) < 1.e-10)
    goto bad;
  b[n - 1] = c[n - 1] / a[n * n - 1];
  /* back substitute */
  for (size_t k = 0; k < nm; k++) {
    const size_t m = n - k - 2;
    b[m] = c[m];
    const size_t mp = m + 1;
    for (size_t j = mp; j < n; j++)
      b[m] -= a[m * n + j] * b[j];
    b[m] /= a[m * n + m];
  }
  /* restore original a,c */
  for (size_t i = 0; i < n; i++)
    c[i] = csave[i];
  for (size_t i = 0; i < nsq; i++)
    a[i] = asave[i];
  free(asave);
  free(csave);
  return;
bad:
  printf("ill-conditioned\n");
  free(asave);
  free(csave);
  return;
}
