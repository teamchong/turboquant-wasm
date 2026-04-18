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

#include <assert.h>
#include <limits.h>
#include <pathplan/pathutil.h>
#include <stdlib.h>
#include <util/list.h>

void freePath(Ppolyline_t *p) {
  free(p->ps);
  free(p);
}

int Ppolybarriers(Ppoly_t **polys, int npolys, Pedge_t **barriers,
                  int *n_barriers) {
  LIST(Pedge_t) bar = {0};

  for (int i = 0; i < npolys; i++) {
    const Ppoly_t pp = *polys[i];
    for (size_t j = 0; j < pp.pn; j++) {
      size_t k = j + 1;
      if (k >= pp.pn)
        k = 0;
      LIST_APPEND(&bar, ((Pedge_t){.a = pp.ps[j], .b = pp.ps[k]}));
    }
  }
  size_t n;
  LIST_DETACH(&bar, barriers, &n);
  assert(n <= INT_MAX);
  *n_barriers = (int)n;
  return 1;
}

void make_polyline(Ppolyline_t line, Ppolyline_t *sline) {
  static LIST(Ppoint_t) ispline;
  LIST_CLEAR(&ispline);

  size_t i = 0;
  LIST_APPEND(&ispline, line.ps[i]);
  LIST_APPEND(&ispline, line.ps[i]);
  i++;
  for (; i + 1 < line.pn; i++) {
    LIST_APPEND(&ispline, line.ps[i]);
    LIST_APPEND(&ispline, line.ps[i]);
    LIST_APPEND(&ispline, line.ps[i]);
  }
  LIST_APPEND(&ispline, line.ps[i]);
  LIST_APPEND(&ispline, line.ps[i]);

  sline->pn = LIST_SIZE(&ispline);
  sline->ps = LIST_FRONT(&ispline);
}

/**
 * @dir lib/pathplan
 * @brief finds and smooths shortest paths, API pathplan.h
 *
 * [man 3 pathplan](https://graphviz.org/pdf/pathplan.3.pdf)
 *
 */
