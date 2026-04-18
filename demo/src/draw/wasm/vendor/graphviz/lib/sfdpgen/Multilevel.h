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

#include <sparse/SparseMatrix.h>
#include <stdbool.h>

typedef struct Multilevel_struct *Multilevel;

struct Multilevel_struct {
  int level; /* 0, 1, ... */
  int n;
  SparseMatrix A; /* the weighting matrix */
  SparseMatrix P;
  SparseMatrix R;
  Multilevel next;
  Multilevel prev;
  bool delete_top_level_A;
};

enum { MAX_CLUSTER_SIZE = 4 };

typedef struct {
  int maxlevel;
} Multilevel_control;

void Multilevel_delete(Multilevel grid);

Multilevel Multilevel_new(SparseMatrix A, const Multilevel_control ctrl);

Multilevel Multilevel_get_coarsest(Multilevel grid);

void print_padding(int n);

#define Multilevel_is_finest(grid) (!((grid)->prev))
#define Multilevel_is_coarsest(grid) (!((grid)->next))
