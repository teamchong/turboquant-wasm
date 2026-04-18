/**
 * @file
 * @brief @ref snode priority queue for @ref shortPath in @ref sgraph
 */

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

#include <ortho/sgraph.h>
#include <util/alloc.h>

#define N_VAL(n) (n)->n_val
#define N_IDX(n) (n)->n_idx
#define N_DAD(n) (n)->n_dad
#define N_EDGE(n) (n)->n_edge
#define E_WT(e) (e->weight)

/// @return Created priority queue
pq_t *PQgen(int sz);

/// @param pq Priority queue to deallocate
void PQfree(pq_t *pq);

/// @param pq Priority queue to initialize
void PQinit(pq_t *pq);

/// @param pq Priority queue to insert into
int PQ_insert(pq_t *pq, snode *np);

/// @param pq Priority queue to pop
snode *PQremove(pq_t *pq);

/// @param pq Priority queue to update
void PQupdate(pq_t *pq, snode *n, int d);
