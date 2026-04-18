/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/* Priority Queue Code for shortest path in graph */

#include "config.h"
#include <assert.h>
#include <util/alloc.h>

#include <ortho/fPQ.h>

struct pq {
  snode **pq;
  int cnt;
  snode guard;
  int size;
};

pq_t *PQgen(int sz) {
  pq_t *const pq = gv_alloc(sizeof(*pq));
  pq->pq = gv_calloc(sz + 1, sizeof(snode *));
  pq->pq[0] = &pq->guard;
  pq->size = sz;
  pq->cnt = 0;
  return pq;
}

void PQfree(pq_t *pq) {
  if (pq != NULL) {
    free(pq->pq);
  }
  free(pq);
}

void PQinit(pq_t *pq) { pq->cnt = 0; }

static void PQcheck(const pq_t *pq) {
  for (int i = 1; i <= pq->cnt; i++) {
    assert(N_IDX(pq->pq[i]) == i);
  }
}

static void PQupheap(pq_t *pq, int k) {
  snode *x = pq->pq[k];
  int v = x->n_val;
  int next = k / 2;
  snode *n;

  while (N_VAL(n = pq->pq[next]) < v) {
    pq->pq[k] = n;
    N_IDX(n) = k;
    k = next;
    next /= 2;
  }
  pq->pq[k] = x;
  N_IDX(x) = k;
}

int PQ_insert(pq_t *pq, snode *np) {
  if (pq->cnt == pq->size) {
    agerrorf("Heap overflow\n");
    return 1;
  }
  pq->cnt++;
  pq->pq[pq->cnt] = np;
  PQupheap(pq, pq->cnt);
  PQcheck(pq);
  return 0;
}

static void PQdownheap(pq_t *pq, int k) {
  snode *x = pq->pq[k];
  int v = N_VAL(x);
  int lim = pq->cnt / 2;

  while (k <= lim) {
    int j = k + k;
    snode *n = pq->pq[j];
    if (j < pq->cnt) {
      if (N_VAL(n) < N_VAL(pq->pq[j + 1])) {
        j++;
        n = pq->pq[j];
      }
    }
    if (v >= N_VAL(n))
      break;
    pq->pq[k] = n;
    N_IDX(n) = k;
    k = j;
  }
  pq->pq[k] = x;
  N_IDX(x) = k;
}

snode *PQremove(pq_t *pq) {
  if (pq->cnt) {
    snode *const n = pq->pq[1];
    pq->pq[1] = pq->pq[pq->cnt];
    pq->cnt--;
    if (pq->cnt)
      PQdownheap(pq, 1);
    PQcheck(pq);
    return n;
  }
  return 0;
}

void PQupdate(pq_t *pq, snode *n, int d) {
  N_VAL(n) = d;
  PQupheap(pq, n->n_idx);
  PQcheck(pq);
}
