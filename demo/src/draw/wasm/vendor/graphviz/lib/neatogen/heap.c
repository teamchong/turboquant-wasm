/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <common/render.h>
#include <stdbool.h>

#include <neatogen/mem.h>
#include <neatogen/hedges.h>
#include <neatogen/heap.h>
#include <util/alloc.h>

struct pq {
  Halfedge *hash; ///< backing storage
  int hashsize;   ///< total allocated backing storage elements
  int count;      ///< occupancy
  int min;        ///< index of minimum element
};

static int PQbucket(pq_t *pq, Halfedge *he) {
    int bucket;
    double b;

    const double deltay = ymax - ymin;
    b = (he->ystar - ymin) / deltay * pq->hashsize;
    if (b < 0)
	bucket = 0;
    else if (b >= pq->hashsize)
	bucket = pq->hashsize - 1;
    else
	bucket = b;
    if (bucket < pq->min)
	pq->min = bucket;
    return bucket;
}

/// `a > b`?
static bool gt(double a_y, double a_x, double b_y, double b_x) {
  if (a_y > b_y) {
    return true;
  }
  if (a_y < b_y) {
    return false;
  }
  return a_x > b_x;
}

void PQinsert(pq_t *pq, Halfedge *he, Site *v, double offset) {
    Halfedge *last, *next;

    he->vertex = v;
    ref(v);
    he->ystar = v->coord.y + offset;
    last = &pq->hash[PQbucket(pq, he)];
    while ((next = last->PQnext) != NULL &&
           gt(he->ystar, v->coord.x, next->ystar, next->vertex->coord.x)) {
	last = next;
    }
    he->PQnext = last->PQnext;
    last->PQnext = he;
    ++pq->count;
}

void PQdelete(pq_t *pq, Halfedge *he) {
    Halfedge *last;

    if (he->vertex != NULL) {
	last = &pq->hash[PQbucket(pq, he)];
	while (last->PQnext != he)
	    last = last->PQnext;
	last->PQnext = he->PQnext;
	--pq->count;
	deref(he->vertex);
	he->vertex = NULL;
    }
}

bool PQempty(const pq_t *pq) {
  return pq->count == 0;
}

Point PQ_min(pq_t *pq) {
    Point answer;

    while (pq->hash[pq->min].PQnext == NULL) {
      ++pq->min;
    }
    answer.x = pq->hash[pq->min].PQnext->vertex->coord.x;
    answer.y = pq->hash[pq->min].PQnext->ystar;
    return answer;
}

Halfedge *PQextractmin(pq_t *pq) {
    Halfedge *curr = pq->hash[pq->min].PQnext;
    pq->hash[pq->min].PQnext = curr->PQnext;
    --pq->count;
    return curr;
}

void PQcleanup(pq_t *pq) {
  if (pq != NULL) {
    free(pq->hash);
  }
  free(pq);
}

pq_t *PQinitialize(void) {
  pq_t *pq = gv_alloc(sizeof(pq_t));
  pq->hashsize = 4 * sqrt_nsites;
  pq->hash = gv_calloc(pq->hashsize, sizeof(Halfedge));
  return pq;
}
