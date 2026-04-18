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

#include "types.h"
#include "cgraph.h"
#include <ortho/rawgraph.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/list.h>

typedef struct {
    double p1, p2;
} paird;

typedef struct {
    int a,b;
} pair;

typedef struct {
	pair t1, t2;
} pair2;

typedef enum {B_NODE, B_UP, B_LEFT, B_DOWN, B_RIGHT} bend;

/* Example : segment connecting maze point (3,2) 
 * and (3,8) has isVert = 1, common coordinate = 3, p1 = 2, p2 = 8
 */
typedef struct segment {
  bool isVert;
  double comm_coord;  /* the common coordinate */
  paird p;      /* end points */
  bend l1, l2; 
  size_t ind_no; ///< index number of this segment in its channel
  int track_no;    /* track number assigned in the channel */
  struct segment* prev;
  struct segment* next;
} segment;

typedef struct {
  size_t n;
  segment* segs;
} route;

typedef LIST(segment *) seg_list_t;

typedef struct {
  Dtlink_t link;
  paird p;   /* extrema of channel */
  seg_list_t seg_list; ///< segment pointers
  rawgraph* G;
  struct cell* cp;
} channel;

#define N_DAD(n) (n)->n_dad
