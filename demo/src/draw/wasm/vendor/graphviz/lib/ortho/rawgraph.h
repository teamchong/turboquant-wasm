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

#include <stdbool.h>
#include <stddef.h>
#include <util/list.h>

typedef LIST(size_t) adj_list_t;

typedef struct {
  int color;
  int topsort_order;
  adj_list_t adj_list; ///< adjacency list
} vertex;

typedef struct {
  size_t nvs;
  vertex* vertices;
} rawgraph;

/// makes a graph with n vertices, 0 edges
rawgraph *make_graph(size_t n);

extern void free_graph(rawgraph*); 

/// inserts edge FROM v1 to v2
void insert_edge(rawgraph *, size_t v1, size_t v2);

/// removes any edge between v1 to v2 -- irrespective of direction
void remove_redge(rawgraph *, size_t v1, size_t v2);

/// tests if there is an edge FROM v1 TO v2
bool edge_exists(rawgraph *, size_t v1, size_t v2);

  /* topologically sorts the directed graph */
extern void top_sort(rawgraph*); 
