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

#include <neatogen/sparsegraph.h>

typedef struct {
    int  nedges; /* no. of edges in triangulation */
    int* edges;  /* 2*nsegs indices of points */
    int  nfaces; /* no. of faces in triangulation */
    int* faces;  /* 3*nfaces indices of points */ 
    int* neigh;  /* 3*nfaces indices of neighbor triangles */ 
} surface_t;

int *delaunay_tri (double *x, double *y, int n, int* nedges);

int *get_triangles (double *x, int n, int* ntris);

v_data *UG_graph(double *x, double *y, int n);

surface_t* mkSurface (double *x, double *y, int n, int* segs, int nsegs);

void freeSurface (surface_t* s);
