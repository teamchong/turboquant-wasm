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

#ifdef __cplusplus
extern "C" {
#endif
#include <neatogen/adjust.h>

#ifdef GVDLL
#ifdef NEATOGEN_EXPORTS
#define NEATOPROCS_API __declspec(dllexport)
#else
#define NEATOPROCS_API __declspec(dllimport)
#endif
#endif

#ifndef NEATOPROCS_API
#define NEATOPROCS_API /* nothing */
#endif

    NEATOPROCS_API int checkStart(graph_t * G, int nG, int);
    NEATOPROCS_API int circuit_model(graph_t *, int);
    NEATOPROCS_API void diffeq_model(graph_t *, int);
    NEATOPROCS_API Ppolyline_t getPath(edge_t*, vconfig_t*, bool);
    NEATOPROCS_API void initial_positions(graph_t *, int);
    NEATOPROCS_API void jitter3d(Agnode_t *, int);
    NEATOPROCS_API void jitter_d(Agnode_t *, int, int);
    NEATOPROCS_API Ppoly_t *makeObstacle(node_t * n, expand_t*, bool);
    NEATOPROCS_API void makeSelfArcs(edge_t * e, int stepx);
    NEATOPROCS_API void makeSpline(edge_t *, Ppoly_t **, int, bool);
    NEATOPROCS_API int init_nop(graph_t * g, int);
    NEATOPROCS_API void neato_cleanup(graph_t * g);
    NEATOPROCS_API node_t *neato_dequeue(void);
    NEATOPROCS_API void neato_enqueue(node_t *);
    NEATOPROCS_API void neato_init_node(node_t * n);
    NEATOPROCS_API void neato_layout(Agraph_t * g);
    NEATOPROCS_API int Plegal_arrangement(Ppoly_t ** polys, int n_polys);
    NEATOPROCS_API void randompos(Agnode_t *, int);
    NEATOPROCS_API void s1(graph_t *, node_t *);
    NEATOPROCS_API int scan_graph(graph_t *);
    NEATOPROCS_API int scan_graph_mode(graph_t * G, int mode);
    NEATOPROCS_API void free_scan_graph(graph_t *);
    NEATOPROCS_API int setSeed (graph_t*, int dflt, long* seedp);
    NEATOPROCS_API void shortest_path(graph_t *, int);
    NEATOPROCS_API void solve(double *, double *, double *, size_t);
    NEATOPROCS_API void solve_model(graph_t *, int);
    NEATOPROCS_API int solveCircuit(int nG, double **Gm, double **Gm_inv);
    NEATOPROCS_API void spline_edges(Agraph_t *);
    NEATOPROCS_API void spline_edges0(Agraph_t *, bool);
    NEATOPROCS_API int spline_edges1(graph_t * g, int);
    NEATOPROCS_API int splineEdges(graph_t *,
			   int (*edgefn) (graph_t *, expand_t*, int), int);
    NEATOPROCS_API void neato_translate(Agraph_t * g);
    NEATOPROCS_API bool neato_set_aspect(graph_t * g);
    NEATOPROCS_API void toggle(int);
    NEATOPROCS_API bool user_pos(Agsym_t *, Agsym_t *, Agnode_t *, int);
    NEATOPROCS_API double **new_array(int i, int j, double val);
    NEATOPROCS_API void free_array(double **rv);
    NEATOPROCS_API int matinv(double **A, double **Ainv, int n);

#undef NEATOPROCS_API

#ifdef __cplusplus
}
#endif
