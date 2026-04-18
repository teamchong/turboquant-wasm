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

#include <cgraph.h>
#include <sparse/SparseMatrix.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {COLOR_SCHEME_NONE, COLOR_SCHEME_PASTEL = 1, COLOR_SCHEME_BLUE_YELLOW, COLOR_SCHEME_WHITE_RED, COLOR_SCHEME_GREY_RED, COLOR_SCHEME_PRIMARY, COLOR_SCHEME_SEQUENTIAL_SINGLEHUE_RED, COLOR_SCHEME_ADAM, COLOR_SCHEME_ADAM_BLEND, COLOR_SCHEME_SEQUENTIAL_SINGLEHUE_RED_LIGHTER, COLOR_SCHEME_GREY};
extern void initDotIO (Agraph_t *g);

extern void setDotNodeID (Agnode_t* n, int v);
extern int getDotNodeID (Agnode_t* n);

extern void attach_edge_colors(Agraph_t* g, int dim, double *colors);

extern SparseMatrix SparseMatrix_import_dot(Agraph_t *g, int dim, double **x,
                                            int format);
SparseMatrix Import_coord_clusters_from_dot(Agraph_t* g, int maxcluster, int dim, int *nn, double **label_sizes, double **x, int **clusters, float **rgb_r,  float **rgb_g,  float **rgb_b,  float **fsz, char ***labels, int default_color_scheme, int clustering_scheme, int useClusters);

void Dot_SetClusterColor(Agraph_t* g, float *rgb_r,  float *rgb_g,  float *rgb_b, int *clustering);
void attached_clustering(Agraph_t* g, int maxcluster, int clustering_scheme);

int Import_dot_splines(Agraph_t* g, int *ne, char ***xsplines);

#ifdef __cplusplus
}
#endif
