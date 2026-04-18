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

enum {ERROR_NOT_SQUARE_MATRIX = -100};

/* a flag to indicate that p should be set auto */
#define AUTOP -1.0001234

enum {SMOOTHING_NONE, SMOOTHING_STRESS_MAJORIZATION_GRAPH_DIST, SMOOTHING_STRESS_MAJORIZATION_AVG_DIST, SMOOTHING_STRESS_MAJORIZATION_POWER_DIST, SMOOTHING_SPRING, SMOOTHING_TRIANGLE, SMOOTHING_RNG};

enum {QUAD_TREE_HYBRID_SIZE = 10000};

enum {QUAD_TREE_NONE = 0, QUAD_TREE_NORMAL, QUAD_TREE_FAST, QUAD_TREE_HYBRID};

struct spring_electrical_control_struct {
  double p;/*a negativve real number default to -1. repulsive force = dist^p */
  double K;/* the natural distance. If K < 0, K will be set to the average distance of an edge */
  int multilevels;/* if <=1, single level */
  int max_qtree_level;/* max level of quadtree */
  int maxiter;
  double step;/* initial step size */
  int random_seed;
  bool random_start : 1; ///< whether to apply SE from a random layout, or from exisiting layout
  bool adaptive_cooling : 1;
  bool beautify_leaves : 1;
  int smoothing;
  int overlap;
  bool do_shrinking;
  int tscheme; /* octree scheme. 0 (no octree), 1 (normal), 2 (fast) */
  double initial_scaling;/* how to scale the layout of the graph before passing to overlap removal algorithm.
			  positive values are absolute in points, negative values are relative
			  to average label size.
			  */
  double rotation;/* degree of rotation */
  int edge_labeling_scheme; /* specifying whether to treat node of the form |edgelabel|* as a special node representing an edge label. 
			       0 (no action, default), 1 (penalty based method to make that kind of node close to the center of its neighbor), 
			       1 (penalty based method to make that kind of node close to the old center of its neighbor),
			       3 (two step process of overlap removal and straightening) */
};

typedef struct  spring_electrical_control_struct  *spring_electrical_control; 

spring_electrical_control spring_electrical_control_new(void);
void spring_electrical_control_print(spring_electrical_control ctrl);

void spring_electrical_embedding(int dim, SparseMatrix A0, spring_electrical_control ctrl, double *x, int *flag);
void spring_electrical_embedding_fast(int dim, SparseMatrix A0, spring_electrical_control ctrl, double *x, int *flag);

void multilevel_spring_electrical_embedding(int dim, SparseMatrix A0,
                                            spring_electrical_control ctrl,
                                            double *label_sizes, double *x,
                                            int n_edge_label_nodes,
                                            int *edge_label_nodes, int *flag);

void spring_electrical_control_delete(spring_electrical_control ctrl);

double average_edge_length(SparseMatrix A, int dim, double *coord);

void spring_electrical_spring_embedding(int dim, SparseMatrix A, SparseMatrix D, spring_electrical_control ctrl, double *x, int *flag);

void pcp_rotate(int n, int dim, double *x);
