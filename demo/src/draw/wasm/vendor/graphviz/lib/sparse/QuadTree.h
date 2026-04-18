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

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct node_data_struct *node_data;

struct node_data_struct {
  double node_weight;
  double *coord;
  int id;
  void *data;
  node_data next;
};

typedef struct QuadTree_struct *QuadTree;

struct QuadTree_struct {
  /* a data structure containing coordinates of n items, their average is in "average".
     The current level is a square or cube of width "width", which is subdivided into 
     2^dim QuadTrees qts. At the last level, all coordinates are stored in a single linked list l. 
     total_weight is the combined weights of the nodes */
  int n;/* number of items */
  double total_weight;
  int dim;
  double *center;/* center of the bounding box, array of dimension dim. Allocated inside */
  double width;/* center +/- width gives the lower/upper bound, so really width is the 
		"radius" */
  double *average;/* the average coordinates. Array of length dim. Allocated inside  */
  QuadTree *qts;/* subtree . If dim = 2, there are 4, dim = 3 gives 8 */
  node_data l;
  int max_level;
  void *data;
};


QuadTree QuadTree_new(int dim, double *center, double width, int max_level);

void QuadTree_delete(QuadTree q);

QuadTree QuadTree_add(QuadTree q, double *coord, double weight, int id);/* coord is copied in */

void QuadTree_print(FILE *fp, QuadTree q);

QuadTree QuadTree_new_from_point_list(int dim, int n, int max_level, double *coord);

double point_distance(double *p1, double *p2, int dim);

void QuadTree_get_supernodes(QuadTree qt, double bh, double *pt, int nodeid, int *nsuper, 
			     int *nsupermax, double **center, double **supernode_wgts, double **distances, double *counts);

void QuadTree_get_repulsive_force(QuadTree qt, double *force, double *x, double bh, double p, double KP, double *counts);

/* find the nearest point and put in ymin, index in imin and distance in min */
void QuadTree_get_nearest(QuadTree qt, double *x, double *ymin, int *imin, double *min);

QuadTree QuadTree_new_in_quadrant(int dim, double *center, double width, int max_level, int i);

#ifdef __cplusplus
}
#endif
