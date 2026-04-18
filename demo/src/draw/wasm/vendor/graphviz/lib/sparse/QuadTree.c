/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <sparse/general.h>
#include <common/geom.h>
#include <common/arith.h>
#include <math.h>
#include <sparse/QuadTree.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/alloc.h>

extern double distance_cropped(double *x, int dim, int i, int j);

static node_data node_data_new(int dim, double weight, double *coord, int id){
  int i;
  node_data nd = gv_alloc(sizeof(struct node_data_struct));
  nd->node_weight = weight;
  nd->coord = gv_calloc(dim, sizeof(double));
  nd->id = id;
  for (i = 0; i < dim; i++) nd->coord[i] = coord[i];
  nd->data = NULL;
  return nd;
}

static void node_data_delete(node_data nd){
  free(nd->coord);
  /*delete outside   if (nd->data) free(nd->data);*/
  free(nd);
}

static void check_or_realloc_arrays(int dim, int *nsuper, int *nsupermax, double **center, double **supernode_wgts, double **distances){
  
  if (*nsuper >= *nsupermax) {
    int new_nsupermax = *nsuper + 10;
    *center = gv_recalloc(*center, dim * *nsupermax, dim * new_nsupermax, sizeof(double));
    *supernode_wgts = gv_recalloc(*supernode_wgts, *nsupermax, new_nsupermax, sizeof(double));
    *distances = gv_recalloc(*distances, *nsupermax, new_nsupermax, sizeof(double));
    *nsupermax = new_nsupermax;
  }
}

static void QuadTree_get_supernodes_internal(QuadTree qt, double bh, double *pt, int nodeid, int *nsuper, int *nsupermax, double **center, double **supernode_wgts, double **distances, double *counts) {
  double *coord, dist;
  int dim, i;

  (*counts)++;

  if (!qt) return;
  dim = qt->dim;
  node_data l = qt->l;
  while (l) {
    check_or_realloc_arrays(dim, nsuper, nsupermax, center, supernode_wgts, distances);
    if (l->id != nodeid){
      coord = l->coord;
      for (i = 0; i < dim; i++){
        (*center)[dim*(*nsuper)+i] = coord[i];
      }
      (*supernode_wgts)[*nsuper] = l->node_weight;
      (*distances)[*nsuper] = point_distance(pt, coord, dim);
      (*nsuper)++;
    }
    l = l->next;
  }

  if (qt->qts){
    dist = point_distance(qt->center, pt, dim); 
    if (qt->width < bh*dist){
      check_or_realloc_arrays(dim, nsuper, nsupermax, center, supernode_wgts, distances);
      for (i = 0; i < dim; i++){
        (*center)[dim*(*nsuper)+i] = qt->average[i];
      }
      (*supernode_wgts)[*nsuper] = qt->total_weight;
      (*distances)[*nsuper] = point_distance(qt->average, pt, dim); 
      (*nsuper)++;
    } else {
      for (i = 0; i < 1<<dim; i++){
	QuadTree_get_supernodes_internal(qt->qts[i], bh, pt, nodeid, nsuper, nsupermax, center, 
					 supernode_wgts, distances, counts);
      }
    }
  }

}

void QuadTree_get_supernodes(QuadTree qt, double bh, double *pt, int nodeid, int *nsuper, 
			     int *nsupermax, double **center, double **supernode_wgts, double **distances, double *counts) {
  int dim = qt->dim;

  (*counts) = 0;

  *nsuper = 0;

  *nsupermax = 10;
  if (!*center) *center = gv_calloc(*nsupermax * dim, sizeof(double));
  if (!*supernode_wgts) *supernode_wgts = gv_calloc(*nsupermax, sizeof(double));
  if (!*distances) *distances = gv_calloc(*nsupermax, sizeof(double));
  QuadTree_get_supernodes_internal(qt, bh, pt, nodeid, nsuper, nsupermax, center, supernode_wgts, distances, counts);

}

static double *get_or_assign_node_force(double *force, int i, node_data l,
                                        int dim) {

  double *f = l->data;

  if (!f){
    l->data = &(force[i*dim]);
    f = l->data;
  }
  return f;
}
static double *get_or_alloc_force_qt(QuadTree qt, int dim){
  double *force = qt->data;
  if (!force){
    qt->data = gv_calloc(dim, sizeof(double));
    force = qt->data;
  }
  return force;
}

static void QuadTree_repulsive_force_interact(QuadTree qt1, QuadTree qt2, double *x, double *force, double bh, double p, double KP, double *counts){
  // calculate the all to all repulsive force and accumulate on each node of the
  // quadtree if an interaction is possible.
  //   force[i × dim + j], j=1,..., dim is the force on node i
  double *x1, *x2, dist, wgt1, wgt2, f, *f1, *f2, w1, w2;
  int dim, i, j, i1, i2, k;
  QuadTree qt11, qt12; 

  if (!qt1 || !qt2) return;
  assert(qt1->n > 0 && qt2->n > 0);
  dim = qt1->dim;

  node_data l1 = qt1->l;
  node_data l2 = qt2->l;

  /* far enough, calculate repulsive force */
  dist = point_distance(qt1->average, qt2->average, dim); 
  if (qt1->width + qt2->width < bh*dist){
    counts[0]++;
    x1 = qt1->average;
    w1 = qt1->total_weight;
    f1 = get_or_alloc_force_qt(qt1, dim);
    x2 = qt2->average;
    w2 = qt2->total_weight;
    f2 = get_or_alloc_force_qt(qt2, dim);
    assert(dist > 0);
    for (k = 0; k < dim; k++){
      if (p == -1){
	f = w1*w2*KP*(x1[k] - x2[k])/(dist*dist);
      } else {
	f = w1*w2*KP*(x1[k] - x2[k])/pow(dist, 1.- p);
      }
      f1[k] += f;
      f2[k] -= f;
    }
    return;
  }


  /* both at leaves, calculate repulsive force */
  if (l1 && l2){
    while (l1){
      x1 = l1->coord;
      wgt1 = l1->node_weight;
      i1 = l1->id;
      f1 = get_or_assign_node_force(force, i1, l1, dim);
      l2 = qt2->l;
      while (l2){
	x2 = l2->coord;
	wgt2 = l2->node_weight;
	i2 = l2->id;
	f2 = get_or_assign_node_force(force, i2, l2, dim);
	if ((qt1 == qt2 && i2 < i1) || i1 == i2) {
	  l2 = l2->next;
	  continue;
	}
	counts[1]++;
	dist = distance_cropped(x, dim, i1, i2);
	for (k = 0; k < dim; k++){
	  if (p == -1){
	    f = wgt1*wgt2*KP*(x1[k] - x2[k])/(dist*dist);
	  } else {
	    f = wgt1*wgt2*KP*(x1[k] - x2[k])/pow(dist, 1.- p);
	  }
	  f1[k] += f;
	  f2[k] -= f;
	}
	l2 = l2->next;
      }
      l1 = l1->next;
    }
    return;
  }


  /* identical, split one */
  if (qt1 == qt2){
      for (i = 0; i < 1<<dim; i++){
	qt11 = qt1->qts[i];
	for (j = i; j < 1<<dim; j++){
	  qt12 = qt1->qts[j];
	  QuadTree_repulsive_force_interact(qt11, qt12, x, force, bh, p, KP, counts);
	}
      }
  } else {
    /* split the one with bigger box, or one not at the last level */
    if (qt1->width > qt2->width && !l1){
      for (i = 0; i < 1<<dim; i++){
	qt11 = qt1->qts[i];
	QuadTree_repulsive_force_interact(qt11, qt2, x, force, bh, p, KP, counts);
      }
    } else if (qt2->width > qt1->width && !l2){
      for (i = 0; i < 1<<dim; i++){
	qt11 = qt2->qts[i];
	QuadTree_repulsive_force_interact(qt11, qt1, x, force, bh, p, KP, counts);
      }
    } else if (!l1){/* pick one that is not at the last level */
      for (i = 0; i < 1<<dim; i++){
	qt11 = qt1->qts[i];
	QuadTree_repulsive_force_interact(qt11, qt2, x, force, bh, p, KP, counts);
      }
    } else if (!l2){
      for (i = 0; i < 1<<dim; i++){
	qt11 = qt2->qts[i];
	QuadTree_repulsive_force_interact(qt11, qt1, x, force, bh, p, KP, counts);
      }
    } else {
      assert(0); // can be both at the leaf level since that should be caught at
                 // the beginning of this func
    }
  }
}

static void QuadTree_repulsive_force_accumulate(QuadTree qt, double *force, double *counts){
  /* push down forces on cells into the node level */
  double wgt, wgt2;
  double *f, *f2;
  node_data l = qt->l;
  int i, k, dim;
  QuadTree qt2;

  dim = qt->dim;
  wgt = qt->total_weight;
  f = get_or_alloc_force_qt(qt, dim);
  assert(wgt > 0);
  counts[2]++;

  if (l){
    while (l){
      i = l->id;
      f2 = get_or_assign_node_force(force, i, l, dim);
      wgt2 = l->node_weight;
      wgt2 = wgt2/wgt;
      for (k = 0; k < dim; k++) f2[k] += wgt2*f[k];
      l = l->next;
    }
    return;
  }

  for (i = 0; i < 1<<dim; i++){
    qt2 = qt->qts[i];
    if (!qt2) continue;
    assert(qt2->n > 0);
    f2 = get_or_alloc_force_qt(qt2, dim);
    wgt2 = qt2->total_weight;
    wgt2 = wgt2/wgt;
    for (k = 0; k < dim; k++) f2[k] += wgt2*f[k];
    QuadTree_repulsive_force_accumulate(qt2, force, counts);
  }

}

void QuadTree_get_repulsive_force(QuadTree qt, double *force, double *x, double bh, double p, double KP, double *counts){
  // get repulsive force by a more efficient algorithm: we consider two cells,
  // if they are well separated, we calculate the overall repulsive force on the
  // cell level, if not well separated, we divide one of the cell. If both cells
  // are at the leaf level, we calculate repulsive force among individual nodes.
  // Finally we accumulate forces at the cell levels to the node level
  //   qt: the quadtree
  //   x: current coordinates for node i is x[i*dim+j], j = 0, ..., dim-1
  //   force: the repulsive force, an array of length dim*nnodes, the force for node i is at force[i*dim+j], j = 0, ..., dim - 1
  //   bh: Barnes-Hut coefficient. If width_cell1+width_cell2 < bh*dist_between_cells, we treat each cell as a supernode.
  //   p: the repulsive force power
  //   KP: pow(K, 1 - p)
  //   counts: array of size 4. 
  //   .  counts[0]: number of cell-cell interaction
  //   .  counts[1]: number of cell-node interaction
  //   .  counts[2]: number of total cells in the quadtree
  //   . Al normalized by dividing by number of nodes
  int n = qt->n, dim = qt->dim, i;

  for (i = 0; i < 4; i++) counts[i] = 0;

  for (i = 0; i < dim*n; i++) force[i] = 0;

  QuadTree_repulsive_force_interact(qt, qt, x, force, bh, p, KP, counts);
  QuadTree_repulsive_force_accumulate(qt, force, counts);
  for (i = 0; i < 4; i++) counts[i] /= n;

}
QuadTree QuadTree_new_from_point_list(int dim, int n, int max_level, double *coord){
  /* form a new QuadTree data structure from a list of coordinates of n points
     coord: of length n*dim, point i sits at [i*dim, i*dim+dim - 1]
   */
  double *xmin, *xmax, *center, width;
  QuadTree qt = NULL;
  int i, k;

  xmin = gv_calloc(dim, sizeof(double));
  xmax = gv_calloc(dim, sizeof(double));
  center = gv_calloc(dim, sizeof(double));
  if (!xmin || !xmax || !center) {
      free(xmin);
      free(xmax);
      free(center);
      return NULL;
  }

  for (i = 0; i < dim; i++) xmin[i] = coord[i];
  for (i = 0; i < dim; i++) xmax[i] = coord[i];

  for (i = 1; i < n; i++){
    for (k = 0; k < dim; k++){
      xmin[k] = fmin(xmin[k], coord[i*dim+k]);
      xmax[k] = fmax(xmax[k], coord[i*dim+k]);
    }
  }

  width = xmax[0] - xmin[0];
  for (i = 0; i < dim; i++) {
    center[i] = (xmin[i] + xmax[i])*0.5;
    width = fmax(width, xmax[i] - xmin[i]);
  }
  width = fmax(width, 0.00001);/* if we only have one point, width = 0! */
  width *= 0.52;
  qt = QuadTree_new(dim, center, width, max_level);

  for (i = 0; i < n; i++){
    qt = QuadTree_add(qt, &(coord[i*dim]), 1, i);
  }


  free(xmin);
  free(xmax);
  free(center);
  return qt;
}

QuadTree QuadTree_new(int dim, double *center, double width, int max_level){
  int i;
  QuadTree q = gv_alloc(sizeof(struct QuadTree_struct));
  q->dim = dim;
  q->n = 0;
  q->center = gv_calloc(dim, sizeof(double));
  for (i = 0; i < dim; i++) q->center[i] = center[i];
  assert(width > 0);
  q->width = width;
  q->total_weight = 0;
  q->average = NULL;
  q->qts = NULL;
  q->l = NULL;
  q->max_level = max_level;
  q->data = NULL;
  return q;
}

void QuadTree_delete(QuadTree q){
  int i, dim;
  if (!q) return;
  dim = q->dim;
  free(q->center);
  free(q->average);
  free(q->data);
  if (q->qts){
    for (i = 0; i < 1<<dim; i++){
      QuadTree_delete(q->qts[i]);
    }
    free(q->qts);
  }
  while (q->l) {
    node_data next = q->l->next;
    node_data_delete(q->l);
    q->l = next;
  }
  free(q);
}

static int QuadTree_get_quadrant(int dim, double *center, double *coord){
  /* find the quadrant that a point of coordinates coord is going into with reference to the center.
     if coord - center == {+,-,+,+} = {1,0,1,1}, then it will sit in the i-quadrant where
     i's binary representation is 1011 (that is, decimal 11).
   */
  int d = 0, i;

  for (i = dim - 1; i >= 0; i--){
    if (coord[i] - center[i] < 0){
      d = 2*d;
    } else {
      d = 2*d+1;
    }
  }
  return d;
}

QuadTree QuadTree_new_in_quadrant(int dim, double *center, double width, int max_level, int i){
  /* a new quadtree in quadrant i of the original cell. The original cell is centered at 'center".
     The new cell have width "width".
   */
  QuadTree qt;
  int k;

  qt = QuadTree_new(dim, center, width, max_level);
  center = qt->center;/* right now this has the center for the parent */
  for (k = 0; k < dim; k++){/* decompose child id into binary, if {1,0}, say, then
				     add {width/2, -width/2} to the parents' center
				     to get the child's center. */
    if (i%2 == 0){
      center[k]  -= width;
    } else {
      center[k] += width;
    }
    i = (i - i%2)/2;
  }
  return qt;

}
static QuadTree QuadTree_add_internal(QuadTree q, double *coord, double weight, int id, int level){
  int i, dim = q->dim, ii;
  node_data nd = NULL;

  int max_level = q->max_level;
  int idd;

  /* Make sure that coord is within bounding box */
  for (i = 0; i < q->dim; i++) {
    if (coord[i] < q->center[i] - q->width - 1.e5*MACHINEACC*q->width || coord[i] > q->center[i] + q->width + 1.e5*MACHINEACC*q->width) {
#ifdef DEBUG_PRINT
      fprintf(stderr,"coordinate %f is outside of the box:{%f, %f}, \n(q->center[i] - q->width) - coord[i] =%g, coord[i]-(q->center[i] + q->width) = %g\n",coord[i], (q->center[i] - q->width), (q->center[i] + q->width),
	      (q->center[i] - q->width) - coord[i],  coord[i]-(q->center[i] + q->width));
#endif
      //return NULL;
    }
  }

  if (q->n == 0){
    /* if this node is empty right now */
    q->n = 1;
    q->total_weight = weight;
    q->average = gv_calloc(dim, sizeof(double));
    for (i = 0; i < q->dim; i++) q->average[i] = coord[i];
    nd = node_data_new(q->dim, weight, coord, id);
    assert(!(q->l));
    q->l = nd;
  } else if (level < max_level){
    /* otherwise open up into 2^dim quadtrees unless the level is too high */
    q->total_weight += weight;
    for (i = 0; i < q->dim; i++) q->average[i] = (q->average[i] * q->n + coord[i]) / (q->n + 1);
    if (!q->qts){
      q->qts = gv_calloc((size_t)1 << dim, sizeof(QuadTree));
    }/* done adding new quadtree, now add points to them */
    
    /* insert the old node (if exist) and the current node into the appropriate child quadtree */
    ii = QuadTree_get_quadrant(dim, q->center, coord);
    assert(ii < 1<<dim && ii >= 0);
    if (q->qts[ii] == NULL) q->qts[ii] = QuadTree_new_in_quadrant(q->dim, q->center, q->width / 2, max_level, ii);
    
    q->qts[ii] = QuadTree_add_internal(q->qts[ii], coord, weight, id, level + 1);
    assert(q->qts[ii]);

    if (q->l){
      idd = q->l->id;
      assert(q->n == 1);
      coord = q->l->coord;
      weight = q->l->node_weight;
      ii = QuadTree_get_quadrant(dim, q->center, coord);
      assert(ii < 1<<dim && ii >= 0);

      if (q->qts[ii] == NULL) q->qts[ii] = QuadTree_new_in_quadrant(q->dim, q->center, (q->width)/2, max_level, ii);

      q->qts[ii] = QuadTree_add_internal(q->qts[ii], coord, weight, idd, level + 1);
      assert(q->qts[ii]);
      
      /* delete the old node data on parent */
      while (q->l != NULL) {
        node_data next = q->l->next;
        node_data_delete(q->l);
        q->l = next;
      }
    }
    
    q->n++;
  } else {
    assert(!(q->qts));
    /* level is too high, append data in the linked list */
    q->n++;
    q->total_weight += weight;
    for (i = 0; i < q->dim; i++) q->average[i] = (q->average[i] * q->n + coord[i]) / (q->n + 1);
    nd = node_data_new(q->dim, weight, coord, id);
    assert(q->l);
    nd->next = q->l;
    q->l = nd;
  }
  return q;
}


QuadTree QuadTree_add(QuadTree q, double *coord, double weight, int id){
  if (!q) return q;
  return QuadTree_add_internal(q, coord, weight, id, 0);
  
}

static void draw_polygon(FILE *fp, int dim, double *center, double width){
  // plot the enclosing square
  if (dim < 2 || dim > 3) return;
  fprintf(fp,"(*in c*){Line[{");

  if (dim == 2){
    fprintf(fp, "{%f, %f}", center[0] + width, center[1] + width);
    fprintf(fp, ",{%f, %f}", center[0] - width, center[1] + width);
    fprintf(fp, ",{%f, %f}", center[0] - width, center[1] - width);
    fprintf(fp, ",{%f, %f}", center[0] + width, center[1] - width);
    fprintf(fp, ",{%f, %f}", center[0] + width, center[1] + width);
  } else if (dim == 3){
    /* top */
    fprintf(fp,"{");
    fprintf(fp, "{%f, %f, %f}", center[0] + width, center[1] + width, center[2] + width);
    fprintf(fp, ",{%f, %f, %f}", center[0] - width, center[1] + width, center[2] + width);
    fprintf(fp, ",{%f, %f, %f}", center[0] - width, center[1] - width, center[2] + width);
    fprintf(fp, ",{%f, %f, %f}", center[0] + width, center[1] - width, center[2] + width);
    fprintf(fp, ",{%f, %f, %f}", center[0] + width, center[1] + width, center[2] + width);
    fprintf(fp,"},");
    /* bot */
    fprintf(fp,"{");
    fprintf(fp, "{%f, %f, %f}", center[0] + width, center[1] + width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] - width, center[1] + width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] - width, center[1] - width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] + width, center[1] - width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] + width, center[1] + width, center[2] - width);
    fprintf(fp,"},");
    /* for sides */
    fprintf(fp,"{");
    fprintf(fp, "{%f, %f, %f}", center[0] + width, center[1] + width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] + width, center[1] + width, center[2] + width);
    fprintf(fp,"},");
    
    fprintf(fp,"{");
    fprintf(fp, "{%f, %f, %f}", center[0] - width, center[1] + width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] - width, center[1] + width, center[2] + width);
    fprintf(fp,"},");
    
    fprintf(fp,"{");
    fprintf(fp, "{%f, %f, %f}", center[0] + width, center[1] - width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] + width, center[1] - width, center[2] + width);
    fprintf(fp,"},");
    
    fprintf(fp,"{");
    fprintf(fp, "{%f, %f, %f}", center[0] - width, center[1] - width, center[2] - width);
    fprintf(fp, ",{%f, %f, %f}", center[0] - width, center[1] - width, center[2] + width);
    fprintf(fp,"}");
  }


  fprintf(fp, "}]}(*end C*)");


}
static void QuadTree_print_internal(FILE *fp, QuadTree q, int level){
  /* dump a quad tree in Mathematica format. */
  node_data l, l0;
  double *coord;
  int i, dim;

  if (!q) return;

  draw_polygon(fp, q->dim, q->center, q->width);
  dim = q->dim;
  
  l0 = l = q->l;
  if (l){
    printf(",(*a*) {Red,");
    while (l){
      if (l != l0) printf(",");
      coord = l->coord;
      fprintf(fp, "(*node %d*) Point[{", l->id);
      for (i = 0; i < dim; i++){
	if (i != 0) printf(",");
	fprintf(fp, "%f",coord[i]);
      }
      fprintf(fp, "}]");
      l = l->next;
    }
    fprintf(fp, "}");
  }

  if (q->qts){
    for (i = 0; i < 1<<dim; i++){
      fprintf(fp, ",(*b*){");
      QuadTree_print_internal(fp, q->qts[i], level + 1); 
      fprintf(fp, "}");
    }
  }


}

void QuadTree_print(FILE *fp, QuadTree q){
  if (!fp) return;
  if (q->dim == 2){
    fprintf(fp, "Graphics[{");
  } else if (q->dim == 3){
    fprintf(fp, "Graphics3D[{");
  } else {
    return;
  }
  QuadTree_print_internal(fp, q, 0);
  if (q->dim == 2){
    fprintf(fp, "}, PlotRange -> All, Frame -> True, FrameTicks -> True]\n");
  } else {
    fprintf(fp, "}, PlotRange -> All]\n");
  }
}

static void QuadTree_get_nearest_internal(QuadTree qt, double *x, double *y,
                                          double *min, int *imin,
                                          bool tentative) {
  /* get the nearest point years to {x[0], ..., x[dim]} and store in y.*/
  double *coord, dist;
  int dim, i, iq = -1;
  double qmin;

  if (!qt) return;
  dim = qt->dim;
  node_data l = qt->l;
  while (l){
    coord = l->coord;
    dist = point_distance(x, coord, dim);
    if(*min < 0 || dist < *min) {
      *min = dist;
      *imin = l->id;
      for (i = 0; i < dim; i++) y[i] = coord[i];
    }
    l = l->next;
  }
  
  if (qt->qts){
    dist = point_distance(qt->center, x, dim); 
    if (*min >= 0 && (dist - sqrt((double) dim) * qt->width > *min)){
      return;
    } else {
      if (tentative){/* quick first approximation*/
	qmin = -1;
	for (i = 0; i < 1<<dim; i++){
	  if (qt->qts[i]){
	    dist = point_distance(qt->qts[i]->average, x, dim);
	    if (dist < qmin || qmin < 0){
	      qmin = dist; iq = i;
	    }
	  }
	}
	assert(iq >= 0);
	QuadTree_get_nearest_internal(qt->qts[iq], x, y, min, imin, tentative);
      } else {
	for (i = 0; i < 1<<dim; i++){
	  QuadTree_get_nearest_internal(qt->qts[i], x, y, min, imin, tentative);
	}
      }
    }
  }

}


void QuadTree_get_nearest(QuadTree qt, double *x, double *ymin, int *imin, double *min){

  *min = -1;

  QuadTree_get_nearest_internal(qt, x, ymin, min, imin, true);
  QuadTree_get_nearest_internal(qt, x, ymin, min, imin, false);
}
