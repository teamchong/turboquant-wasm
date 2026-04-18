/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#define STANDALONE
#include <sparse/general.h>
#include <sparse/DotIO.h>
#include <sparse/clustering.h>
#include <math.h>
#include <sparse/mq.h>
#include <sparse/color_palette.h>
#include <sparse/colorutil.h>
#include <stdbool.h>
#include <string.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/startswith.h>
#include <util/unreachable.h>

typedef struct {
    Agrec_t h;
    unsigned int id;
} Agnodeinfo_t;

#define ND_id(n)  (((Agnodeinfo_t*)((n)->base.data))->id)

static void color_string(agxbuf *buf, int dim, double *color){
  if (dim > 3 || dim < 1){
    fprintf(stderr,"can only 1, 2 or 3 dimensional color space. with color value between 0 to 1\n");
    assert(0);
  }
  if (dim == 3){
    agxbprint(buf, "#%02x%02x%02x", MIN((unsigned int)(color[0] *255), 255),
	    MIN((unsigned int) (color[1]*255), 255), MIN((unsigned int)(color[2]*255), 255));
  } else if (dim == 1){
    agxbprint(buf, "#%02x%02x%02x", MIN((unsigned int)(color[0] * 255), 255),
	    MIN((unsigned int) (color[0]*255), 255), MIN((unsigned int)(color[0]*255), 255));
  } else if (dim == 2){
    agxbprint(buf, "#%02x%02x%02x", MIN((unsigned int)(color[0] * 255), 255),
	    0, MIN((unsigned int)(color[1]*255), 255));
  }
}

void attach_edge_colors(Agraph_t* g, int dim, double *colors){
  /* colors is array of dim*nedges, with color for edge i at colors[dim*i, dim(i+1))
   */
  Agsym_t* sym = agattr(g, AGEDGE, "color", 0); 
  Agedge_t* e;
  Agnode_t* n;
  agxbuf buf = {0};
  unsigned row, col;
  int ie = 0;

  if (!sym)
    sym = agattr (g, AGEDGE, "color", "");

  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    row = ND_id(n);
    for (e = agfstout (g, n); e; e = agnxtout (g, e)) {
      col = ND_id(aghead(e));
      if (row == col) continue;
      color_string(&buf, dim, colors + ie*dim);
      agxset(e, sym, agxbuse(&buf));
      ie++;
    }
  }
  agxbfree(&buf);
}

/* SparseMatrix_import_dot:
 * Assumes g is connected and simple, i.e., we can have a->b and b->a
 * but not a->b and a->b
 */
SparseMatrix SparseMatrix_import_dot(Agraph_t *g, int dim,
                                     double **x, int format) {
  SparseMatrix A = 0;
  Agnode_t* n;
  Agedge_t* e;
  Agsym_t *sym;
  Agsym_t *psym;
  int nnodes;
  int nedges;
  int i, row;
  int* I;
  int* J;
  double *val;
  double v;
  int type = MATRIX_TYPE_REAL;

  if (!g) return NULL;
  nnodes = agnnodes (g);
  nedges = agnedges (g);
  if (format != FORMAT_CSR && format != FORMAT_COORD) {
    fprintf (stderr, "Format %d not supported\n", format);
    graphviz_exit(1);
  }

  /* Assign node ids */
  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n))
    ND_id(n) = i++;

  if (format == FORMAT_COORD){
    A = SparseMatrix_new(i, i, nedges, MATRIX_TYPE_REAL, format);
    A->nz = nedges;
    I = A->ia;
    J = A->ja;
    val = A->a;
  } else {
    I = gv_calloc(nedges, sizeof(int));
    J = gv_calloc(nedges, sizeof(int));
    val = gv_calloc(nedges, sizeof(double));
  }

  sym = agattr(g, AGEDGE, "weight", NULL);
  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    row = ND_id(n);
    for (e = agfstout (g, n); e; e = agnxtout (g, e)) {
      I[i] = row;
      J[i] = ND_id(aghead(e));

      /* edge weight */
      if (sym) {
        if (sscanf (agxget(e,sym), "%lf", &v) != 1) v = 1;
      } else {
        v = 1;
      }
      val[i] = v;

      i++;
    }
  }
  
  if (x && (psym = agattr(g, AGNODE, "pos", NULL))) {
    bool has_positions = true;
    char* pval;
    if (!(*x)) {
      *x = gv_calloc(dim * nnodes, sizeof(double));
    }
    for (n = agfstnode (g); n && has_positions; n = agnxtnode (g, n)) {
      double xx,yy, zz,ww;
      int nitems;
      i = ND_id(n);
      if ((pval = agxget(n, psym)) && *pval) {
	if (dim == 2){
	  nitems = sscanf(pval, "%lf,%lf", &xx, &yy);
	  if (nitems != 2) {
            has_positions = false;
            agerrorf("Node \"%s\" pos has %d < 2 values", agnameof(n), nitems);
	  }
	  (*x)[i*dim] = xx;
	  (*x)[i*dim+1] = yy;
	} else if (dim == 3){
	  nitems = sscanf(pval, "%lf,%lf,%lf", &xx, &yy, &zz);
	  if (nitems != 3) {
            has_positions = false;
            agerrorf("Node \"%s\" pos has %d < 3 values", agnameof(n), nitems);
	  }
	  (*x)[i*dim] = xx;
	  (*x)[i*dim+1] = yy;
	  (*x)[i*dim+2] = zz;
	} else if (dim == 4){
	  nitems = sscanf(pval, "%lf,%lf,%lf,%lf", &xx, &yy, &zz,&ww);
	  if (nitems != 4) {
            has_positions = false;
            agerrorf("Node \"%s\" pos has %d < 4 values", agnameof(n), nitems);
	  }
	  (*x)[i*dim] = xx;
	  (*x)[i*dim+1] = yy;
	  (*x)[i*dim+2] = zz;
	  (*x)[i*dim+3] = ww;
	} else if (dim == 1){
	  nitems = sscanf(pval, "%lf", &xx);
	  if (nitems != 1){
	    SparseMatrix_delete(A);
	    A = NULL;
	    goto done;
          }
	  (*x)[i*dim] = xx;
	} else {
	  assert(0);
	}
      } else {
        has_positions = false;
	agerrorf("Node \"%s\" lacks position info", agnameof(n));
      }
    }
    if (!has_positions) {
      free(*x);
      *x = NULL;
    }
  }
  else if (x)
    agerrorf("Error: graph %s has missing \"pos\" information", agnameof(g));

  size_t sz = sizeof(double);
  if (format == FORMAT_CSR) A = SparseMatrix_from_coordinate_arrays(nedges, nnodes, nnodes, I, J, val, type, sz);

done:
  if (format != FORMAT_COORD){
    free(I);
    free(J);
    free(val);
  }

  return A;
}

/* get spline info */
int Import_dot_splines(Agraph_t* g, int *ne, char ***xsplines){
  /* get the list of splines for the edges in the order they appear, and store as a list of strings in xspline.
     If *xsplines = NULL, it will be allocated. On exit (*xsplines)[i] is the control point string for the i-th edge. This string
     is of the form "x1,y1 x2,y2...", the two end points of the edge is not included per Dot format 
     Return 1 if success. 0 if not.
  */
  Agnode_t* n;
  Agedge_t* e;
  Agsym_t *sym;
  int nedges;
  unsigned i;

  if (!g){
    return 0;
  }

  *ne = nedges = agnedges (g);

  /* Assign node ids */
  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n))
    ND_id(n) = i++;

  sym = agattr(g, AGEDGE, "pos", 0); 
  if (!sym) return 0;
 
  *xsplines = gv_calloc(nedges, sizeof(char*));

  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    for (e = agfstout (g, n); e; e = agnxtout (g, e)) {
      /* edge weight */
      char *pos = agxget (e, sym);
      (*xsplines)[i] = strdup(pos);
      i++;
    }
  }
  return 1;
}

static int hex2int(char h){
  if (h >= '0' && h <= '9') return h - '0';
  if (h >= 'a' && h <= 'f') return 10 + h - 'a';
  if (h >= 'A' && h <= 'F') return 10 + h - 'A';
  return 0;
}

static float hexcol2rgb(const char *h) {
  return (hex2int(h[0])*16 + hex2int(h[1]))/255.;
}

void Dot_SetClusterColor(Agraph_t* g, float *rgb_r,  float *rgb_g,  float *rgb_b, int *clusters){

  Agnode_t* n;
  agxbuf scluster = {0};
  unsigned i;
  Agsym_t* clust_clr_sym = agattr(g, AGNODE, "clustercolor", NULL); 

  if (!clust_clr_sym) clust_clr_sym = agattr(g, AGNODE, "clustercolor", "-1"); 
  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    i = ND_id(n);
    if (rgb_r && rgb_g && rgb_b) {
      rgb2hex(rgb_r[clusters[i]], rgb_g[clusters[i]], rgb_b[clusters[i]],
              &scluster, NULL);
    }
    agxset(n, clust_clr_sym, agxbuse(&scluster));
  }
  agxbfree(&scluster);
}

SparseMatrix Import_coord_clusters_from_dot(Agraph_t* g, int maxcluster, int dim, int *nn, double **label_sizes, double **x, int **clusters, float **rgb_r,  float **rgb_g,  float **rgb_b,  float **fsz, char ***labels, int default_color_scheme, int clustering_scheme, int useClusters){
  SparseMatrix A = 0;
  Agnode_t* n;
  Agedge_t* e;
  Agsym_t* sym;
  Agsym_t* clust_sym;
  Agsym_t* clust_clr_sym;
  int nnodes;
  int nedges;
  int i, row, ic,nc, j;
  double v;
  int type = MATRIX_TYPE_REAL;
  char scluster[100];
  float ff;

  int MAX_GRPS, MIN_GRPS;
  bool noclusterinfo = false;
  bool first = true;
  const float *pal;
  int max_color = MAX_COLOR;

  switch (default_color_scheme){
  case COLOR_SCHEME_BLUE_YELLOW:
    pal = &(palette_blue_to_yellow[0][0]);
    break;
  case COLOR_SCHEME_WHITE_RED:
    pal = &(palette_white_to_red[0][0]);
    break;
  case COLOR_SCHEME_GREY_RED:
    pal = &(palette_grey_to_red[0][0]);
    break;
  case COLOR_SCHEME_GREY:
    pal = &(palette_grey[0][0]);
    break;
  case COLOR_SCHEME_PASTEL:
    pal = &(palette_pastel[0][0]);
    break;
  case COLOR_SCHEME_SEQUENTIAL_SINGLEHUE_RED:
    pal = &(palette_sequential_singlehue_red[0][0]);
    break;
  case COLOR_SCHEME_SEQUENTIAL_SINGLEHUE_RED_LIGHTER:
    pal = &(palette_sequential_singlehue_red_lighter[0][0]);
    break;
  case COLOR_SCHEME_PRIMARY:
    pal = &(palette_primary[0][0]);
    break;
  case COLOR_SCHEME_ADAM_BLEND:
    pal = &(palette_adam_blend[0][0]);
    break;
  case COLOR_SCHEME_ADAM:
    pal = &(palette_adam[0][0]);
    max_color = 11;
    break;
  case COLOR_SCHEME_NONE:
    pal = NULL;
    break;
  default:
    pal = &(palette_pastel[0][0]);
    break;
  }
    
  if (!g) return NULL;
  nnodes = agnnodes (g);
  nedges = agnedges (g);
  *nn = nnodes;

  /* Assign node ids */
  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n))
    ND_id(n) = i++;

  /* form matrix */  
  int* I = gv_calloc(nedges, sizeof(int));
  int* J = gv_calloc(nedges, sizeof(int));
  double* val = gv_calloc(nedges, sizeof(double));

  sym = agattr(g, AGEDGE, "weight", NULL); 
  clust_sym = agattr(g, AGNODE, "cluster", NULL); 
  clust_clr_sym = agattr(g, AGNODE, "clustercolor", NULL); 
  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    row = ND_id(n);
    for (e = agfstout (g, n); e; e = agnxtout (g, e)) {
      I[i] = row;
      J[i] = ND_id(aghead(e));
      if (sym) {
        if (sscanf (agxget(e,sym), "%lf", &v) != 1)
          v = 1;
      }
      else
        v = 1;
      val[i] = v;
      i++;
    }
  }
  A = SparseMatrix_from_coordinate_arrays(nedges, nnodes, nnodes, I, J, val,
                                          type, sizeof(double));

  /* get clustering info */
  *clusters = gv_calloc(nnodes, sizeof(int));
  nc = 1;
  MIN_GRPS = 0;
  /* if useClusters, the nodes in each top-level cluster subgraph are assigned to
   * clusters 2, 3, .... Any nodes not in a cluster subgraph are tossed into cluster 1.
   */
  if (useClusters) {
    Agraph_t* sg;
    int gid = 1;  
    memset (*clusters, 0, sizeof(int)*nnodes);
    for (sg = agfstsubg(g); sg; sg = agnxtsubg(sg)) {
      if (!startswith(agnameof(sg), "cluster")) continue;
      gid++;
      for (n = agfstnode(sg); n; n = agnxtnode (sg, n)) {
        i = ND_id(n);
        if ((*clusters)[i])
          fprintf (stderr, "Warning: node %s appears in multiple clusters.\n", agnameof(n));
        else
          (*clusters)[i] = gid;
      }
    }
    for (n = agfstnode(g); n; n = agnxtnode (g, n)) {
      i = ND_id(n);
      if ((*clusters)[i] == 0)
        (*clusters)[i] = 1;
    }
    MIN_GRPS = 1;
    nc = gid;
  }
  else if (clust_sym) {
    for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
      i = ND_id(n);
      if ((sscanf(agxget(n,clust_sym), "%d", &ic)>0)) {
        (*clusters)[i] = ic;
        nc = MAX(nc, ic);
        if (first){
	  MIN_GRPS = ic;
	  first = false;
        } else {
	  MIN_GRPS = MIN(MIN_GRPS, ic);
        }
      } else {
        noclusterinfo = true;
        break;
      }
    }
  }
  else 
    noclusterinfo = true;
  MAX_GRPS = nc;

  if (noclusterinfo) {
    double modularity;
    if (!clust_sym) clust_sym = agattr(g,AGNODE,"cluster","-1");

    if (clustering_scheme == CLUSTERING_MQ){
      mq_clustering(A, maxcluster,
		    &nc, clusters, &modularity);
    } else if (clustering_scheme == CLUSTERING_MODULARITY){ 
      modularity_clustering(A, false, maxcluster,
		    &nc, clusters, &modularity);
    } else {
      UNREACHABLE();
    }
    for (i = 0; i < nnodes; i++) (*clusters)[i]++;/* make into 1 based */
    for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
      i = ND_id(n);
      snprintf(scluster, sizeof(scluster), "%d", (*clusters)[i]);
      agxset(n,clust_sym,scluster);
    }
    MIN_GRPS = 1;
    MAX_GRPS = nc;
    if (Verbose){
      fprintf(stderr," no complement clustering info in dot file, using modularity clustering. Modularity = %f, ncluster=%d\n",modularity, nc);
    }
  }

  *label_sizes = gv_calloc(dim * nnodes, sizeof(double));
  if (pal || (!noclusterinfo && clust_clr_sym)){
    *rgb_r = gv_calloc(1 + MAX_GRPS, sizeof(float));
    *rgb_g = gv_calloc(1 + MAX_GRPS, sizeof(float));
    *rgb_b = gv_calloc(1 + MAX_GRPS, sizeof(float));
  } else {
    *rgb_r = NULL;
    *rgb_g = NULL;
    *rgb_b = NULL;
  }
  *fsz = gv_calloc(nnodes, sizeof(float));
  *labels = gv_calloc(nnodes, sizeof(char*));

  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    gvcolor_t color;
    double sz;
    i = ND_id(n);
    if (agget(n, "width") && agget(n, "height")){
      sscanf(agget(n, "width"), "%lf", &sz);
      (*label_sizes)[i*2] = POINTS(sz*0.5);
      sscanf(agget(n, "height"), "%lf", &sz);
      (*label_sizes)[i*2+1] = POINTS(sz*0.5);
    } else {
      (*label_sizes)[i*2] = POINTS(0.75/2);
      (*label_sizes)[i*2+1] = POINTS(0.5*2);
    }

   if (agget(n, "fontsize")){
      sscanf(agget(n, "fontsize"), "%f", &ff);
      (*fsz)[i] = ff;
    } else {
     (*fsz)[i] = 14;
    }

   if (agget(n, "label") && strcmp(agget(n, "label"), "") != 0 && strcmp(agget(n, "label"), "\\N") != 0){
     char *lbs = agget(n, "label");
      (*labels)[i] = strdup(lbs);
    } else {
     (*labels)[i] = strdup(agnameof(n));
    }



   j = (*clusters)[i];
   if (MAX_GRPS-MIN_GRPS < max_color) {
     j = (j-MIN_GRPS)*((int)((max_color-1)/MAX((MAX_GRPS-MIN_GRPS),1)));
   } else {
     j = (j-MIN_GRPS)%max_color;
   }

   if (pal){
     (*rgb_r)[(*clusters)[i]] = pal[3*j+0];
     (*rgb_g)[(*clusters)[i]] =  pal[3*j+1];
     (*rgb_b)[(*clusters)[i]] = pal[3*j+2];
   }

   if (!noclusterinfo && clust_clr_sym && (colorxlate(agxget(n,clust_clr_sym),&color,RGBA_DOUBLE) == COLOR_OK)) {
     (*rgb_r)[(*clusters)[i]] = color.u.RGBA[0];
     (*rgb_g)[(*clusters)[i]] = color.u.RGBA[1];
     (*rgb_b)[(*clusters)[i]] = color.u.RGBA[2];
   }

   const char *cc = agget(n, "clustercolor");
   if (!noclusterinfo && agget(n, "cluster") && cc && strlen(cc) >= 7 && pal) {
     (*rgb_r)[(*clusters)[i]] = hexcol2rgb(cc+1);
     (*rgb_g)[(*clusters)[i]] = hexcol2rgb(cc+3);
     (*rgb_b)[(*clusters)[i]] = hexcol2rgb(cc+5);
   }

  }


  if (x){
    bool has_position = false;
    *x = gv_calloc(dim * nnodes, sizeof(double));
    for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
      double xx,yy;
      i = ND_id(n);
      if (agget(n, "pos")){
        has_position = true;
	sscanf(agget(n, "pos"), "%lf,%lf", &xx, &yy);
	(*x)[i*dim] = xx;
	(*x)[i*dim+1] = yy;
      } else {
	fprintf(stderr,"WARNING: pos field missing for node %d, set to origin\n",i);
	(*x)[i*dim] = 0;
	(*x)[i*dim+1] = 0;
      }
    }
    if (!has_position){
      free(*x);
      *x = NULL;
    }
  }


  free(I);
  free(J);
  free(val);

  return A;
}

void attached_clustering(Agraph_t* g, int maxcluster, int clustering_scheme){
  SparseMatrix A = 0;
  Agnode_t* n;
  Agedge_t* e;
  Agsym_t *sym, *clust_sym;
  int nnodes;
  int nedges;
  int i, row,nc;
  double v;
  int type = MATRIX_TYPE_REAL;
  size_t sz = sizeof(double);

  if (!g) return;
  nnodes = agnnodes (g);
  nedges = agnedges (g);
  
  /* Assign node ids */
  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n))
    ND_id(n) = i++;
  
  /* form matrix */  
  int* I = gv_calloc(nedges, sizeof(int));
  int* J = gv_calloc(nedges, sizeof(int));
  double* val = gv_calloc(nedges, sizeof(double));
  
  sym = agattr(g, AGEDGE, "weight", NULL);
  clust_sym = agattr(g, AGNODE, "cluster", NULL);

  i = 0;
  for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
    row = ND_id(n);
    for (e = agfstout (g, n); e; e = agnxtout (g, e)) {
      I[i] = row;
      J[i] = ND_id(aghead(e));
      if (sym) {
        if (sscanf (agxget(e,sym), "%lf", &v) != 1)
          v = 1;
      }
      else
        v = 1;
      val[i] = v;
      i++;
    }
  }
  A = SparseMatrix_from_coordinate_arrays(nedges, nnodes, nnodes, I, J, val, type, sz);

  int *clusters = gv_calloc(nnodes, sizeof(int));

  {
    double modularity;
    if (!clust_sym) clust_sym = agattr(g,AGNODE,"cluster","-1");
    
    if (clustering_scheme == CLUSTERING_MQ){
      mq_clustering(A, maxcluster,
		    &nc, &clusters, &modularity);
    } else if (clustering_scheme == CLUSTERING_MODULARITY){ 
      modularity_clustering(A, false, maxcluster,
			    &nc, &clusters, &modularity);
    } else {
      UNREACHABLE();
    }
    for (i = 0; i < nnodes; i++) (clusters)[i]++;/* make into 1 based */
    for (n = agfstnode (g); n; n = agnxtnode (g, n)) {
      i = ND_id(n);
      agxbuf value_buffer = {0};
      agxbprint(&value_buffer, "%d", clusters[i]);
      agxset(n, clust_sym, agxbuse(&value_buffer));
      agxbfree(&value_buffer);
    }
    if (Verbose){
      fprintf(stderr," no complement clustering info in dot file, using modularity clustering. Modularity = %f, ncluster=%d\n",modularity, nc);
    }
  }

  free(I);
  free(J);
  free(val);
  free(clusters);

  SparseMatrix_delete(A);

}



void initDotIO (Agraph_t *g)
{
  aginit(g, AGNODE, "info", sizeof(Agnodeinfo_t), true);
}

void setDotNodeID (Agnode_t* n, int v)
{
    ND_id(n) = v;
}

int getDotNodeID (Agnode_t* n)
{
    return ND_id(n);
}

