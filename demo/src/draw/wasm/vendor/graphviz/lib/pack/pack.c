/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/* Module for packing disconnected graphs together.
 * Based on "Disconnected Graph Layout and the Polyomino Packing Approach",
 * K. Freivalds et al., GD0'01, LNCS 2265, pp. 378-391.
 */

#include "config.h"

#include <assert.h>
#include <common/geomprocs.h>
#include <common/pointset.h>
#include <common/render.h>
#include <math.h>
#include <pack/pack.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/alloc.h>
#include <util/prisize_t.h>
#include <util/sort.h>
#include <util/startswith.h>
#include <util/streq.h>

#define C 100 /* Max. avg. polyomino size */

#define MOVEPT(p) ((p).x += dx, (p).y += dy)

/// given cell size `s`, how many cells are required by size x?
static int GRID(double x, int s) {
  const double required = ceil(x / s);
  return (int)required;
}

/* Given grid cell size s, CVAL(v:int,s:int) returns index of cell containing
 * point v */
#define CVAL(v, s) ((v) >= 0 ? (v) / (s) : (((v) + 1) / (s)) - 1)
/* Given grid cell size s, CELL(p:point,s:int) sets p to cell containing point p
 */
#define CELL(p, s) ((p).x = CVAL((p).x, s), (p).y = CVAL((p).y, (s)))

typedef struct {
  int perim;     /* half size of bounding rectangle perimeter */
  pointf *cells; ///< cells in covering polyomino
  int nc;        /* no. of cells */
  size_t index;  ///<  index in original array
} ginfo;

typedef struct {
  double width, height;
  size_t index; ///< index in original array
} ainfo;

/* Compute grid step size. This is a root of the
 * quadratic equation a×l² + b×l + c, where a, b and
 * c are defined below.
 */
static int computeStep(size_t ng, const boxf *bbs, unsigned int margin) {
  double l1, l2;
  double a, b, c, d, r;
  double W, H; /* width and height of graph, with margin */
  int root;

  a = C * (double)ng - 1;
  c = 0;
  b = 0;
  for (size_t i = 0; i < ng; i++) {
    boxf bb = bbs[i];
    W = bb.UR.x - bb.LL.x + 2 * margin;
    H = bb.UR.y - bb.LL.y + 2 * margin;
    b -= W + H;
    c -= W * H;
  }
  d = b * b - 4.0 * a * c;
  assert(d >= 0);
  r = sqrt(d);
  l1 = (-b + r) / (2 * a);
  l2 = (-b - r) / (2 * a);
  root = (int)l1;
  if (root == 0)
    root = 1;
  if (Verbose > 2) {
    fprintf(stderr, "Packing: compute grid size\n");
    fprintf(stderr, "a %f b %f c %f d %f r %f\n", a, b, c, d, r);
    fprintf(stderr, "root %d (%f) %d (%f)\n", root, l1, (int)l2, l2);
    fprintf(stderr, " r1 %f r2 %f\n", a * l1 * l1 + b * l1 + c,
            a * l2 * l2 + b * l2 + c);
  }

  return root;
}

/* Comparison function for polyominoes.
 * Size is determined by perimeter.
 */
static int cmpf(const void *X, const void *Y) {
  const ginfo *x = *(ginfo *const *)X;
  const ginfo *y = *(ginfo *const *)Y;
  /* flip order to get descending array */
  if (y->perim < x->perim) {
    return -1;
  }
  if (y->perim > x->perim) {
    return 1;
  }
  return 0;
}

/// `sgn`, as defined in Graphics Gems I, §11.8, pp. 99
static int sgn(int x) { return x > 0 ? 1 : -1; }

/* Mark cells crossed by line from cell p to cell q.
 * Bresenham's algorithm, from Graphics Gems I, pp. 99-100.
 */
static void fillLine(pointf p, pointf q, PointSet *ps) {
  int x1 = ROUND(p.x);
  int y1 = ROUND(p.y);
  int x2 = ROUND(q.x);
  int y2 = ROUND(q.y);
  int d, x, y, ax, ay, sx, sy, dx, dy;

  dx = x2 - x1;
  ax = abs(dx) << 1;
  sx = sgn(dx);
  dy = y2 - y1;
  ay = abs(dy) << 1;
  sy = sgn(dy);

  x = x1;
  y = y1;
  if (ax > ay) { /* x dominant */
    d = ay - (ax >> 1);
    for (;;) {
      addPS(ps, x, y);
      if (x == x2)
        return;
      if (d >= 0) {
        y += sy;
        d -= ax;
      }
      x += sx;
      d += ay;
    }
  } else { /* y dominant */
    d = ax - (ay >> 1);
    for (;;) {
      addPS(ps, x, y);
      if (y == y2)
        return;
      if (d >= 0) {
        x += sx;
        d -= ay;
      }
      y += sy;
      d += ax;
    }
  }
}

/* It appears that spline_edges always have the start point at the
 * beginning and the end point at the end.
 */
static void fillEdge(Agedge_t *e, pointf p, PointSet *ps, double dx, double dy,
                     int ssize, bool doS) {
  size_t k;
  bezier bz;
  pointf pt, hpt;
  Agnode_t *h;

  pt = p;

  /* If doS is false or the edge has not splines, use line segment */
  if (!doS || !ED_spl(e)) {
    h = aghead(e);
    hpt = coord(h);
    MOVEPT(hpt);
    CELL(hpt, ssize);
    fillLine(pt, hpt, ps);
    return;
  }

  for (size_t j = 0; j < ED_spl(e)->size; j++) {
    bz = ED_spl(e)->list[j];
    if (bz.sflag) {
      pt = bz.sp;
      hpt = bz.list[0];
      k = 1;
    } else {
      pt = bz.list[0];
      hpt = bz.list[1];
      k = 2;
    }
    MOVEPT(pt);
    CELL(pt, ssize);
    MOVEPT(hpt);
    CELL(hpt, ssize);
    fillLine(pt, hpt, ps);

    for (; k < bz.size; k++) {
      pt = hpt;
      hpt = bz.list[k];
      MOVEPT(hpt);
      CELL(hpt, ssize);
      fillLine(pt, hpt, ps);
    }

    if (bz.eflag) {
      pt = hpt;
      hpt = bz.ep;
      MOVEPT(hpt);
      CELL(hpt, ssize);
      fillLine(pt, hpt, ps);
    }
  }
}

/* Generate polyomino info from graph using the bounding box of
 * the graph.
 */
static void genBox(boxf bb0, ginfo *info, int ssize, unsigned int margin,
                   pointf center, char *s) {
  PointSet *ps;
  int W, H;
  pointf UR, LL;
  double x, y;

  const boxf bb = {.LL = {.x = round(bb0.LL.x), .y = round(bb0.LL.y)},
                   .UR = {.x = round(bb0.UR.x), .y = round(bb0.UR.y)}};
  ps = newPS();

  LL.x = center.x - margin;
  LL.y = center.y - margin;
  UR.x = center.x + bb.UR.x - bb.LL.x + margin;
  UR.y = center.y + bb.UR.y - bb.LL.y + margin;
  CELL(LL, ssize);
  LL = (pointf){.x = round(LL.x), .y = round(LL.y)};
  CELL(UR, ssize);
  UR = (pointf){.x = round(UR.x), .y = round(UR.y)};

  for (x = LL.x; x <= UR.x; x++)
    for (y = LL.y; y <= UR.y; y++)
      addPS(ps, x, y);

  info->cells = pointsOf(ps);
  info->nc = sizeOf(ps);
  W = GRID(bb0.UR.x - bb0.LL.x + 2 * margin, ssize);
  H = GRID(bb0.UR.y - bb0.LL.y + 2 * margin, ssize);
  info->perim = W + H;

  if (Verbose > 2) {
    int i;
    fprintf(stderr, "%s no. cells %d W %d H %d\n", s, info->nc, W, H);
    for (i = 0; i < info->nc; i++)
      fprintf(stderr, "  %.0f %.0f cell\n", info->cells[i].x, info->cells[i].y);
  }

  freePS(ps);
}

/* Generate polyomino info from graph.
 * We add all cells covered partially by the bounding box of the
 * node. If doSplines is true and an edge has a spline, we use the
 * polyline determined by the control point. Otherwise,
 * we use each cell crossed by a straight edge between the head and tail.
 * If mode = l_clust, we use the graph's GD_clust array to treat the
 * top level clusters like large nodes.
 * Returns 0 if okay.
 */
static int genPoly(Agraph_t *root, Agraph_t *g, ginfo *info, int ssize,
                   pack_info *pinfo, pointf center) {
  PointSet *ps;
  int W, H;
  Agraph_t *eg; /* graph containing edges */
  Agnode_t *n;
  Agedge_t *e;
  graph_t *subg;
  unsigned int margin = pinfo->margin;
  bool doSplines = pinfo->doSplines;

  if (root)
    eg = root;
  else
    eg = g;

  ps = newPS();
  const double dx = center.x - round(GD_bb(g).LL.x);
  const double dy = center.y - round(GD_bb(g).LL.y);

  if (pinfo->mode == l_clust) {
    int i;

    /* backup the alg data */
    void **alg = gv_calloc(agnnodes(g), sizeof(void *));
    for (i = 0, n = agfstnode(g); n; n = agnxtnode(g, n)) {
      alg[i++] = ND_alg(n);
      ND_alg(n) = 0;
    }

    /* do bbox of top clusters */
    for (i = 1; i <= GD_n_cluster(g); i++) {
      subg = GD_clust(g)[i];
      boxf bb = {
          .LL = {.x = round(GD_bb(subg).LL.x), .y = round(GD_bb(subg).LL.y)},
          .UR = {.x = round(GD_bb(subg).UR.x), .y = round(GD_bb(subg).UR.y)}};
      if (bb.UR.x > bb.LL.x && bb.UR.y > bb.LL.y) {
        MOVEPT(bb.LL);
        MOVEPT(bb.UR);
        bb.LL.x -= margin;
        bb.LL.y -= margin;
        bb.UR.x += margin;
        bb.UR.y += margin;
        CELL(bb.LL, ssize);
        bb.LL = (pointf){.x = round(bb.LL.x), .y = round(bb.LL.y)};
        CELL(bb.UR, ssize);
        bb.UR = (pointf){.x = round(bb.UR.x), .y = round(bb.UR.y)};

        for (double x = bb.LL.x; x <= bb.UR.x; x++)
          for (double y = bb.LL.y; y <= bb.UR.y; y++)
            addPS(ps, x, y);

        /* note which nodes are in clusters */
        for (n = agfstnode(subg); n; n = agnxtnode(subg, n))
          ND_clust(n) = subg;
      }
    }

    /* now do remaining nodes and edges */
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
      const pointf ptf = coord(n);
      pointf pt = {.x = round(ptf.x), .y = round(ptf.y)};
      MOVEPT(pt);
      if (!ND_clust(n)) { /* n is not in a top-level cluster */
        const pointf s2 = {.x = round(margin + ND_xsize(n) / 2),
                           .y = round(margin + ND_ysize(n) / 2)};
        pointf LL = sub_pointf(pt, s2);
        pointf UR = add_pointf(pt, s2);
        CELL(LL, ssize);
        LL = (pointf){.x = round(LL.x), .y = round(LL.y)};
        CELL(UR, ssize);
        UR = (pointf){.x = round(UR.x), .y = round(UR.y)};

        for (double x = LL.x; x <= UR.x; x++)
          for (double y = LL.y; y <= UR.y; y++)
            addPS(ps, x, y);

        CELL(pt, ssize);
        pt = (pointf){.x = round(pt.x), .y = round(pt.y)};
        for (e = agfstout(eg, n); e; e = agnxtout(eg, e)) {
          fillEdge(e, pt, ps, dx, dy, ssize, doSplines);
        }
      } else {
        CELL(pt, ssize);
        pt = (pointf){.x = round(pt.x), .y = round(pt.y)};
        for (e = agfstout(eg, n); e; e = agnxtout(eg, e)) {
          if (ND_clust(n) == ND_clust(aghead(e)))
            continue;
          fillEdge(e, pt, ps, dx, dy, ssize, doSplines);
        }
      }
    }

    /* restore the alg data */
    for (i = 0, n = agfstnode(g); n; n = agnxtnode(g, n)) {
      ND_alg(n) = alg[i++];
    }
    free(alg);

  } else
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
      const pointf ptf = coord(n);
      pointf pt = {.x = round(ptf.x), .y = round(ptf.y)};
      MOVEPT(pt);
      pointf s2 = {.x = round(margin + ND_xsize(n) / 2),
                   .y = round(margin + ND_ysize(n) / 2)};
      pointf LL = sub_pointf(pt, s2);
      pointf UR = add_pointf(pt, s2);
      CELL(LL, ssize);
      LL = (pointf){.x = round(LL.x), .y = round(LL.y)};
      CELL(UR, ssize);
      UR = (pointf){.x = round(UR.x), .y = round(UR.y)};

      for (double x = LL.x; x <= UR.x; x++)
        for (double y = LL.y; y <= UR.y; y++)
          addPS(ps, x, y);

      CELL(pt, ssize);
      pt = (pointf){.x = round(pt.x), .y = round(pt.y)};
      for (e = agfstout(eg, n); e; e = agnxtout(eg, e)) {
        fillEdge(e, pt, ps, dx, dy, ssize, doSplines);
      }
    }

  info->cells = pointsOf(ps);
  info->nc = sizeOf(ps);
  W = GRID(GD_bb(g).UR.x - GD_bb(g).LL.x + 2 * margin, ssize);
  H = GRID(GD_bb(g).UR.y - GD_bb(g).LL.y + 2 * margin, ssize);
  info->perim = W + H;

  if (Verbose > 2) {
    int i;
    fprintf(stderr, "%s no. cells %d W %d H %d\n", agnameof(g), info->nc, W, H);
    for (i = 0; i < info->nc; i++)
      fprintf(stderr, "  %.0f %.0f cell\n", info->cells[i].x, info->cells[i].y);
  }

  freePS(ps);
  return 0;
}

/* Check if polyomino fits at given point.
 * If so, add cells to pointset, store point in place and return true.
 */
static int fits(int x, int y, ginfo *info, PointSet *ps, pointf *place,
                int step, const boxf *bbs) {
  pointf *cells = info->cells;
  int n = info->nc;
  int i;

  for (i = 0; i < n; i++) {
    pointf cell = *cells;
    cell.x += x;
    cell.y += y;
    if (inPS(ps, cell))
      return 0;
    cells++;
  }

  const pointf LL = {.x = round(bbs[info->index].LL.x),
                     .y = round(bbs[info->index].LL.y)};
  place->x = step * x - LL.x;
  place->y = step * y - LL.y;

  cells = info->cells;
  for (i = 0; i < n; i++) {
    pointf cell = *cells;
    cell.x += x;
    cell.y += y;
    insertPS(ps, cell);
    cells++;
  }

  if (Verbose >= 2)
    fprintf(stderr, "cc (%d cells) at (%d,%d) (%.0f,%.0f)\n", n, x, y, place->x,
            place->y);
  return 1;
}

/* Position fixed graph. Store final translation and
 * fill polyomino set. Note that polyomino set for the
 * graph is constructed where it will be.
 */
static void placeFixed(ginfo *info, PointSet *ps, pointf *place,
                       pointf center) {
  pointf *cells = info->cells;
  int n = info->nc;
  int i;

  place->x = -center.x;
  place->y = -center.y;

  for (i = 0; i < n; i++) {
    insertPS(ps, *cells++);
  }

  if (Verbose >= 2)
    fprintf(stderr, "cc (%d cells) at (%.0f,%.0f)\n", n, place->x, place->y);
}

/* Search for points on concentric "circles" out
 * from the origin. Check if polyomino can be placed
 * with bounding box origin at point.
 * First graph (i == 0) is centered on the origin if possible.
 */
static void placeGraph(size_t i, ginfo *info, PointSet *ps, pointf *place,
                       int step, unsigned int margin, const boxf *bbs) {
  int x, y;
  int bnd;
  boxf bb = bbs[info->index];

  if (i == 0) {
    const int W = GRID(bb.UR.x - bb.LL.x + 2 * margin, step);
    const int H = GRID(bb.UR.y - bb.LL.y + 2 * margin, step);
    if (fits(-W / 2, -H / 2, info, ps, place, step, bbs))
      return;
  }

  if (fits(0, 0, info, ps, place, step, bbs))
    return;
  const double W = ceil(bb.UR.x - bb.LL.x);
  const double H = ceil(bb.UR.y - bb.LL.y);
  if (W >= H) {
    for (bnd = 1;; bnd++) {
      x = 0;
      y = -bnd;
      for (; x < bnd; x++)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; y < bnd; y++)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; x > -bnd; x--)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; y > -bnd; y--)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; x < 0; x++)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
    }
  } else {
    for (bnd = 1;; bnd++) {
      y = 0;
      x = -bnd;
      for (; y > -bnd; y--)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; x < bnd; x++)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; y < bnd; y++)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; x > -bnd; x--)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
      for (; y > 0; y--)
        if (fits(x, y, info, ps, place, step, bbs))
          return;
    }
  }
}

#ifdef DEBUG
void dumpp(ginfo *info, char *pfx) {
  pointf *cells = info->cells;
  int i, c_cnt = info->nc;

  fprintf(stderr, "%s\n", pfx);
  for (i = 0; i < c_cnt; i++) {
    fprintf(stderr, "%.0f %.0f box\n", cells[i].x, cells[i].y);
  }
}
#endif

/// Sort by user values.
static int ucmpf(const void *X, const void *Y, void *user_values) {
  const ainfo *x = *(ainfo *const *)X;
  const ainfo *y = *(ainfo *const *)Y;
  const packval_t *userVals = user_values;

  const unsigned int dX = userVals[x->index];
  const unsigned int dY = userVals[y->index];
  if (dX > dY)
    return 1;
  if (dX < dY)
    return -1;
  return 0;
}

/// Sort by height + width
static int acmpf(const void *X, const void *Y) {
  const ainfo *x = *(ainfo *const *)X;
  const ainfo *y = *(ainfo *const *)Y;
  double dX = x->height + x->width;
  double dY = y->height + y->width;
  if (dX < dY)
    return 1;
  if (dX > dY)
    return -1;
  return 0;
}

/// step to the next iteration of a matrix cell loop
///
/// @param m Is the matrix in row-major order?
/// @param c [in,out] Column index
/// @param r [in,out] Row index
/// @param nc Total column count
/// @param nr Total row count
static void INC(bool m, size_t *c, size_t *r, size_t nc, size_t nr) {
  if (m) {
    (*c)++;
    if (*c == nc) {
      *c = 0;
      (*r)++;
    }
  } else {
    (*r)++;
    if (*r == nr) {
      *r = 0;
      (*c)++;
    }
  }
}

static pointf *arrayRects(size_t ng, const boxf *gs, pack_info *pinfo) {
  size_t nr = 0, nc;
  size_t r, c;
  ainfo *info;
  double v, wd, ht;
  pointf *places = gv_calloc(ng, sizeof(pointf));
  boxf bb;
  int sz;
  bool rowMajor;

  /* set up no. of rows and columns */
  sz = pinfo->sz;
  if (pinfo->flags & PK_COL_MAJOR) {
    rowMajor = false;
    if (sz > 0) {
      nr = (size_t)sz;
      nc = (ng + (nr - 1)) / nr;
    } else {
      nr = ceil(sqrt(ng));
      nc = (ng + (nr - 1)) / nr;
    }
  } else {
    rowMajor = true;
    if (sz > 0) {
      nc = (size_t)sz;
      nr = (ng + (nc - 1)) / nc;
    } else {
      nc = ceil(sqrt(ng));
      nr = (ng + (nc - 1)) / nc;
    }
  }
  if (Verbose)
    fprintf(stderr,
            "array packing: %s %" PRISIZE_T " rows %" PRISIZE_T " columns\n",
            rowMajor ? "row major" : "column major", nr, nc);
  double *widths = gv_calloc(nc + 1, sizeof(double));
  double *heights = gv_calloc(nr + 1, sizeof(double));

  ainfo *ip = info = gv_calloc(ng, sizeof(ainfo));
  for (size_t i = 0; i < ng; i++, ip++) {
    bb = gs[i];
    ip->width = bb.UR.x - bb.LL.x + pinfo->margin;
    ip->height = bb.UR.y - bb.LL.y + pinfo->margin;
    ip->index = i;
  }

  ainfo **sinfo = gv_calloc(ng, sizeof(ainfo *));
  for (size_t i = 0; i < ng; i++) {
    sinfo[i] = info + i;
  }

  if (pinfo->vals) {
    gv_sort(sinfo, ng, sizeof(ainfo *), ucmpf, pinfo->vals);
  } else if (!(pinfo->flags & PK_INPUT_ORDER)) {
    qsort(sinfo, ng, sizeof(ainfo *), acmpf);
  }

  /* compute column widths and row heights */
  r = c = 0;
  for (size_t i = 0; i < ng; i++, ip++) {
    ip = sinfo[i];
    widths[c] = fmax(widths[c], ip->width);
    heights[r] = fmax(heights[r], ip->height);
    INC(rowMajor, &c, &r, nc, nr);
  }

  /* convert widths and heights to positions */
  wd = 0;
  for (size_t i = 0; i <= nc; i++) {
    v = widths[i];
    widths[i] = wd;
    wd += v;
  }

  ht = 0;
  for (size_t i = nr; 0 < i; i--) {
    v = heights[i - 1];
    heights[i] = ht;
    ht += v;
  }
  heights[0] = ht;

  /* position rects */
  r = c = 0;
  for (size_t i = 0; i < ng; i++, ip++) {
    ip = sinfo[i];
    const size_t idx = ip->index;
    bb = gs[idx];
    if (pinfo->flags & PK_LEFT_ALIGN)
      places[idx].x = round(widths[c]);
    else if (pinfo->flags & PK_RIGHT_ALIGN)
      places[idx].x = round(widths[c + 1] - (bb.UR.x - bb.LL.x));
    else
      places[idx].x =
          round((widths[c] + widths[c + 1] - bb.UR.x - bb.LL.x) / 2.0);
    if (pinfo->flags & PK_TOP_ALIGN)
      places[idx].y = round(heights[r] - (bb.UR.y - bb.LL.y));
    else if (pinfo->flags & PK_BOT_ALIGN)
      places[idx].y = round(heights[r + 1]);
    else
      places[idx].y =
          round((heights[r] + heights[r + 1] - bb.UR.y - bb.LL.y) / 2.0);
    INC(rowMajor, &c, &r, nc, nr);
  }

  free(info);
  free(sinfo);
  free(widths);
  free(heights);
  return places;
}

static pointf *polyRects(size_t ng, const boxf *gs, pack_info *pinfo) {
  int stepSize;
  Dict_t *ps;

  /* calculate grid size */
  stepSize = computeStep(ng, gs, pinfo->margin);
  if (Verbose)
    fprintf(stderr, "step size = %d\n", stepSize);
  if (stepSize <= 0)
    return 0;

  /* generate polyomino cover for the rectangles */
  ginfo *info = gv_calloc(ng, sizeof(ginfo));
  for (size_t i = 0; i < ng; i++) {
    info[i].index = i;
    genBox(gs[i], info + i, stepSize, pinfo->margin, (pointf){0}, "");
  }

  /* sort */
  ginfo **sinfo = gv_calloc(ng, sizeof(ginfo *));
  for (size_t i = 0; i < ng; i++) {
    sinfo[i] = info + i;
  }
  qsort(sinfo, ng, sizeof(ginfo *), cmpf);

  ps = newPS();
  pointf *places = gv_calloc(ng, sizeof(pointf));
  for (size_t i = 0; i < ng; i++)
    placeGraph(i, sinfo[i], ps, places + sinfo[i]->index, stepSize,
               pinfo->margin, gs);

  free(sinfo);
  for (size_t i = 0; i < ng; i++)
    free(info[i].cells);
  free(info);
  freePS(ps);

  if (Verbose > 1)
    for (size_t i = 0; i < ng; i++)
      fprintf(stderr, "pos[%" PRISIZE_T "] %.0f %.0f\n", i, places[i].x,
              places[i].y);

  return places;
}

/* Given a collection of graphs, reposition them in the plane
 *  to not overlap but pack "nicely".
 *   ng is the number of graphs
 *   gs is a pointer to an array of graph pointers
 *   root gives the graph containing the edges; if null, the function
 *     looks in each graph in gs for its edges
 *   pinfo->margin gives the amount of extra space left around nodes in points
 *   If pinfo->doSplines is true, use edge splines, if computed,
 *     in calculating polyomino.
 *   pinfo->mode specifies the packing granularity and technique:
 *     l_node : pack at the node/cluster level
 *     l_graph : pack at the bounding box level
 *  Returns array of points to which graphs should be translated;
 *  the array needs to be freed;
 * Returns NULL if problem occur or if ng == 0.
 *
 * Depends on graph fields GD_bb, node fields ND_pos(inches), ND_xsize and
 * ND_ysize, and edge field ED_spl.
 *
 * FIX: fixed mode does not always work. The fixed ones get translated
 * back to be centered on the origin.
 * FIX: Check CELL and GRID macros for negative coordinates
 * FIX: Check width and height computation
 */
static pointf *polyGraphs(size_t ng, Agraph_t **gs, Agraph_t *root,
                          pack_info *pinfo) {
  int stepSize;
  ginfo *info;
  Dict_t *ps;
  bool *fixed = pinfo->fixed;
  int fixed_cnt = 0;
  boxf fixed_bb = {{0, 0}, {0, 0}};

  if (ng == 0)
    return 0;

  /* update bounding box info for each graph */
  /* If fixed, compute bbox of fixed graphs */
  for (size_t i = 0; i < ng; i++) {
    Agraph_t *g = gs[i];
    compute_bb(g);
    if (fixed && fixed[i]) {
      const boxf bb = {
          .LL = {.x = round(GD_bb(g).LL.x), .y = round(GD_bb(g).LL.y)},
          .UR = {.x = round(GD_bb(g).UR.x), .y = round(GD_bb(g).UR.y)}};
      if (fixed_cnt) {
        fixed_bb.LL.x = fmin(bb.LL.x, fixed_bb.LL.x);
        fixed_bb.LL.y = fmin(bb.LL.y, fixed_bb.LL.y);
        fixed_bb.UR.x = fmax(bb.UR.x, fixed_bb.UR.x);
        fixed_bb.UR.y = fmax(bb.UR.y, fixed_bb.UR.y);
      } else
        fixed_bb = bb;
      fixed_cnt++;
    }
    if (Verbose > 2) {
      fprintf(stderr, "bb[%s] %.5g %.5g %.5g %.5g\n", agnameof(g),
              GD_bb(g).LL.x, GD_bb(g).LL.y, GD_bb(g).UR.x, GD_bb(g).UR.y);
    }
  }

  /* calculate grid size */
  boxf *bbs = gv_calloc(ng, sizeof(boxf));
  for (size_t i = 0; i < ng; i++)
    bbs[i] = GD_bb(gs[i]);
  stepSize = computeStep(ng, bbs, pinfo->margin);
  if (Verbose)
    fprintf(stderr, "step size = %d\n", stepSize);
  if (stepSize <= 0) {
    free(bbs);
    return 0;
  }

  /* generate polyomino cover for the graphs */
  pointf center = {0};
  if (fixed) {
    center.x = round((fixed_bb.LL.x + fixed_bb.UR.x) / 2);
    center.y = round((fixed_bb.LL.y + fixed_bb.UR.y) / 2);
  }
  info = gv_calloc(ng, sizeof(ginfo));
  for (size_t i = 0; i < ng; i++) {
    Agraph_t *g = gs[i];
    info[i].index = i;
    if (pinfo->mode == l_graph)
      genBox(GD_bb(g), info + i, stepSize, pinfo->margin, center, agnameof(g));
    else if (genPoly(root, gs[i], info + i, stepSize, pinfo, center)) {
      free(bbs);
      return 0;
    }
  }

  /* sort */
  ginfo **sinfo = gv_calloc(ng, sizeof(ginfo *));
  for (size_t i = 0; i < ng; i++) {
    sinfo[i] = info + i;
  }
  qsort(sinfo, ng, sizeof(ginfo *), cmpf);

  ps = newPS();
  pointf *places = gv_calloc(ng, sizeof(pointf));
  if (fixed) {
    for (size_t i = 0; i < ng; i++) {
      if (fixed[i])
        placeFixed(sinfo[i], ps, places + sinfo[i]->index, center);
    }
    for (size_t i = 0; i < ng; i++) {
      if (!fixed[i])
        placeGraph(i, sinfo[i], ps, places + sinfo[i]->index, stepSize,
                   pinfo->margin, bbs);
    }
  } else {
    for (size_t i = 0; i < ng; i++)
      placeGraph(i, sinfo[i], ps, places + sinfo[i]->index, stepSize,
                 pinfo->margin, bbs);
  }

  free(sinfo);
  for (size_t i = 0; i < ng; i++)
    free(info[i].cells);
  free(info);
  freePS(ps);
  free(bbs);

  if (Verbose > 1)
    for (size_t i = 0; i < ng; i++)
      fprintf(stderr, "pos[%" PRISIZE_T "] %.0f %.0f\n", i, places[i].x,
              places[i].y);

  return places;
}

pointf *putGraphs(size_t ng, Agraph_t **gs, Agraph_t *root, pack_info *pinfo) {
  int v;
  Agraph_t *g;
  pointf *pts = NULL;
  char *s;

  if (ng == 0)
    return NULL;

  if (pinfo->mode <= l_graph)
    return polyGraphs(ng, gs, root, pinfo);

  boxf *bbs = gv_calloc(ng, sizeof(boxf));

  for (size_t i = 0; i < ng; i++) {
    g = gs[i];
    compute_bb(g);
    bbs[i] = GD_bb(g);
  }

  if (pinfo->mode == l_array) {
    if (pinfo->flags & PK_USER_VALS) {
      pinfo->vals = gv_calloc(ng, sizeof(packval_t));
      for (size_t i = 0; i < ng; i++) {
        s = agget(gs[i], "sortv");
        if (s && sscanf(s, "%d", &v) > 0 && v >= 0)
          pinfo->vals[i] = v;
      }
    }
    pts = arrayRects(ng, bbs, pinfo);
    if (pinfo->flags & PK_USER_VALS)
      free(pinfo->vals);
  }

  free(bbs);

  return pts;
}

pointf *putRects(size_t ng, boxf *bbs, pack_info *pinfo) {
  if (ng == 0)
    return NULL;
  if (pinfo->mode == l_node || pinfo->mode == l_clust)
    return NULL;
  if (pinfo->mode == l_graph)
    return polyRects(ng, bbs, pinfo);
  if (pinfo->mode == l_array)
    return arrayRects(ng, bbs, pinfo);
  return NULL;
}

/* Packs rectangles.
 *  ng - number of rectangles
 *  bbs - array of rectangles
 *  info - parameters used in packing
 * This decides where to layout the rectangles and repositions
 * the bounding boxes.
 *
 * Returns 0 on success.
 */
int packRects(size_t ng, boxf *bbs, pack_info *pinfo) {
  boxf bb;

  if (ng <= 1)
    return 0;

  pointf *pp = putRects(ng, bbs, pinfo);
  if (!pp)
    return 1;

  for (size_t i = 0; i < ng; i++) {
    bb = bbs[i];
    const pointf p = pp[i];
    bb.LL = add_pointf(bb.LL, p);
    bb.UR = add_pointf(bb.UR, p);
    bbs[i] = bb;
  }
  free(pp);
  return 0;
}

/// Translate all of the edge components by the given offset.
static void shiftEdge(Agedge_t *e, double dx, double dy) {

  if (ED_label(e))
    MOVEPT(ED_label(e)->pos);
  if (ED_xlabel(e))
    MOVEPT(ED_xlabel(e)->pos);
  if (ED_head_label(e))
    MOVEPT(ED_head_label(e)->pos);
  if (ED_tail_label(e))
    MOVEPT(ED_tail_label(e)->pos);

  if (ED_spl(e) == NULL)
    return;

  for (size_t j = 0; j < ED_spl(e)->size; j++) {
    bezier bz = ED_spl(e)->list[j];
    for (size_t k = 0; k < bz.size; k++)
      MOVEPT(bz.list[k]);
    if (bz.sflag)
      MOVEPT(ED_spl(e)->list[j].sp);
    if (bz.eflag)
      MOVEPT(ED_spl(e)->list[j].ep);
  }
}

static void shiftGraph(Agraph_t *g, double dx, double dy) {
  graph_t *subg;
  boxf bb = GD_bb(g);
  int i;

  bb.LL.x += dx;
  bb.UR.x += dx;
  bb.LL.y += dy;
  bb.UR.y += dy;
  GD_bb(g) = bb;

  if (GD_label(g) && GD_label(g)->set)
    MOVEPT(GD_label(g)->pos);

  for (i = 1; i <= GD_n_cluster(g); i++) {
    subg = GD_clust(g)[i];
    shiftGraph(subg, dx, dy);
  }
}

/* The function takes ng graphs gs and a similar
 * number of points pp and translates each graph so
 * that the lower left corner of the bounding box of graph gs[i] is at
 * point ps[i]. To do this, it assumes the bb field in
 * Agraphinfo_t accurately reflects the current graph layout.
 * The graph is repositioned by translating the pos and coord fields of
 * each node appropriately.
 *
 * If doSplines is non-zero, the function also translates splines coordinates
 * of each edge, if they have been calculated. In addition, edge labels are
 * repositioned.
 *
 * If root is non-NULL, it is taken as the root graph of
 * the graphs in gs and is used to find the edges. Otherwise, the function
 * uses the edges found in each graph gs[i].
 *
 * It returns 0 on success.
 *
 * This function uses the bb field in Agraphinfo_t,
 * the pos and coord fields in nodehinfo_t and
 * the spl field in Aedgeinfo_t.
 */
int shiftGraphs(size_t ng, Agraph_t **gs, pointf *pp, Agraph_t *root,
                bool doSplines) {
  double fx, fy;
  Agraph_t *g;
  Agraph_t *eg;
  Agnode_t *n;
  Agedge_t *e;

  if (ng == 0)
    return 0;

  for (size_t i = 0; i < ng; i++) {
    g = gs[i];
    if (root)
      eg = root;
    else
      eg = g;
    const pointf p = pp[i];
    const double dx = p.x;
    const double dy = p.y;
    fx = PS2INCH(dx);
    fy = PS2INCH(dy);

    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
      ND_pos(n)[0] += fx;
      ND_pos(n)[1] += fy;
      MOVEPT(ND_coord(n));
      if (ND_xlabel(n)) {
        MOVEPT(ND_xlabel(n)->pos);
      }
      if (doSplines) {
        for (e = agfstout(eg, n); e; e = agnxtout(eg, e))
          shiftEdge(e, dx, dy);
      }
    }
    shiftGraph(g, dx, dy);
  }

  return 0;
}

/* Packs graphs.
 *  ng - number of graphs
 *  gs - pointer to array of graphs
 *  root - graph used to find edges
 *  info - parameters used in packing
 *  info->doSplines - if true, use already computed spline control points
 * This decides where to layout the graphs and repositions the graph's
 * position info.
 *
 * Returns 0 on success.
 */
int packGraphs(size_t ng, Agraph_t **gs, Agraph_t *root, pack_info *info) {
  int ret;
  pointf *pp = putGraphs(ng, gs, root, info);

  if (!pp)
    return 1;
  ret = shiftGraphs(ng, gs, pp, root, info->doSplines);
  free(pp);
  return ret;
}

/* Packs subgraphs of given root graph, then recalculates root's bounding box.
 * Note that it does not recompute subgraph bounding boxes.
 * Cluster bounding boxes are recomputed in shiftGraphs.
 */
int packSubgraphs(size_t ng, Agraph_t **gs, Agraph_t *root, pack_info *info) {
  int ret;

  ret = packGraphs(ng, gs, root, info);
  if (ret == 0) {
    int j;
    boxf bb;
    graph_t *g;

    compute_bb(root);
    bb = GD_bb(root);
    for (size_t i = 0; i < ng; i++) {
      g = gs[i];
      for (j = 1; j <= GD_n_cluster(g); j++) {
        EXPANDBB(&bb, GD_bb(GD_clust(g)[j]));
      }
    }
    GD_bb(root) = bb;
  }
  return ret;
}

/// Pack subgraphs followed by postprocessing.
int pack_graph(size_t ng, Agraph_t **gs, Agraph_t *root, bool *fixed) {
  int ret;
  pack_info info;

  getPackInfo(root, l_graph, CL_OFFSET, &info);
  info.doSplines = true;
  info.fixed = fixed;
  ret = packSubgraphs(ng, gs, root, &info);
  if (ret == 0)
    dotneato_postprocess(root);
  return ret;
}

static const char *chkFlags(const char *p, pack_info *pinfo) {
  int c, more;

  if (*p != '_')
    return p;
  p++;
  more = 1;
  while (more && (c = *p)) {
    switch (c) {
    case 'c':
      pinfo->flags |= PK_COL_MAJOR;
      p++;
      break;
    case 'i':
      pinfo->flags |= PK_INPUT_ORDER;
      p++;
      break;
    case 'u':
      pinfo->flags |= PK_USER_VALS;
      p++;
      break;
    case 't':
      pinfo->flags |= PK_TOP_ALIGN;
      p++;
      break;
    case 'b':
      pinfo->flags |= PK_BOT_ALIGN;
      p++;
      break;
    case 'l':
      pinfo->flags |= PK_LEFT_ALIGN;
      p++;
      break;
    case 'r':
      pinfo->flags |= PK_RIGHT_ALIGN;
      p++;
      break;
    default:
      more = 0;
      break;
    }
  }
  return p;
}

static const char *mode2Str(pack_mode m) {

  switch (m) {
  case l_clust:
    return "cluster";
  case l_node:
    return "node";
  case l_graph:
    return "graph";
  case l_array:
    return "array";
  case l_aspect:
    return "aspect";
  case l_undef:
  default:
    break;
  }
  return "undefined";
}

/* Return pack_mode of graph using "packmode" attribute.
 * If not defined, return dflt
 */
pack_mode parsePackModeInfo(const char *p, pack_mode dflt, pack_info *pinfo) {
  float v;
  int i;

  assert(pinfo);
  pinfo->flags = 0;
  pinfo->mode = dflt;
  pinfo->sz = 0;
  pinfo->vals = NULL;
  if (p) {
    if (startswith(p, "array")) {
      pinfo->mode = l_array;
      p += strlen("array");
      p = chkFlags(p, pinfo);
      if (sscanf(p, "%d", &i) > 0 && i > 0)
        pinfo->sz = i;
    } else if (startswith(p, "aspect")) {
      pinfo->mode = l_aspect;
      if (sscanf(p + strlen("aspect"), "%f", &v) > 0 && v > 0)
        pinfo->aspect = v;
      else
        pinfo->aspect = 1;
    } else if (streq(p, "cluster")) {
      pinfo->mode = l_clust;
    } else if (streq(p, "graph")) {
      pinfo->mode = l_graph;
    } else if (streq(p, "node")) {
      pinfo->mode = l_node;
    }
  }

  if (Verbose) {
    fprintf(stderr, "pack info:\n");
    fprintf(stderr, "  mode   %s\n", mode2Str(pinfo->mode));
    if (pinfo->mode == l_aspect)
      fprintf(stderr, "  aspect %f\n", pinfo->aspect);
    fprintf(stderr, "  size   %d\n", pinfo->sz);
    fprintf(stderr, "  flags  %d\n", pinfo->flags);
  }
  return pinfo->mode;
}

/* Return pack_mode of graph using "packmode" attribute.
 * If not defined, return dflt
 */
pack_mode getPackModeInfo(Agraph_t *g, pack_mode dflt, pack_info *pinfo) {
  return parsePackModeInfo(agget(g, "packmode"), dflt, pinfo);
}

pack_mode getPackMode(Agraph_t *g, pack_mode dflt) {
  pack_info info;
  return getPackModeInfo(g, dflt, &info);
}

/* Return "pack" attribute of g.
 * If not defined or negative, return not_def.
 * If defined but not specified, return dflt.
 */
int getPack(Agraph_t *g, int not_def, int dflt) {
  char *p;
  int i;
  int v = not_def;

  if ((p = agget(g, "pack"))) {
    if (sscanf(p, "%d", &i) == 1 && i >= 0)
      v = i;
    else if (*p == 't' || *p == 'T')
      v = dflt;
  }

  return v;
}

pack_mode getPackInfo(Agraph_t *g, pack_mode dflt, int dfltMargin,
                      pack_info *pinfo) {
  assert(pinfo);

  pinfo->margin = getPack(g, dfltMargin, dfltMargin);
  if (Verbose) {
    fprintf(stderr, "  margin %u\n", pinfo->margin);
  }
  pinfo->doSplines = false;
  pinfo->fixed = NULL;
  getPackModeInfo(g, dflt, pinfo);

  return pinfo->mode;
}
/**
 * @dir lib/pack
 * @brief support for connected components, API pack.h
 *
 * [man 3 libpack](https://graphviz.org/pdf/pack.3.pdf)
 *
 */
