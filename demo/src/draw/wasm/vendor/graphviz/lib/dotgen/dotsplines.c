/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/*
 * set edge splines.
 */

#include "config.h"

#include <assert.h>
#include <common/boxes.h>
#include <dotgen/dot.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/gv_math.h>
#include <util/list.h>

#ifdef ORTHO
#include <ortho/ortho.h>
#endif

#define NSUB 9 /* number of subdivisions, re-aiming splines */

#define MINW 16 /* minimum width of a box in the edge path */
#define HALFMINW 8

#define FWDEDGE 16
#define BWDEDGE 32

#define MAINGRAPH 64
#define AUXGRAPH 128
#define GRAPHTYPEMASK 192 /* the OR of the above */

static void makefwdedge(edge_t *new, edge_t *old) {
  Agedgeinfo_t *const info =
      (Agedgeinfo_t *)((uintptr_t)new->base.data - offsetof(Agedgeinfo_t, hdr));
  const Agedgeinfo_t *const old_info =
      (Agedgeinfo_t *)((uintptr_t)old->base.data - offsetof(Agedgeinfo_t, hdr));
  *info = *old_info;
  *new = *old;
  new->base.data = &info->hdr;
  AGTAIL(new) = AGHEAD(old);
  AGHEAD(new) = AGTAIL(old);
  ED_tail_port(new) = ED_head_port(old);
  ED_head_port(new) = ED_tail_port(old);
  ED_edge_type(new) = VIRTUAL;
  ED_to_orig(new) = old;
}

typedef struct {
  double LeftBound;
  double RightBound;
  double Splinesep;
  double Multisep;
  boxf *Rank_box;
} spline_info_t;

typedef LIST(pointf) points_t;

static void adjustregularpath(path *, size_t, size_t);
static Agedge_t *bot_bound(Agedge_t *, int);
static bool pathscross(Agnode_t *, Agnode_t *, Agedge_t *, Agedge_t *);
static Agraph_t *cl_bound(graph_t *, Agnode_t *, Agnode_t *);
static bool cl_vninside(Agraph_t *, Agnode_t *);
static void completeregularpath(path *, Agedge_t *, Agedge_t *, pathend_t *,
                                pathend_t *, const boxes_t *);
static int edgecmp(const void *, const void *);
static int make_flat_edge(graph_t *, const spline_info_t, path *, Agedge_t **,
                          unsigned, int);
static void make_regular_edge(graph_t *g, spline_info_t *, path *, Agedge_t **,
                              unsigned, int);
static boxf makeregularend(boxf, int, double);
static boxf maximal_bbox(graph_t *g, const spline_info_t, Agnode_t *,
                         Agedge_t *, Agedge_t *);
static Agnode_t *neighbor(graph_t *, Agnode_t *, Agedge_t *, Agedge_t *, int);
static void place_vnlabel(Agnode_t *);
static boxf rank_box(spline_info_t *sp, Agraph_t *, int);
static void recover_slack(Agedge_t *, path *);
static void resize_vn(Agnode_t *, double, double, double);
static void setflags(Agedge_t *, int, int, int);
static int straight_len(Agnode_t *);
static Agedge_t *straight_path(Agedge_t *, int, points_t *);
static Agedge_t *top_bound(Agedge_t *, int);

static edge_t *getmainedge(edge_t *e) {
  edge_t *le = e;
  while (ED_to_virt(le))
    le = ED_to_virt(le);
  while (ED_to_orig(le))
    le = ED_to_orig(le);
  return le;
}

static bool spline_merge(node_t *n) {
  return ND_node_type(n) == VIRTUAL &&
         (ND_in(n).size > 1 || ND_out(n).size > 1);
}

static bool swap_ends_p(edge_t *e) {
  while (ED_to_orig(e))
    e = ED_to_orig(e);
  if (ND_rank(aghead(e)) > ND_rank(agtail(e)))
    return false;
  if (ND_rank(aghead(e)) < ND_rank(agtail(e)))
    return true;
  if (ND_order(aghead(e)) >= ND_order(agtail(e)))
    return false;
  return true;
}

static splineInfo sinfo = {.swapEnds = swap_ends_p,
                           .splineMerge = spline_merge};

int portcmp(port p0, port p1) {
  if (!p1.defined)
    return p0.defined ? 1 : 0;
  if (!p0.defined)
    return -1;
  if (p0.p.x < p1.p.x)
    return -1;
  if (p0.p.x > p1.p.x)
    return 1;
  if (p0.p.y < p1.p.y)
    return -1;
  if (p0.p.y > p1.p.y)
    return 1;
  return 0;
}

static void swap_bezier(bezier *b) {
  const size_t sz = b->size;
  for (size_t i = 0; i < sz / 2; ++i) { // reverse list of points
    SWAP(&b->list[i], &b->list[sz - 1 - i]);
  }

  SWAP(&b->sflag, &b->eflag);
  SWAP(&b->sp, &b->ep);
}

static void swap_spline(splines *s) {
  const size_t sz = s->size;

  // reverse list
  for (size_t i = 0; i < sz / 2; ++i) {
    SWAP(&s->list[i], &s->list[sz - 1 - i]);
  }

  // swap Béziers
  for (size_t i = 0; i < sz; ++i) {
    swap_bezier(&s->list[i]);
  }
}

/* Some back edges are reversed during layout and the reversed edge
 * is used to compute the spline. We would like to guarantee that
 * the order of control points always goes from tail to head, so
 * we reverse them if necessary.
 */
static void edge_normalize(graph_t *g) {
  for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
    for (edge_t *e = agfstout(g, n); e; e = agnxtout(g, e)) {
      if (sinfo.swapEnds(e) && ED_spl(e))
        swap_spline(ED_spl(e));
    }
  }
}

/* In position, each node has its rw stored in mval and,
 * if a node is part of a loop, rw may be increased to
 * reflect the loops and associated labels. We restore
 * the original value here.
 */
static void resetRW(graph_t *g) {
  for (node_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
    if (ND_other(n).list) {
      SWAP(&ND_rw(n), &ND_mval(n));
    }
  }
}

/* Set edge label position information for regular and non-adjacent flat edges.
 * Dot has allocated space and position for these labels. This info will be
 * used when routing orthogonal edges.
 */
static void setEdgeLabelPos(graph_t *g) {
  textlabel_t *l;

  /* place regular edge labels */
  for (node_t *n = GD_nlist(g); n; n = ND_next(n)) {
    if (ND_node_type(n) == VIRTUAL) {
      if (ND_alg(n)) { // label of non-adjacent flat edge
        edge_t *fe = ND_alg(n);
        l = ED_label(fe);
        assert(l);
        l->pos = ND_coord(n);
        l->set = true;
      } else if ((l = ND_label(n))) { // label of regular edge
        place_vnlabel(n);
      }
      if (l)
        updateBB(g, l);
    }
  }
}

/** Main spline routing code.
 * The normalize parameter allows this function to be called by the
 * recursive call in make_flat_edge without normalization occurring,
 * so that the edge will only be normalized once in the top level call
 * of dot_splines.
 *
 * @return 0 on success
 */
static int dot_splines_(graph_t *g, int normalize) {
  int i, j, n_nodes;
  node_t *n;
  Agedgeinfo_t fwdedgeai, fwdedgebi;
  Agedgepair_t fwdedgea, fwdedgeb;
  edge_t *e, *e0, *e1, *ea, *eb, *le1;
  path P = {0};
  int et = EDGE_TYPE(g);
  fwdedgea.out.base.data = &fwdedgeai.hdr;
  fwdedgeb.out.base.data = &fwdedgebi.hdr;

  if (et == EDGETYPE_NONE)
    return 0;
  if (et == EDGETYPE_CURVED) {
    resetRW(g);
    if (GD_has_labels(g->root) & EDGE_LABEL) {
      agwarningf("edge labels with splines=curved not supported in dot - use "
                 "xlabels\n");
    }
  }
  spline_info_t sd = {0};
  LIST(edge_t *) edges = {0};
#ifdef ORTHO
  if (et == EDGETYPE_ORTHO) {
    resetRW(g);
    if (GD_has_labels(g->root) & EDGE_LABEL) {
      setEdgeLabelPos(g);
      orthoEdges(g, true);
    } else
      orthoEdges(g, false);
    goto finish;
  }
#else
  (void)setEdgeLabelPos;
#endif

  mark_lowclusters(g);
  if (routesplinesinit())
    return 0;
  sd = (spline_info_t){.Splinesep = GD_nodesep(g) / 4,
                       .Multisep = GD_nodesep(g)};

  /* compute boundaries and list of splines */
  n_nodes = 0;
  for (i = GD_minrank(g); i <= GD_maxrank(g); i++) {
    n_nodes += GD_rank(g)[i].n;
    if ((n = GD_rank(g)[i].v[0]))
      sd.LeftBound = MIN(sd.LeftBound, ND_coord(n).x - ND_lw(n));
    if (GD_rank(g)[i].n && (n = GD_rank(g)[i].v[GD_rank(g)[i].n - 1]))
      sd.RightBound = MAX(sd.RightBound, ND_coord(n).x + ND_rw(n));
    sd.LeftBound -= MINW;
    sd.RightBound += MINW;

    for (j = 0; j < GD_rank(g)[i].n; j++) {
      n = GD_rank(g)[i].v[j];
      /* if n is the label of a flat edge, copy its position to
       * the label.
       */
      if (ND_alg(n)) {
        edge_t *fe = ND_alg(n);
        assert(ED_label(fe));
        ED_label(fe)->pos = ND_coord(n);
        ED_label(fe)->set = true;
      }
      if (ND_node_type(n) != NORMAL && !sinfo.splineMerge(n))
        continue;
      for (int k = 0; (e = ND_out(n).list[k]); k++) {
        if (ED_edge_type(e) == FLATORDER || ED_edge_type(e) == IGNORED)
          continue;
        setflags(e, REGULAREDGE, FWDEDGE, MAINGRAPH);
        LIST_APPEND(&edges, e);
      }
      if (ND_flat_out(n).list)
        for (int k = 0; (e = ND_flat_out(n).list[k]); k++) {
          setflags(e, FLATEDGE, 0, AUXGRAPH);
          LIST_APPEND(&edges, e);
        }
      if (ND_other(n).list) {
        /* In position, each node has its rw stored in mval and,
         * if a node is part of a loop, rw may be increased to
         * reflect the loops and associated labels. We restore
         * the original value here.
         */
        if (ND_node_type(n) == NORMAL) {
          SWAP(&ND_rw(n), &ND_mval(n));
        }
        for (int k = 0; (e = ND_other(n).list[k]); k++) {
          setflags(e, 0, 0, AUXGRAPH);
          LIST_APPEND(&edges, e);
        }
      }
    }
  }

  /* Sort so that equivalent edges are contiguous.
   * Equivalence should basically mean that 2 edges have the
   * same set {(tailnode,tailport),(headnode,headport)}, or
   * alternatively, the edges would be routed identically if
   * routed separately.
   */
  LIST_SORT(&edges, edgecmp);

  /* FIXME: just how many boxes can there be? */
  P.boxes = gv_calloc(n_nodes + 20 * 2 * NSUB, sizeof(boxf));
  sd.Rank_box = gv_calloc(i, sizeof(boxf));

  if (et == EDGETYPE_LINE) {
    /* place regular edge labels */
    for (n = GD_nlist(g); n; n = ND_next(n)) {
      if (ND_node_type(n) == VIRTUAL && ND_label(n)) {
        place_vnlabel(n);
      }
    }
  }

  for (unsigned l = 0; l < LIST_SIZE(&edges);) {
    const unsigned ind = l;
    edge_t *le0 = getmainedge((e0 = LIST_GET(&edges, l++)));
    if (ED_tail_port(e0).defined || ED_head_port(e0).defined) {
      ea = e0;
    } else {
      ea = le0;
    }
    if (ED_tree_index(ea) & BWDEDGE) {
      makefwdedge(&fwdedgea.out, ea);
      ea = &fwdedgea.out;
    }
    unsigned cnt;
    for (cnt = 1; l < LIST_SIZE(&edges); cnt++, l++) {
      if (le0 != (le1 = getmainedge((e1 = LIST_GET(&edges, l)))))
        break;
      if (ED_adjacent(e0))
        continue; /* all flat adjacent edges at once */
      if (ED_tail_port(e1).defined || ED_head_port(e1).defined) {
        eb = e1;
      } else {
        eb = le1;
      }
      if (ED_tree_index(eb) & BWDEDGE) {
        makefwdedge(&fwdedgeb.out, eb);
        eb = &fwdedgeb.out;
      }
      if (portcmp(ED_tail_port(ea), ED_tail_port(eb)))
        break;
      if (portcmp(ED_head_port(ea), ED_head_port(eb)))
        break;
      if ((ED_tree_index(e0) & EDGETYPEMASK) == FLATEDGE &&
          ED_label(e0) != ED_label(e1))
        break;
      if (ED_tree_index(LIST_GET(&edges, l)) & MAINGRAPH) /* Aha! -C is on */
        break;
    }

    if (et == EDGETYPE_CURVED) {
      edge_t **edgelist = gv_calloc(cnt, sizeof(edge_t *));
      edgelist[0] = getmainedge(LIST_GET(&edges, ind));
      for (unsigned ii = 1; ii < cnt; ii++)
        edgelist[ii] = LIST_GET(&edges, ind + ii);
      makeStraightEdges(g, edgelist, cnt, et, &sinfo);
      free(edgelist);
    } else if (agtail(e0) == aghead(e0)) {
      double sizey;
      n = agtail(e0);
      const int r = ND_rank(n);
      if (r == GD_maxrank(g)) {
        if (r > 0)
          sizey = ND_coord(GD_rank(g)[r - 1].v[0]).y - ND_coord(n).y;
        else
          sizey = ND_ht(n);
      } else if (r == GD_minrank(g)) {
        sizey = ND_coord(n).y - ND_coord(GD_rank(g)[r + 1].v[0]).y;
      } else {
        double upy = ND_coord(GD_rank(g)[r - 1].v[0]).y - ND_coord(n).y;
        double dwny = ND_coord(n).y - ND_coord(GD_rank(g)[r + 1].v[0]).y;
        sizey = fmin(upy, dwny);
      }
      makeSelfEdge(LIST_AT(&edges, ind), cnt, sd.Multisep, sizey / 2, &sinfo);
      for (unsigned b = 0; b < cnt; b++) {
        e = LIST_GET(&edges, ind + b);
        if (ED_label(e))
          updateBB(g, ED_label(e));
      }
    } else if (ND_rank(agtail(e0)) == ND_rank(aghead(e0))) {
      const int rc = make_flat_edge(g, sd, &P, LIST_AT(&edges, ind), cnt, et);
      if (rc != 0) {
        free(sd.Rank_box);
        LIST_FREE(&edges);
        free(P.boxes);
        return rc;
      }
    } else
      make_regular_edge(g, &sd, &P, LIST_AT(&edges, ind), cnt, et);
  }

  /* place regular edge labels */
  for (n = GD_nlist(g); n; n = ND_next(n)) {
    if (ND_node_type(n) == VIRTUAL && ND_label(n)) {
      place_vnlabel(n);
      updateBB(g, ND_label(n));
    }
  }

  /* normalize splines so they always go from tail to head */
  /* place_portlabel relies on this being done first */
  if (normalize)
    edge_normalize(g);

#ifdef ORTHO
finish:
#endif
  /* place port labels */
  /* FIX: head and tail labels are not part of cluster bbox */
  if ((E_headlabel || E_taillabel) && (E_labelangle || E_labeldistance)) {
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
      if (E_headlabel) {
        for (e = agfstin(g, n); e; e = agnxtin(g, e))
          if (ED_head_label(AGMKOUT(e))) {
            place_portlabel(AGMKOUT(e), true);
            updateBB(g, ED_head_label(AGMKOUT(e)));
          }
      }
      if (E_taillabel) {
        for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
          if (ED_tail_label(e)) {
            if (place_portlabel(e, false))
              updateBB(g, ED_tail_label(e));
          }
        }
      }
    }
  }

#ifdef ORTHO
  if (et != EDGETYPE_ORTHO && et != EDGETYPE_CURVED) {
#else
  if (et != EDGETYPE_CURVED) {
#endif
    routesplinesterm();
  }
  free(sd.Rank_box);
  LIST_FREE(&edges);
  free(P.boxes);
  State = GVSPLINES;
  EdgeLabelsDone = 1;
  return 0;
}

/* If the splines attribute is defined but equal to "", skip edge routing.
 *
 * @return 0 on success
 */
int dot_splines(graph_t *g) { return dot_splines_(g, 1); }

/* assign position of an edge label from its virtual node
 * This is for regular edges only.
 */
static void place_vnlabel(node_t *n) {
  edge_t *e;
  if (ND_in(n).size == 0)
    return; /* skip flat edge labels here */
  for (e = ND_out(n).list[0]; ED_edge_type(e) != NORMAL; e = ED_to_orig(e))
    ;
  const pointf dimen = ED_label(e)->dimen;
  const double width = GD_flip(agraphof(n)) ? dimen.y : dimen.x;
  ED_label(e)->pos.x = ND_coord(n).x + width / 2.0;
  ED_label(e)->pos.y = ND_coord(n).y;
  ED_label(e)->set = true;
}

static void setflags(edge_t *e, int hint1, int hint2, int f3) {
  int f1, f2;
  if (hint1 != 0)
    f1 = hint1;
  else {
    if (agtail(e) == aghead(e))
      if (ED_tail_port(e).defined || ED_head_port(e).defined)
        f1 = SELFWPEDGE;
      else
        f1 = SELFNPEDGE;
    else if (ND_rank(agtail(e)) == ND_rank(aghead(e)))
      f1 = FLATEDGE;
    else
      f1 = REGULAREDGE;
  }
  if (hint2 != 0)
    f2 = hint2;
  else {
    if (f1 == REGULAREDGE)
      f2 = ND_rank(agtail(e)) < ND_rank(aghead(e)) ? FWDEDGE : BWDEDGE;
    else if (f1 == FLATEDGE)
      f2 = ND_order(agtail(e)) < ND_order(aghead(e)) ? FWDEDGE : BWDEDGE;
    else /* f1 == SELF*EDGE */
      f2 = FWDEDGE;
  }
  ED_tree_index(e) = f1 | f2 | f3;
}

/* lexicographically order edges by
 *  - edge type
 *  - |rank difference of nodes|
 *  - |x difference of nodes|
 *  - id of witness edge for equivalence class
 *  - port comparison
 *  - graph type
 *  - labels if flat edges
 *  - edge id
 */
static int edgecmp(const void *p0, const void *p1) {
  edge_t *const *ptr0 = p0;
  edge_t *const *ptr1 = p1;
  Agedgeinfo_t fwdedgeai, fwdedgebi;
  Agedgepair_t fwdedgea, fwdedgeb;
  int rv;

  fwdedgea.out.base.data = &fwdedgeai.hdr;
  fwdedgeb.out.base.data = &fwdedgebi.hdr;
  edge_t *const e0 = *ptr0;
  edge_t *const e1 = *ptr1;
  int et0 = ED_tree_index(e0) & EDGETYPEMASK;
  int et1 = ED_tree_index(e1) & EDGETYPEMASK;
  if (et0 < et1) {
    return 1;
  }
  if (et0 > et1) {
    return -1;
  }

  edge_t *le0 = getmainedge(e0);
  edge_t *le1 = getmainedge(e1);

  {
    const int rank_diff0 = ND_rank(agtail(le0)) - ND_rank(aghead(le0));
    const int rank_diff1 = ND_rank(agtail(le1)) - ND_rank(aghead(le1));
    const int v0 = abs(rank_diff0);
    const int v1 = abs(rank_diff1);
    if (v0 < v1) {
      return -1;
    }
    if (v0 > v1) {
      return 1;
    }
  }

  {
    const double t0 = ND_coord(agtail(le0)).x - ND_coord(aghead(le0)).x;
    const double t1 = ND_coord(agtail(le1)).x - ND_coord(aghead(le1)).x;
    const double v0 = fabs(t0);
    const double v1 = fabs(t1);
    if (v0 < v1) {
      return -1;
    }
    if (v0 > v1) {
      return 1;
    }
  }

  /* This provides a cheap test for edges having the same set of endpoints.  */
  if (AGSEQ(le0) < AGSEQ(le1)) {
    return -1;
  }
  if (AGSEQ(le0) > AGSEQ(le1)) {
    return 1;
  }

  edge_t *ea =
      (ED_tail_port(e0).defined || ED_head_port(e0).defined) ? e0 : le0;
  if (ED_tree_index(ea) & BWDEDGE) {
    makefwdedge(&fwdedgea.out, ea);
    ea = &fwdedgea.out;
  }
  edge_t *eb =
      (ED_tail_port(e1).defined || ED_head_port(e1).defined) ? e1 : le1;
  if (ED_tree_index(eb) & BWDEDGE) {
    makefwdedge(&fwdedgeb.out, eb);
    eb = &fwdedgeb.out;
  }
  if ((rv = portcmp(ED_tail_port(ea), ED_tail_port(eb))))
    return rv;
  if ((rv = portcmp(ED_head_port(ea), ED_head_port(eb))))
    return rv;

  et0 = ED_tree_index(e0) & GRAPHTYPEMASK;
  et1 = ED_tree_index(e1) & GRAPHTYPEMASK;
  if (et0 < et1) {
    return -1;
  }
  if (et0 > et1) {
    return 1;
  }

  if (et0 == FLATEDGE) {
    if (ED_label(e0) < ED_label(e1)) {
      return -1;
    }
    if (ED_label(e0) > ED_label(e1)) {
      return 1;
    }
  }

  if (AGSEQ(e0) < AGSEQ(e1)) {
    return -1;
  }
  if (AGSEQ(e0) > AGSEQ(e1)) {
    return 1;
  }
  return 0;
}

typedef struct {
  attrsym_t *E_constr;
  attrsym_t *E_dir;
  attrsym_t *E_samehead;
  attrsym_t *E_sametail;
  attrsym_t *E_weight;
  attrsym_t *E_minlen;
  attrsym_t *E_fontcolor;
  attrsym_t *E_fontname;
  attrsym_t *E_fontsize;
  attrsym_t *E_headclip;
  attrsym_t *E_headlabel;
  attrsym_t *E_label;
  attrsym_t *E_label_float;
  attrsym_t *E_labelfontcolor;
  attrsym_t *E_labelfontname;
  attrsym_t *E_labelfontsize;
  attrsym_t *E_tailclip;
  attrsym_t *E_taillabel;
  attrsym_t *E_xlabel;

  attrsym_t *N_height;
  attrsym_t *N_width;
  attrsym_t *N_shape;
  attrsym_t *N_style;
  attrsym_t *N_fontsize;
  attrsym_t *N_fontname;
  attrsym_t *N_fontcolor;
  attrsym_t *N_label;
  attrsym_t *N_xlabel;
  attrsym_t *N_showboxes;
  attrsym_t *N_ordering;
  attrsym_t *N_sides;
  attrsym_t *N_peripheries;
  attrsym_t *N_skew;
  attrsym_t *N_orientation;
  attrsym_t *N_distortion;
  attrsym_t *N_fixed;
  attrsym_t *N_nojustify;
  attrsym_t *N_group;

  attrsym_t *G_ordering;
  int State;
} attr_state_t;

static void setState(graph_t *auxg, attr_state_t *attr_state) {
  /* save state */
  attr_state->E_constr = E_constr;
  attr_state->E_dir = E_dir;
  attr_state->E_samehead = E_samehead;
  attr_state->E_sametail = E_sametail;
  attr_state->E_weight = E_weight;
  attr_state->E_minlen = E_minlen;
  attr_state->E_fontcolor = E_fontcolor;
  attr_state->E_fontname = E_fontname;
  attr_state->E_fontsize = E_fontsize;
  attr_state->E_headclip = E_headclip;
  attr_state->E_headlabel = E_headlabel;
  attr_state->E_label = E_label;
  attr_state->E_label_float = E_label_float;
  attr_state->E_labelfontcolor = E_labelfontcolor;
  attr_state->E_labelfontname = E_labelfontname;
  attr_state->E_labelfontsize = E_labelfontsize;
  attr_state->E_tailclip = E_tailclip;
  attr_state->E_taillabel = E_taillabel;
  attr_state->E_xlabel = E_xlabel;
  attr_state->N_height = N_height;
  attr_state->N_width = N_width;
  attr_state->N_shape = N_shape;
  attr_state->N_style = N_style;
  attr_state->N_fontsize = N_fontsize;
  attr_state->N_fontname = N_fontname;
  attr_state->N_fontcolor = N_fontcolor;
  attr_state->N_label = N_label;
  attr_state->N_xlabel = N_xlabel;
  attr_state->N_showboxes = N_showboxes;
  attr_state->N_ordering = N_ordering;
  attr_state->N_sides = N_sides;
  attr_state->N_peripheries = N_peripheries;
  attr_state->N_skew = N_skew;
  attr_state->N_orientation = N_orientation;
  attr_state->N_distortion = N_distortion;
  attr_state->N_fixed = N_fixed;
  attr_state->N_nojustify = N_nojustify;
  attr_state->N_group = N_group;
  attr_state->State = State;
  attr_state->G_ordering = G_ordering;

  E_constr = NULL;
  E_dir = agattr_text(auxg, AGEDGE, "dir", NULL);
  E_samehead = agattr_text(auxg, AGEDGE, "samehead", NULL);
  E_sametail = agattr_text(auxg, AGEDGE, "sametail", NULL);
  E_weight = agattr_text(auxg, AGEDGE, "weight", NULL);
  if (!E_weight)
    E_weight = agattr_text(auxg, AGEDGE, "weight", "");
  E_minlen = NULL;
  E_fontcolor = NULL;
  E_fontname = agfindedgeattr(auxg, "fontname");
  E_fontsize = agfindedgeattr(auxg, "fontsize");
  E_headclip = agfindedgeattr(auxg, "headclip");
  E_headlabel = NULL;
  E_label = agfindedgeattr(auxg, "label");
  E_label_float = agfindedgeattr(auxg, "label_float");
  E_labelfontcolor = NULL;
  E_labelfontname = agfindedgeattr(auxg, "labelfontname");
  E_labelfontsize = agfindedgeattr(auxg, "labelfontsize");
  E_tailclip = agfindedgeattr(auxg, "tailclip");
  E_taillabel = NULL;
  E_xlabel = NULL;
  N_height = agfindnodeattr(auxg, "height");
  N_width = agfindnodeattr(auxg, "width");
  N_shape = agfindnodeattr(auxg, "shape");
  N_style = NULL;
  N_fontsize = agfindnodeattr(auxg, "fontsize");
  N_fontname = agfindnodeattr(auxg, "fontname");
  N_fontcolor = NULL;
  N_label = agfindnodeattr(auxg, "label");
  N_xlabel = NULL;
  N_showboxes = NULL;
  N_ordering = agfindnodeattr(auxg, "ordering");
  N_sides = agfindnodeattr(auxg, "sides");
  N_peripheries = agfindnodeattr(auxg, "peripheries");
  N_skew = agfindnodeattr(auxg, "skew");
  N_orientation = agfindnodeattr(auxg, "orientation");
  N_distortion = agfindnodeattr(auxg, "distortion");
  N_fixed = agfindnodeattr(auxg, "fixed");
  N_nojustify = NULL;
  N_group = NULL;
  G_ordering = agfindgraphattr(auxg, "ordering");
}

/* Create clone graph. It stores the global Agsyms, to be
 * restored in cleanupCloneGraph. The graph uses the main
 * graph's settings for certain geometry parameters, and
 * declares all node and edge attributes used in the original
 * graph.
 */
static graph_t *cloneGraph(graph_t *g, attr_state_t *attr_state) {
  Agsym_t *sym;
  graph_t *auxg;
  if (agisdirected(g))
    auxg = agopen("auxg", Agdirected, NULL);
  else
    auxg = agopen("auxg", Agundirected, NULL);
  agbindrec(auxg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
  agattr_text(auxg, AGRAPH, "rank", "");
  GD_drawing(auxg) = gv_alloc(sizeof(layout_t));
  GD_drawing(auxg)->quantum = GD_drawing(g)->quantum;
  GD_drawing(auxg)->dpi = GD_drawing(g)->dpi;

  GD_charset(auxg) = GD_charset(g);
  if (GD_flip(g))
    SET_RANKDIR(auxg, RANKDIR_TB);
  else
    SET_RANKDIR(auxg, RANKDIR_LR);
  GD_nodesep(auxg) = GD_nodesep(g);
  GD_ranksep(auxg) = GD_ranksep(g);

  // copy node attrs to auxg
  sym = agnxtattr(agroot(g), AGNODE, NULL); // get the first attr.
  for (; sym; sym = agnxtattr(agroot(g), AGNODE, sym)) {
    const bool is_html = aghtmlstr(sym->defval);
    is_html ? agattr_html(auxg, AGNODE, sym->name, sym->defval)
            : agattr_text(auxg, AGNODE, sym->name, sym->defval);
  }

  // copy edge attributes
  sym = agnxtattr(agroot(g), AGEDGE, NULL); // get the first attr.
  for (; sym; sym = agnxtattr(agroot(g), AGEDGE, sym)) {
    const bool is_html = aghtmlstr(sym->defval);
    is_html ? agattr_html(auxg, AGEDGE, sym->name, sym->defval)
            : agattr_text(auxg, AGEDGE, sym->name, sym->defval);
  }

  if (!agattr_text(auxg, AGEDGE, "headport", NULL))
    agattr_text(auxg, AGEDGE, "headport", "");
  if (!agattr_text(auxg, AGEDGE, "tailport", NULL))
    agattr_text(auxg, AGEDGE, "tailport", "");

  setState(auxg, attr_state);

  return auxg;
}

static void cleanupCloneGraph(graph_t *g, attr_state_t *attr_state) {
  /* restore main graph syms */
  E_constr = attr_state->E_constr;
  E_dir = attr_state->E_dir;
  E_samehead = attr_state->E_samehead;
  E_sametail = attr_state->E_sametail;
  E_weight = attr_state->E_weight;
  E_minlen = attr_state->E_minlen;
  E_fontcolor = attr_state->E_fontcolor;
  E_fontname = attr_state->E_fontname;
  E_fontsize = attr_state->E_fontsize;
  E_headclip = attr_state->E_headclip;
  E_headlabel = attr_state->E_headlabel;
  E_label = attr_state->E_label;
  E_label_float = attr_state->E_label_float;
  E_labelfontcolor = attr_state->E_labelfontcolor;
  E_labelfontname = attr_state->E_labelfontname;
  E_labelfontsize = attr_state->E_labelfontsize;
  E_tailclip = attr_state->E_tailclip;
  E_taillabel = attr_state->E_taillabel;
  E_xlabel = attr_state->E_xlabel;
  N_height = attr_state->N_height;
  N_width = attr_state->N_width;
  N_shape = attr_state->N_shape;
  N_style = attr_state->N_style;
  N_fontsize = attr_state->N_fontsize;
  N_fontname = attr_state->N_fontname;
  N_fontcolor = attr_state->N_fontcolor;
  N_label = attr_state->N_label;
  N_xlabel = attr_state->N_xlabel;
  N_showboxes = attr_state->N_showboxes;
  N_ordering = attr_state->N_ordering;
  N_sides = attr_state->N_sides;
  N_peripheries = attr_state->N_peripheries;
  N_skew = attr_state->N_skew;
  N_orientation = attr_state->N_orientation;
  N_distortion = attr_state->N_distortion;
  N_fixed = attr_state->N_fixed;
  N_nojustify = attr_state->N_nojustify;
  N_group = attr_state->N_group;
  G_ordering = attr_state->G_ordering;
  State = attr_state->State;

  dot_cleanup(g);
  agclose(g);
}

/* If original graph has rankdir=LR or RL, records change shape,
 * so we wrap a record node's label in "{...}" to prevent this.
 */
static node_t *cloneNode(graph_t *g, node_t *orign) {
  node_t *n = agnode(g, agnameof(orign), 1);
  agbindrec(n, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);
  agcopyattr(orign, n);
  if (shapeOf(orign) == SH_RECORD) {
    agxbuf buf = {0};
    agxbprint(&buf, "{%s}", ND_label(orign)->text);
    agset(n, "label", agxbuse(&buf));
    agxbfree(&buf);
  }

  return n;
}

static edge_t *cloneEdge(graph_t *g, node_t *tn, node_t *hn, edge_t *orig) {
  edge_t *e = agedge(g, tn, hn, NULL, 1);
  agbindrec(e, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);
  agcopyattr(orig, e);

  return e;
}

/// rotate, if necessary, then translate points
static pointf transformf(pointf p, pointf del, int flip) {
  if (flip) {
    double i = p.x;
    p.x = p.y;
    p.y = -i;
  }
  return add_pointf(p, del);
}

/* lexicographically order edges by
 *  - has label
 *  - label is wider
 *  - label is higher
 */
static int edgelblcmpfn(const void *x, const void *y) {
  edge_t *const *const ptr0 = x;
  edge_t *const *const ptr1 = y;
  pointf sz0, sz1;

  edge_t *e0 = *ptr0;
  edge_t *e1 = *ptr1;

  if (ED_label(e0)) {
    if (ED_label(e1)) {
      sz0 = ED_label(e0)->dimen;
      sz1 = ED_label(e1)->dimen;
      if (sz0.x > sz1.x)
        return -1;
      if (sz0.x < sz1.x)
        return 1;
      if (sz0.y > sz1.y)
        return -1;
      if (sz0.y < sz1.y)
        return 1;
      return 0;
    }
    return -1;
  }
  if (ED_label(e1)) {
    return 1;
  }
  return 0;
}

#define LBL_SPACE 6 /* space between labels, in points */

/* This handles the second simplest case for flat edges between
 * two adjacent nodes. We still invoke a dot on a rotated problem
 * to handle edges with ports. This usually works, but fails for
 * records because of their weird nature.
 */
static void makeSimpleFlatLabels(node_t *tn, node_t *hn, edge_t **edges,
                                 unsigned cnt, int et, unsigned n_lbls) {
  Ppoly_t poly;
  edge_t *e = *edges;
  pointf points[10], tp, hp;
  double leftend, rightend, ctrx, ctry, miny, maxy;
  double uminx, umaxx;
  double lminx = 0.0, lmaxx = 0.0;

  edge_t **earray = gv_calloc(cnt, sizeof(edge_t *));

  for (unsigned i = 0; i < cnt; i++) {
    earray[i] = edges[i];
  }

  qsort(earray, cnt, sizeof(edge_t *), edgelblcmpfn);

  tp = add_pointf(ND_coord(tn), ED_tail_port(e).p);
  hp = add_pointf(ND_coord(hn), ED_head_port(e).p);

  leftend = tp.x + ND_rw(tn);
  rightend = hp.x - ND_lw(hn);
  ctrx = (leftend + rightend) / 2.0;

  /* do first edge */
  e = earray[0];
  size_t pointn = 0;
  points[pointn++] = tp;
  points[pointn++] = tp;
  points[pointn++] = hp;
  points[pointn++] = hp;
  clip_and_install(e, aghead(e), points, pointn, &sinfo);
  ED_label(e)->pos.x = ctrx;
  ED_label(e)->pos.y = tp.y + (ED_label(e)->dimen.y + LBL_SPACE) / 2.0;
  ED_label(e)->set = true;

  miny = tp.y + LBL_SPACE / 2.0;
  maxy = miny + ED_label(e)->dimen.y;
  uminx = ctrx - ED_label(e)->dimen.x / 2.0;
  umaxx = ctrx + ED_label(e)->dimen.x / 2.0;

  unsigned i;
  for (i = 1; i < n_lbls; i++) {
    e = earray[i];
    if (i % 2) { /* down */
      if (i == 1) {
        lminx = ctrx - ED_label(e)->dimen.x / 2.0;
        lmaxx = ctrx + ED_label(e)->dimen.x / 2.0;
      }
      miny -= LBL_SPACE + ED_label(e)->dimen.y;
      points[0] = tp;
      points[1] = (pointf){.x = tp.x, .y = miny - LBL_SPACE};
      points[2] = (pointf){.x = hp.x, .y = points[1].y};
      points[3] = hp;
      points[4] = (pointf){.x = lmaxx, .y = hp.y};
      points[5] = (pointf){.x = lmaxx, .y = miny};
      points[6] = (pointf){.x = lminx, .y = miny};
      points[7] = (pointf){.x = lminx, .y = tp.y};
      ctry = miny + ED_label(e)->dimen.y / 2.0;
    } else { /* up */
      points[0] = tp;
      points[1] = (pointf){.x = uminx, .y = tp.y};
      points[2] = (pointf){.x = uminx, .y = maxy};
      points[3] = (pointf){.x = umaxx, .y = maxy};
      points[4] = (pointf){.x = umaxx, .y = hp.y};
      points[5] = hp;
      points[6] = (pointf){.x = hp.x, .y = maxy + LBL_SPACE};
      points[7] = (pointf){.x = tp.x, .y = maxy + LBL_SPACE};
      ctry = maxy + ED_label(e)->dimen.y / 2.0 + LBL_SPACE;
      maxy += ED_label(e)->dimen.y + LBL_SPACE;
    }
    poly.pn = 8;
    poly.ps = (Ppoint_t *)points;
    size_t pn;
    pointf *ps = simpleSplineRoute(tp, hp, poly, &pn, et == EDGETYPE_PLINE);
    if (ps == NULL || pn == 0) {
      free(ps);
      free(earray);
      return;
    }
    ED_label(e)->pos.x = ctrx;
    ED_label(e)->pos.y = ctry;
    ED_label(e)->set = true;
    clip_and_install(e, aghead(e), ps, pn, &sinfo);
    free(ps);
  }

  /* edges with no labels */
  for (; i < cnt; i++) {
    e = earray[i];
    if (i % 2) { /* down */
      if (i == 1) {
        lminx = (2 * leftend + rightend) / 3.0;
        lmaxx = (leftend + 2 * rightend) / 3.0;
      }
      miny -= LBL_SPACE;
      points[0] = tp;
      points[1] = (pointf){.x = tp.x, .y = miny - LBL_SPACE};
      points[2] = (pointf){.x = hp.x, .y = points[1].y};
      points[3] = hp;
      points[4] = (pointf){.x = lmaxx, .y = hp.y};
      points[5] = (pointf){.x = lmaxx, .y = miny};
      points[6] = (pointf){.x = lminx, .y = miny};
      points[7] = (pointf){.x = lminx, .y = tp.y};
    } else { /* up */
      points[0] = tp;
      points[1] = (pointf){.x = uminx, .y = tp.y};
      points[2] = (pointf){.x = uminx, .y = maxy};
      points[3] = (pointf){.x = umaxx, .y = maxy};
      points[4] = (pointf){.x = umaxx, .y = hp.y};
      points[5] = hp;
      points[6] = (pointf){.x = hp.x, .y = maxy + LBL_SPACE};
      points[7] = (pointf){.x = tp.x, .y = maxy + LBL_SPACE};
      maxy += +LBL_SPACE;
    }
    poly.pn = 8;
    poly.ps = (Ppoint_t *)points;
    size_t pn;
    pointf *ps = simpleSplineRoute(tp, hp, poly, &pn, et == EDGETYPE_PLINE);
    if (ps == NULL || pn == 0) {
      free(ps);
      free(earray);
      return;
    }
    clip_and_install(e, aghead(e), ps, pn, &sinfo);
    free(ps);
  }

  free(earray);
}

static void makeSimpleFlat(node_t *tn, node_t *hn, edge_t **edges, unsigned cnt,
                           int et) {
  edge_t *e = *edges;
  pointf points[10], tp, hp;
  double stepy, dy;

  tp = add_pointf(ND_coord(tn), ED_tail_port(e).p);
  hp = add_pointf(ND_coord(hn), ED_head_port(e).p);

  stepy = cnt > 1 ? ND_ht(tn) / (double)(cnt - 1) : 0.;
  dy = tp.y - (cnt > 1 ? ND_ht(tn) / 2. : 0.);

  for (unsigned i = 0; i < cnt; i++) {
    e = edges[i];
    size_t pointn = 0;
    if (et == EDGETYPE_SPLINE || et == EDGETYPE_LINE) {
      points[pointn++] = tp;
      points[pointn++] = (pointf){(2 * tp.x + hp.x) / 3, dy};
      points[pointn++] = (pointf){(2 * hp.x + tp.x) / 3, dy};
      points[pointn++] = hp;
    } else { /* EDGETYPE_PLINE */
      points[pointn++] = tp;
      points[pointn++] = tp;
      points[pointn++] = (pointf){(2 * tp.x + hp.x) / 3, dy};
      points[pointn++] = (pointf){(2 * tp.x + hp.x) / 3, dy};
      points[pointn++] = (pointf){(2 * tp.x + hp.x) / 3, dy};
      points[pointn++] = (pointf){(2 * hp.x + tp.x) / 3, dy};
      points[pointn++] = (pointf){(2 * hp.x + tp.x) / 3, dy};
      points[pointn++] = (pointf){(2 * hp.x + tp.x) / 3, dy};
      points[pointn++] = hp;
      points[pointn++] = hp;
    }
    dy += stepy;
    clip_and_install(e, aghead(e), points, pointn, &sinfo);
  }
}

/* In the simple case, with no labels or ports, this creates a simple
 * spindle of splines.
 * If there are only labels, cobble something together.
 * Otherwise, we run dot recursively on the 2 nodes and the edges,
 * essentially using rankdir=LR, to get the needed spline info.
 * This is probably to cute and fragile, and should be rewritten in a
 * more straightforward and laborious fashion.
 *
 * @return 0 on success
 */
static int make_flat_adj_edges(graph_t *g, edge_t **edges, unsigned cnt,
                               edge_t *e0, int et) {
  node_t *n;
  node_t *tn, *hn;
  edge_t *e;
  graph_t *auxg;
  graph_t *subg;
  node_t *auxt, *auxh;
  edge_t *auxe;
  double midx, midy, leftx, rightx;
  pointf del;
  edge_t *hvye = NULL;
  static atomic_flag warned;

  tn = agtail(e0), hn = aghead(e0);
  if (shapeOf(tn) == SH_RECORD || shapeOf(hn) == SH_RECORD) {
    if (!atomic_flag_test_and_set(&warned)) {
      agwarningf("flat edge between adjacent nodes one of which has a record "
                 "shape - replace records with HTML-like labels\n");
      agerr(AGPREV, "  Edge %s %s %s\n", agnameof(tn),
            agisdirected(g) ? "->" : "--", agnameof(hn));
    }
    return 0;
  }
  unsigned labels = 0;
  bool ports = false;
  for (unsigned i = 0; i < cnt; i++) {
    e = edges[i];
    if (ED_label(e))
      labels++;
    if (ED_tail_port(e).defined || ED_head_port(e).defined)
      ports = true;
  }

  if (!ports) {
    /* flat edges without ports and labels can go straight left to right */
    if (labels == 0) {
      makeSimpleFlat(tn, hn, edges, cnt, et);
    }
    /* flat edges without ports but with labels take more work */
    else {
      makeSimpleFlatLabels(tn, hn, edges, cnt, et, labels);
    }
    return 0;
  }

  attr_state_t attrs = {0};
  auxg = cloneGraph(g, &attrs);
  subg = agsubg(auxg, "xxx", 1);
  agbindrec(subg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
  agset(subg, "rank", "source");
  rightx = ND_coord(hn).x;
  leftx = ND_coord(tn).x;
  if (GD_flip(g)) {
    SWAP(&tn, &hn);
  }
  auxt = cloneNode(subg, tn);
  auxh = cloneNode(auxg, hn);
  for (unsigned i = 0; i < cnt; i++) {
    e = edges[i];
    for (; ED_edge_type(e) != NORMAL; e = ED_to_orig(e))
      ;
    if (agtail(e) == tn)
      auxe = cloneEdge(auxg, auxt, auxh, e);
    else
      auxe = cloneEdge(auxg, auxh, auxt, e);
    ED_alg(e) = auxe;
    if (!hvye && !ED_tail_port(e).defined && !ED_head_port(e).defined) {
      hvye = auxe;
      ED_alg(hvye) = e;
    }
  }
  if (!hvye) {
    hvye = agedge(auxg, auxt, auxh, NULL, 1);
  }
  agxset(hvye, E_weight, "10000");
  GD_gvc(auxg) = GD_gvc(g);
  GD_dotroot(auxg) = auxg;
  setEdgeType(auxg, et);
  dot_init_node_edge(auxg);

  dot_rank(auxg);
  const int r = dot_mincross(auxg);
  if (r != 0) {
    return r;
  }
  dot_position(auxg);

  /* reposition */
  midx = (ND_coord(tn).x - ND_rw(tn) + ND_coord(hn).x + ND_lw(hn)) / 2;
  midy = (ND_coord(auxt).x + ND_coord(auxh).x) / 2;
  for (n = GD_nlist(auxg); n; n = ND_next(n)) {
    if (n == auxt) {
      ND_coord(n).y = rightx;
      ND_coord(n).x = midy;
    } else if (n == auxh) {
      ND_coord(n).y = leftx;
      ND_coord(n).x = midy;
    } else
      ND_coord(n).y = midx;
  }
  dot_sameports(auxg);
  const int rc = dot_splines_(auxg, 0);
  if (rc != 0) {
    return rc;
  }
  dotneato_postprocess(auxg);

  /* copy splines */
  if (GD_flip(g)) {
    del.x = ND_coord(tn).x - ND_coord(auxt).y;
    del.y = ND_coord(tn).y + ND_coord(auxt).x;
  } else {
    del.x = ND_coord(tn).x - ND_coord(auxt).x;
    del.y = ND_coord(tn).y - ND_coord(auxt).y;
  }
  for (unsigned i = 0; i < cnt; i++) {
    bezier *auxbz;
    bezier *bz;

    e = edges[i];
    for (; ED_edge_type(e) != NORMAL; e = ED_to_orig(e))
      ;
    auxe = ED_alg(e);
    if ((auxe == hvye) & !ED_alg(auxe))
      continue; /* pseudo-edge */
    auxbz = ED_spl(auxe)->list;
    bz = new_spline(e, auxbz->size);
    bz->sflag = auxbz->sflag;
    bz->sp = transformf(auxbz->sp, del, GD_flip(g));
    bz->eflag = auxbz->eflag;
    bz->ep = transformf(auxbz->ep, del, GD_flip(g));
    for (size_t j = 0; j < auxbz->size;) {
      pointf cp[4];
      cp[0] = bz->list[j] = transformf(auxbz->list[j], del, GD_flip(g));
      j++;
      if (j >= auxbz->size)
        break;
      cp[1] = bz->list[j] = transformf(auxbz->list[j], del, GD_flip(g));
      j++;
      cp[2] = bz->list[j] = transformf(auxbz->list[j], del, GD_flip(g));
      j++;
      cp[3] = transformf(auxbz->list[j], del, GD_flip(g));
      update_bb_bz(&GD_bb(g), cp);
    }
    if (ED_label(e)) {
      ED_label(e)->pos = transformf(ED_label(auxe)->pos, del, GD_flip(g));
      ED_label(e)->set = true;
      updateBB(g, ED_label(e));
    }
  }

  cleanupCloneGraph(auxg, &attrs);
  return 0;
}

static void makeFlatEnd(graph_t *g, const spline_info_t sp, path *P, node_t *n,
                        edge_t *e, pathend_t *endp, bool isBegin) {
  boxf b = endp->nb = maximal_bbox(g, sp, n, NULL, e);
  endp->sidemask = TOP;
  if (isBegin)
    beginpath(P, e, FLATEDGE, endp, false);
  else
    endpath(P, e, FLATEDGE, endp, false);
  b.UR.y = endp->boxes[endp->boxn - 1].UR.y;
  b.LL.y = endp->boxes[endp->boxn - 1].LL.y;
  b = makeregularend(b, TOP, ND_coord(n).y + GD_rank(g)[ND_rank(n)].ht2);
  if (b.LL.x < b.UR.x && b.LL.y < b.UR.y)
    endp->boxes[endp->boxn++] = b;
}

static void makeBottomFlatEnd(graph_t *g, const spline_info_t sp, path *P,
                              node_t *n, edge_t *e, pathend_t *endp,
                              bool isBegin) {
  boxf b = endp->nb = maximal_bbox(g, sp, n, NULL, e);
  endp->sidemask = BOTTOM;
  if (isBegin)
    beginpath(P, e, FLATEDGE, endp, false);
  else
    endpath(P, e, FLATEDGE, endp, false);
  b.UR.y = endp->boxes[endp->boxn - 1].UR.y;
  b.LL.y = endp->boxes[endp->boxn - 1].LL.y;
  b = makeregularend(b, BOTTOM, ND_coord(n).y - GD_rank(g)[ND_rank(n)].ht2);
  if (b.LL.x < b.UR.x && b.LL.y < b.UR.y)
    endp->boxes[endp->boxn++] = b;
}

static void make_flat_labeled_edge(graph_t *g, const spline_info_t sp, path *P,
                                   edge_t *e, int et) {
  node_t *tn, *hn, *ln;
  pointf *ps;
  bool ps_needs_free = false;
  pathend_t tend, hend;
  boxf lb;
  int i;
  edge_t *f;
  pointf points[7];

  tn = agtail(e);
  hn = aghead(e);

  for (f = ED_to_virt(e); ED_to_virt(f); f = ED_to_virt(f))
    ;
  ln = agtail(f);
  ED_label(e)->pos = ND_coord(ln);
  ED_label(e)->set = true;

  size_t pn;
  if (et == EDGETYPE_LINE) {
    pointf startp, endp, lp;

    startp = add_pointf(ND_coord(tn), ED_tail_port(e).p);
    endp = add_pointf(ND_coord(hn), ED_head_port(e).p);

    lp = ED_label(e)->pos;
    lp.y -= ED_label(e)->dimen.y / 2.0;
    points[1] = points[0] = startp;
    points[2] = points[3] = points[4] = lp;
    points[5] = points[6] = endp;
    ps = points;
    pn = 7;
  } else {
    lb.LL.x = ND_coord(ln).x - ND_lw(ln);
    lb.UR.x = ND_coord(ln).x + ND_rw(ln);
    lb.UR.y = ND_coord(ln).y + ND_ht(ln) / 2;
    double ydelta = ND_coord(ln).y - GD_rank(g)[ND_rank(tn)].ht1 -
                    ND_coord(tn).y + GD_rank(g)[ND_rank(tn)].ht2;
    ydelta /= 6;
    lb.LL.y = lb.UR.y - MAX(5, ydelta);

    makeFlatEnd(g, sp, P, tn, e, &tend, true);
    makeFlatEnd(g, sp, P, hn, e, &hend, false);

    boxf boxes[] = {
        {
            .LL =
                {
                    .x = tend.boxes[tend.boxn - 1].LL.x,
                    .y = tend.boxes[tend.boxn - 1].UR.y,
                },
            .UR = lb.LL,
        },
        {
            .LL =
                {
                    .x = tend.boxes[tend.boxn - 1].LL.x,
                    .y = lb.LL.y,
                },
            .UR =
                {
                    .x = hend.boxes[hend.boxn - 1].UR.x,
                    .y = lb.UR.y,
                },
        },
        {
            .LL =
                {
                    .x = lb.UR.x,
                    .y = hend.boxes[hend.boxn - 1].UR.y,
                },
            .UR =
                {
                    .x = hend.boxes[hend.boxn - 1].UR.x,
                    .y = lb.LL.y,
                },
        },
    };
    const size_t boxn = sizeof(boxes) / sizeof(boxes[0]);

    for (i = 0; i < tend.boxn; i++)
      add_box(P, tend.boxes[i]);
    for (size_t j = 0; j < boxn; j++)
      add_box(P, boxes[j]);
    for (i = hend.boxn - 1; i >= 0; i--)
      add_box(P, hend.boxes[i]);

    ps_needs_free = true;
    if (et == EDGETYPE_SPLINE)
      ps = routesplines(P, &pn);
    else
      ps = routepolylines(P, &pn);
    if (pn == 0) {
      free(ps);
      return;
    }
  }
  clip_and_install(e, aghead(e), ps, pn, &sinfo);
  if (ps_needs_free)
    free(ps);
}

static void make_flat_bottom_edges(graph_t *g, const spline_info_t sp, path *P,
                                   edge_t **edges, unsigned cnt, edge_t *e,
                                   bool use_splines) {
  node_t *tn, *hn;
  int j, r;
  double stepx, stepy, vspace;
  rank_t *nextr;
  pathend_t tend, hend;

  tn = agtail(e);
  hn = aghead(e);
  r = ND_rank(tn);
  if (r < GD_maxrank(g)) {
    nextr = GD_rank(g) + (r + 1);
    vspace = ND_coord(tn).y - GD_rank(g)[r].pht1 -
             (ND_coord(nextr->v[0]).y + nextr->pht2);
  } else {
    vspace = GD_ranksep(g);
  }
  stepx = sp.Multisep / (cnt + 1);
  stepy = vspace / (cnt + 1);

  makeBottomFlatEnd(g, sp, P, tn, e, &tend, true);
  makeBottomFlatEnd(g, sp, P, hn, e, &hend, false);

  for (unsigned i = 0; i < cnt; i++) {
    boxf b;
    e = edges[i];
    size_t boxn = 0;

    boxf boxes[3];

    b = tend.boxes[tend.boxn - 1];
    boxes[boxn].LL.x = b.LL.x;
    boxes[boxn].UR.y = b.LL.y;
    boxes[boxn].UR.x = b.UR.x + (i + 1) * stepx;
    boxes[boxn].LL.y = b.LL.y - (i + 1) * stepy;
    boxn++;
    boxes[boxn].LL.x = tend.boxes[tend.boxn - 1].LL.x;
    boxes[boxn].UR.y = boxes[boxn - 1].LL.y;
    boxes[boxn].UR.x = hend.boxes[hend.boxn - 1].UR.x;
    boxes[boxn].LL.y = boxes[boxn].UR.y - stepy;
    boxn++;
    b = hend.boxes[hend.boxn - 1];
    boxes[boxn].UR.x = b.UR.x;
    boxes[boxn].UR.y = b.LL.y;
    boxes[boxn].LL.x = b.LL.x - (i + 1) * stepx;
    boxes[boxn].LL.y = boxes[boxn - 1].UR.y;
    boxn++;
    assert(boxn == sizeof(boxes) / sizeof(boxes[0]));

    for (j = 0; j < tend.boxn; j++)
      add_box(P, tend.boxes[j]);
    for (size_t k = 0; k < boxn; k++)
      add_box(P, boxes[k]);
    for (j = hend.boxn - 1; j >= 0; j--)
      add_box(P, hend.boxes[j]);

    pointf *ps = NULL;
    size_t pn = 0;
    if (use_splines)
      ps = routesplines(P, &pn);
    else
      ps = routepolylines(P, &pn);
    if (pn == 0) {
      free(ps);
      return;
    }
    clip_and_install(e, aghead(e), ps, pn, &sinfo);
    free(ps);
    P->nbox = 0;
  }
}

/* Construct flat edges edges[ind...ind+cnt-1]
 * There are 4 main cases:
 *  - all edges between a and b where a and b are adjacent
 *  - one labeled edge
 *  - all non-labeled edges with identical ports between non-adjacent a and b
 *     = connecting bottom to bottom/left/right - route along bottom
 *     = the rest - route along top
 *
 * @return 0 on success
 */
static int make_flat_edge(graph_t *g, const spline_info_t sp, path *P,
                          edge_t **edges, unsigned cnt, int et) {
  Agedgeinfo_t fwdedgei;
  Agedgepair_t fwdedge;
  int j;
  double stepx, stepy, vspace;
  pathend_t tend, hend;

  fwdedge.out.base.data = &fwdedgei.hdr;

  /* Get sample edge; normalize to go from left to right */
  edge_t *e = *edges;
  bool isAdjacent = ED_adjacent(e) != 0;
  if (ED_tree_index(e) & BWDEDGE) {
    makefwdedge(&fwdedge.out, e);
    e = &fwdedge.out;
  }
  for (unsigned i = 1; i < cnt; i++) {
    if (ED_adjacent(edges[i])) {
      isAdjacent = true;
      break;
    }
  }
  // The lead edge edges[0] might not have been marked earlier as adjacent, so
  // check them all.
  if (isAdjacent) {
    return make_flat_adj_edges(g, edges, cnt, e, et);
  }
  if (ED_label(e)) { /* edges with labels aren't multi-edges */
    make_flat_labeled_edge(g, sp, P, e, et);
    return 0;
  }

  if (et == EDGETYPE_LINE) {
    makeSimpleFlat(agtail(e), aghead(e), edges, cnt, et);
    return 0;
  }

  const int tside = ED_tail_port(e).side;
  const int hside = ED_head_port(e).side;
  if ((tside == BOTTOM && hside != TOP) || (hside == BOTTOM && tside != TOP)) {
    make_flat_bottom_edges(g, sp, P, edges, cnt, e, et == EDGETYPE_SPLINE);
    return 0;
  }

  node_t *tn = agtail(e);
  node_t *hn = aghead(e);
  const int r = ND_rank(tn);
  if (r > 0) {
    rank_t *prevr;
    if (GD_has_labels(g->root) & EDGE_LABEL)
      prevr = GD_rank(g) + (r - 2);
    else
      prevr = GD_rank(g) + (r - 1);
    vspace = ND_coord(prevr->v[0]).y - prevr->ht1 - ND_coord(tn).y -
             GD_rank(g)[r].ht2;
  } else {
    vspace = GD_ranksep(g);
  }
  stepx = sp.Multisep / (cnt + 1);
  stepy = vspace / (cnt + 1);

  makeFlatEnd(g, sp, P, tn, e, &tend, true);
  makeFlatEnd(g, sp, P, hn, e, &hend, false);

  for (unsigned i = 0; i < cnt; i++) {
    boxf b;
    e = edges[i];
    size_t boxn = 0;

    boxf boxes[3];

    b = tend.boxes[tend.boxn - 1];
    boxes[boxn].LL.x = b.LL.x;
    boxes[boxn].LL.y = b.UR.y;
    boxes[boxn].UR.x = b.UR.x + (i + 1) * stepx;
    boxes[boxn].UR.y = b.UR.y + (i + 1) * stepy;
    boxn++;
    boxes[boxn].LL.x = tend.boxes[tend.boxn - 1].LL.x;
    boxes[boxn].LL.y = boxes[boxn - 1].UR.y;
    boxes[boxn].UR.x = hend.boxes[hend.boxn - 1].UR.x;
    boxes[boxn].UR.y = boxes[boxn].LL.y + stepy;
    boxn++;
    b = hend.boxes[hend.boxn - 1];
    boxes[boxn].UR.x = b.UR.x;
    boxes[boxn].LL.y = b.UR.y;
    boxes[boxn].LL.x = b.LL.x - (i + 1) * stepx;
    boxes[boxn].UR.y = boxes[boxn - 1].LL.y;
    boxn++;
    assert(boxn == sizeof(boxes) / sizeof(boxes[0]));

    for (j = 0; j < tend.boxn; j++)
      add_box(P, tend.boxes[j]);
    for (size_t k = 0; k < boxn; k++)
      add_box(P, boxes[k]);
    for (j = hend.boxn - 1; j >= 0; j--)
      add_box(P, hend.boxes[j]);

    pointf *ps = NULL;
    size_t pn = 0;
    if (et == EDGETYPE_SPLINE)
      ps = routesplines(P, &pn);
    else
      ps = routepolylines(P, &pn);
    if (pn == 0) {
      free(ps);
      return 0;
    }
    clip_and_install(e, aghead(e), ps, pn, &sinfo);
    free(ps);
    P->nbox = 0;
  }
  return 0;
}

/// Return true if p3 is to left of ray p1->p2
static bool leftOf(pointf p1, pointf p2, pointf p3) {
  return (p1.y - p2.y) * (p3.x - p2.x) - (p3.y - p2.y) * (p1.x - p2.x) > 0;
}

/* Create an edge as line segment. We guarantee that the points
 * are always drawn downwards. This means that for flipped edges,
 * we draw from the head to the tail. The routine returns the
 * end node of the edge in *hp. The points are stored in the
 * given array of points, and the number of points is returned.
 *
 * If the edge has a label, the edge is draw as two segments, with
 * the bend near the label.
 *
 * If the endpoints are on adjacent ranks, revert to usual code by
 * returning 0.
 * This is done because the usual code handles the interaction of
 * multiple edges better.
 */
static int makeLineEdge(graph_t *g, edge_t *fe, points_t *points, node_t **hp) {
  int delr, pn;
  node_t *hn;
  node_t *tn;
  edge_t *e = fe;
  pointf startp, endp, lp;
  pointf dimen;
  double width, height;

  while (ED_edge_type(e) != NORMAL)
    e = ED_to_orig(e);
  hn = aghead(e);
  tn = agtail(e);
  delr = abs(ND_rank(hn) - ND_rank(tn));
  if (delr == 1 || (delr == 2 && (GD_has_labels(g->root) & EDGE_LABEL)))
    return 0;
  if (agtail(fe) == agtail(e)) {
    *hp = hn;
    startp = add_pointf(ND_coord(tn), ED_tail_port(e).p);
    endp = add_pointf(ND_coord(hn), ED_head_port(e).p);
  } else {
    *hp = tn;
    startp = add_pointf(ND_coord(hn), ED_head_port(e).p);
    endp = add_pointf(ND_coord(tn), ED_tail_port(e).p);
  }

  if (ED_label(e)) {
    dimen = ED_label(e)->dimen;
    if (GD_flip(agraphof(hn))) {
      width = dimen.y;
      height = dimen.x;
    } else {
      width = dimen.x;
      height = dimen.y;
    }

    lp = ED_label(e)->pos;
    if (leftOf(endp, startp, lp)) {
      lp.x += width / 2.0;
      lp.y -= height / 2.0;
    } else {
      lp.x -= width / 2.0;
      lp.y += height / 2.0;
    }

    LIST_APPEND(points, startp);
    LIST_APPEND(points, startp);
    LIST_APPEND(points, lp);
    LIST_APPEND(points, lp);
    LIST_APPEND(points, lp);
    LIST_APPEND(points, endp);
    LIST_APPEND(points, endp);
    pn = 7;
  } else {
    LIST_APPEND(points, startp);
    LIST_APPEND(points, startp);
    LIST_APPEND(points, endp);
    LIST_APPEND(points, endp);
    pn = 4;
  }

  return pn;
}

static void make_regular_edge(graph_t *g, spline_info_t *sp, path *P,
                              edge_t **edges, unsigned cnt, int et) {
  node_t *tn, *hn;
  Agedgeinfo_t fwdedgeai, fwdedgebi, fwdedgei;
  Agedgepair_t fwdedgea, fwdedgeb, fwdedge;
  edge_t *e, *fe, *le, *segfirst;
  pathend_t tend, hend;
  boxf b;
  int sl;
  points_t pointfs = {0};
  points_t pointfs2 = {0};

  fwdedgea.out.base.data = &fwdedgeai.hdr;
  fwdedgeb.out.base.data = &fwdedgebi.hdr;
  fwdedge.out.base.data = &fwdedgei.hdr;

  sl = 0;
  e = *edges;
  bool hackflag = false;
  if (abs(ND_rank(agtail(e)) - ND_rank(aghead(e))) > 1) {
    fwdedgeai = *(Agedgeinfo_t *)((uintptr_t)e->base.data -
                                  offsetof(Agedgeinfo_t, hdr));
    fwdedgea.out = *e;
    fwdedgea.in = *AGOUT2IN(e);
    fwdedgea.out.base.data = &fwdedgeai.hdr;
    if (ED_tree_index(e) & BWDEDGE) {
      makefwdedge(&fwdedgeb.out, e);
      agtail(&fwdedgea.out) = aghead(e);
      ED_tail_port(&fwdedgea.out) = ED_head_port(e);
    } else {
      fwdedgebi = *(Agedgeinfo_t *)((uintptr_t)e->base.data -
                                    offsetof(Agedgeinfo_t, hdr));
      fwdedgeb.out = *e;
      fwdedgeb.out.base.data = &fwdedgebi.hdr;
      agtail(&fwdedgea.out) = agtail(e);
      fwdedgeb.in = *AGOUT2IN(e);
    }
    le = getmainedge(e);
    while (ED_to_virt(le))
      le = ED_to_virt(le);
    aghead(&fwdedgea.out) = aghead(le);
    ED_head_port(&fwdedgea.out).defined = false;
    ED_edge_type(&fwdedgea.out) = VIRTUAL;
    ED_head_port(&fwdedgea.out).p.x = ED_head_port(&fwdedgea.out).p.y = 0;
    ED_to_orig(&fwdedgea.out) = e;
    e = &fwdedgea.out;
    hackflag = true;
  } else {
    if (ED_tree_index(e) & BWDEDGE) {
      makefwdedge(&fwdedgea.out, e);
      e = &fwdedgea.out;
    }
  }
  fe = e;

  /* compute the spline points for the edge */

  if (et == EDGETYPE_LINE && makeLineEdge(g, fe, &pointfs, &hn)) {
  } else {
    bool is_spline = et == EDGETYPE_SPLINE;
    boxes_t boxes = {0};
    segfirst = e;
    tn = agtail(e);
    hn = aghead(e);
    b = tend.nb = maximal_bbox(g, *sp, tn, NULL, e);
    beginpath(P, e, REGULAREDGE, &tend, spline_merge(tn));
    b.UR.y = tend.boxes[tend.boxn - 1].UR.y;
    b.LL.y = tend.boxes[tend.boxn - 1].LL.y;
    b = makeregularend(b, BOTTOM, ND_coord(tn).y - GD_rank(g)[ND_rank(tn)].ht1);
    if (b.LL.x < b.UR.x && b.LL.y < b.UR.y)
      tend.boxes[tend.boxn++] = b;
    bool smode = false;
    bool si = false;
    while (ND_node_type(hn) == VIRTUAL && !sinfo.splineMerge(hn)) {
      LIST_APPEND(&boxes, rank_box(sp, g, ND_rank(tn)));
      if (!smode && ((sl = straight_len(hn)) >=
                     ((GD_has_labels(g->root) & EDGE_LABEL) ? 4 + 1 : 2 + 1))) {
        smode = true;
        si = true;
        sl -= 2;
      }
      if (!smode || si) {
        si = false;
        LIST_APPEND(&boxes, maximal_bbox(g, *sp, hn, e, ND_out(hn).list[0]));
        e = ND_out(hn).list[0];
        tn = agtail(e);
        hn = aghead(e);
        continue;
      }
      hend.nb = maximal_bbox(g, *sp, hn, e, ND_out(hn).list[0]);
      endpath(P, e, REGULAREDGE, &hend, spline_merge(aghead(e)));
      b = makeregularend(hend.boxes[hend.boxn - 1], TOP,
                         ND_coord(hn).y + GD_rank(g)[ND_rank(hn)].ht2);
      if (b.LL.x < b.UR.x && b.LL.y < b.UR.y)
        hend.boxes[hend.boxn++] = b;
      P->end.theta = M_PI / 2, P->end.constrained = true;
      completeregularpath(P, segfirst, e, &tend, &hend, &boxes);
      pointf *ps = NULL;
      size_t pn = 0;
      if (is_spline)
        ps = routesplines(P, &pn);
      else {
        ps = routepolylines(P, &pn);
        if (et == EDGETYPE_LINE && pn > 4) {
          ps[1] = ps[0];
          ps[3] = ps[2] = ps[pn - 1];
          pn = 4;
        }
      }
      if (pn == 0) {
        free(ps);
        LIST_FREE(&boxes);
        LIST_FREE(&pointfs);
        LIST_FREE(&pointfs2);
        return;
      }

      for (size_t i = 0; i < pn; i++) {
        LIST_APPEND(&pointfs, ps[i]);
      }
      free(ps);
      e = straight_path(ND_out(hn).list[0], sl, &pointfs);
      recover_slack(segfirst, P);
      segfirst = e;
      tn = agtail(e);
      hn = aghead(e);
      LIST_CLEAR(&boxes);
      tend.nb = maximal_bbox(g, *sp, tn, ND_in(tn).list[0], e);
      beginpath(P, e, REGULAREDGE, &tend, spline_merge(tn));
      b = makeregularend(tend.boxes[tend.boxn - 1], BOTTOM,
                         ND_coord(tn).y - GD_rank(g)[ND_rank(tn)].ht1);
      if (b.LL.x < b.UR.x && b.LL.y < b.UR.y)
        tend.boxes[tend.boxn++] = b;
      P->start.theta = -M_PI / 2, P->start.constrained = true;
      smode = false;
    }
    LIST_APPEND(&boxes, rank_box(sp, g, ND_rank(tn)));
    b = hend.nb = maximal_bbox(g, *sp, hn, e, NULL);
    endpath(P, hackflag ? &fwdedgeb.out : e, REGULAREDGE, &hend,
            spline_merge(aghead(e)));
    b.UR.y = hend.boxes[hend.boxn - 1].UR.y;
    b.LL.y = hend.boxes[hend.boxn - 1].LL.y;
    b = makeregularend(b, TOP, ND_coord(hn).y + GD_rank(g)[ND_rank(hn)].ht2);
    if (b.LL.x < b.UR.x && b.LL.y < b.UR.y)
      hend.boxes[hend.boxn++] = b;
    completeregularpath(P, segfirst, e, &tend, &hend, &boxes);
    LIST_FREE(&boxes);
    pointf *ps = NULL;
    size_t pn = 0;
    if (is_spline)
      ps = routesplines(P, &pn);
    else
      ps = routepolylines(P, &pn);
    if (et == EDGETYPE_LINE && pn > 4) {
      /* Here we have used the polyline case to handle
       * an edge between two nodes on adjacent ranks. If the
       * results really is a polyline, straighten it.
       */
      ps[1] = ps[0];
      ps[3] = ps[2] = ps[pn - 1];
      pn = 4;
    }
    if (pn == 0) {
      free(ps);
      LIST_FREE(&pointfs);
      LIST_FREE(&pointfs2);
      return;
    }
    for (size_t i = 0; i < pn; i++) {
      LIST_APPEND(&pointfs, ps[i]);
    }
    free(ps);
    recover_slack(segfirst, P);
    hn = hackflag ? aghead(&fwdedgeb.out) : aghead(e);
  }

  /* make copies of the spline points, one per multi-edge */

  if (cnt == 1) {
    LIST_SYNC(&pointfs);
    clip_and_install(fe, hn, LIST_FRONT(&pointfs), LIST_SIZE(&pointfs), &sinfo);
    LIST_FREE(&pointfs);
    LIST_FREE(&pointfs2);
    return;
  }
  const double dx = sp->Multisep * (cnt - 1) / 2;
  for (size_t k = 1; k + 1 < LIST_SIZE(&pointfs); k++)
    LIST_AT(&pointfs, k)->x -= dx;

  for (size_t k = 0; k < LIST_SIZE(&pointfs); k++)
    LIST_APPEND(&pointfs2, LIST_GET(&pointfs, k));
  LIST_SYNC(&pointfs2);
  clip_and_install(fe, hn, LIST_FRONT(&pointfs2), LIST_SIZE(&pointfs2), &sinfo);
  for (unsigned j = 1; j < cnt; j++) {
    e = edges[j];
    if (ED_tree_index(e) & BWDEDGE) {
      makefwdedge(&fwdedge.out, e);
      e = &fwdedge.out;
    }
    for (size_t k = 1; k + 1 < LIST_SIZE(&pointfs); k++)
      LIST_AT(&pointfs, k)->x += sp->Multisep;
    LIST_CLEAR(&pointfs2);
    for (size_t k = 0; k < LIST_SIZE(&pointfs); k++)
      LIST_APPEND(&pointfs2, LIST_GET(&pointfs, k));
    LIST_SYNC(&pointfs2);
    clip_and_install(e, aghead(e), LIST_FRONT(&pointfs2), LIST_SIZE(&pointfs2),
                     &sinfo);
  }
  LIST_FREE(&pointfs);
  LIST_FREE(&pointfs2);
}

/* regular edges */

static void completeregularpath(path *P, edge_t *first, edge_t *last,
                                pathend_t *tendp, pathend_t *hendp,
                                const boxes_t *boxes) {
  edge_t *uleft = top_bound(first, -1);
  edge_t *uright = top_bound(first, 1);
  if (uleft) {
    if (getsplinepoints(uleft) == NULL)
      return;
  }
  if (uright) {
    if (getsplinepoints(uright) == NULL)
      return;
  }
  edge_t *lleft = bot_bound(last, -1);
  edge_t *lright = bot_bound(last, 1);
  if (lleft) {
    if (getsplinepoints(lleft) == NULL)
      return;
  }
  if (lright) {
    if (getsplinepoints(lright) == NULL)
      return;
  }
  for (int i = 0; i < tendp->boxn; i++)
    add_box(P, tendp->boxes[i]);
  const size_t fb = P->nbox + 1;
  const size_t lb = fb + LIST_SIZE(boxes) - 3;
  for (size_t i = 0; i < LIST_SIZE(boxes); i++)
    add_box(P, LIST_GET(boxes, i));
  for (int i = hendp->boxn - 1; i >= 0; i--)
    add_box(P, hendp->boxes[i]);
  adjustregularpath(P, fb, lb);
}

/* Add box to fill between node and interrank space. Needed because
 * nodes in a given rank can differ in height.
 * for now, regular edges always go from top to bottom
 */
static boxf makeregularend(boxf b, int side, double y) {
  assert(side == BOTTOM || side == TOP);
  if (side == BOTTOM) {
    return (boxf){{b.LL.x, y}, {b.UR.x, b.LL.y}};
  }
  return (boxf){{b.LL.x, b.UR.y}, {b.UR.x, y}};
}

/* make sure the path is wide enough.
 * the % 2 was so that in rank boxes would only be grown if
 * they were == 0 while inter-rank boxes could be stretched to a min
 * width.
 * The list of boxes has three parts: tail boxes, path boxes, and head
 * boxes. (Note that because of back edges, the tail boxes might actually
 * belong to the head node, and vice versa.) fb is the index of the
 * first interrank path box and lb is the last interrank path box.
 * If fb > lb, there are none.
 *
 * The second for loop was added by ek long ago, and apparently is intended
 * to guarantee an overlap between adjacent boxes of at least MINW.
 * It doesn't do this.
 */
static void adjustregularpath(path *P, size_t fb, size_t lb) {
  boxf *bp1, *bp2;

  for (size_t i = fb - 1; i < lb + 1; i++) {
    bp1 = &P->boxes[i];
    if ((i - fb) % 2 == 0) {
      if (bp1->LL.x >= bp1->UR.x) {
        double x = (bp1->LL.x + bp1->UR.x) / 2;
        bp1->LL.x = x - HALFMINW;
        bp1->UR.x = x + HALFMINW;
      }
    } else {
      if (bp1->LL.x + MINW > bp1->UR.x) {
        double x = (bp1->LL.x + bp1->UR.x) / 2;
        bp1->LL.x = x - HALFMINW;
        bp1->UR.x = x + HALFMINW;
      }
    }
  }
  for (size_t i = 0; i + 1 < P->nbox; i++) {
    bp1 = &P->boxes[i], bp2 = &P->boxes[i + 1];
    if (i >= fb && i <= lb && (i - fb) % 2 == 0) {
      if (bp1->LL.x + MINW > bp2->UR.x)
        bp2->UR.x = bp1->LL.x + MINW;
      if (bp1->UR.x - MINW < bp2->LL.x)
        bp2->LL.x = bp1->UR.x - MINW;
    } else if (i + 1 >= fb && i < lb && (i + 1 - fb) % 2 == 0) {
      if (bp1->LL.x + MINW > bp2->UR.x)
        bp1->LL.x = bp2->UR.x - MINW;
      if (bp1->UR.x - MINW < bp2->LL.x)
        bp1->UR.x = bp2->LL.x + MINW;
    }
  }
}

static boxf rank_box(spline_info_t *sp, graph_t *g, int r) {
  boxf b = sp->Rank_box[r];
  if (b.LL.x == b.UR.x) {
    node_t *const left0 = GD_rank(g)[r].v[0];
    node_t *const left1 = GD_rank(g)[r + 1].v[0];
    b.LL.x = sp->LeftBound;
    b.LL.y = ND_coord(left1).y + GD_rank(g)[r + 1].ht2;
    b.UR.x = sp->RightBound;
    b.UR.y = ND_coord(left0).y - GD_rank(g)[r].ht1;
    sp->Rank_box[r] = b;
  }
  return b;
}

/* returns count of vertically aligned edges starting at n */
static int straight_len(node_t *n) {
  int cnt = 0;
  node_t *v;

  v = n;
  while (1) {
    v = aghead(ND_out(v).list[0]);
    if (ND_node_type(v) != VIRTUAL)
      break;
    if (ND_out(v).size != 1 || ND_in(v).size != 1)
      break;
    if (ND_coord(v).x != ND_coord(n).x)
      break;
    cnt++;
  }
  return cnt;
}

static edge_t *straight_path(edge_t *e, int cnt, points_t *plist) {
  edge_t *f = e;

  while (cnt--)
    f = ND_out(aghead(f)).list[0];
  assert(!LIST_IS_EMPTY(plist));
  LIST_APPEND(plist, LIST_GET(plist, LIST_SIZE(plist) - 1));
  LIST_APPEND(plist, LIST_GET(plist, LIST_SIZE(plist) - 1));

  return f;
}

static void recover_slack(edge_t *e, path *p) {
  node_t *vn;

  size_t b = 0; // skip first rank box
  for (vn = aghead(e); ND_node_type(vn) == VIRTUAL && !sinfo.splineMerge(vn);
       vn = aghead(ND_out(vn).list[0])) {
    while (b < p->nbox && p->boxes[b].LL.y > ND_coord(vn).y)
      b++;
    if (b >= p->nbox)
      break;
    if (p->boxes[b].UR.y < ND_coord(vn).y)
      continue;
    if (ND_label(vn))
      resize_vn(vn, p->boxes[b].LL.x, p->boxes[b].UR.x,
                p->boxes[b].UR.x + ND_rw(vn));
    else
      resize_vn(vn, p->boxes[b].LL.x, (p->boxes[b].LL.x + p->boxes[b].UR.x) / 2,
                p->boxes[b].UR.x);
  }
}

static void resize_vn(node_t *vn, double lx, double cx, double rx) {
  ND_coord(vn).x = cx;
  ND_lw(vn) = cx - lx, ND_rw(vn) = rx - cx;
}

/* side > 0 means right. side < 0 means left */
static edge_t *top_bound(edge_t *e, int side) {
  edge_t *f, *ans = NULL;
  int i;

  for (i = 0; (f = ND_out(agtail(e)).list[i]); i++) {
    if (side * (ND_order(aghead(f)) - ND_order(aghead(e))) <= 0)
      continue;
    if (ED_spl(f) == NULL &&
        (ED_to_orig(f) == NULL || ED_spl(ED_to_orig(f)) == NULL))
      continue;
    if (ans == NULL || side * (ND_order(aghead(ans)) - ND_order(aghead(f))) > 0)
      ans = f;
  }
  return ans;
}

static edge_t *bot_bound(edge_t *e, int side) {
  edge_t *f, *ans = NULL;
  int i;

  for (i = 0; (f = ND_in(aghead(e)).list[i]); i++) {
    if (side * (ND_order(agtail(f)) - ND_order(agtail(e))) <= 0)
      continue;
    if (ED_spl(f) == NULL &&
        (ED_to_orig(f) == NULL || ED_spl(ED_to_orig(f)) == NULL))
      continue;
    if (ans == NULL || side * (ND_order(agtail(ans)) - ND_order(agtail(f))) > 0)
      ans = f;
  }
  return ans;
}

/* common routines */

static bool cl_vninside(graph_t *cl, node_t *n) {
  return BETWEEN(GD_bb(cl).LL.x, ND_coord(n).x, GD_bb(cl).UR.x) &&
         BETWEEN(GD_bb(cl).LL.y, ND_coord(n).y, GD_bb(cl).UR.y);
}

/* All nodes belong to some cluster, which may be the root graph.
 * For the following, we only want a cluster if it is a real cluster
 * It is not clear this will handle all potential problems. It seems one
 * could have hcl and tcl contained in cl, which would also cause problems.
 */
#define REAL_CLUSTER(n) (ND_clust(n) == g ? NULL : ND_clust(n))

/* returns the cluster of (adj) that interferes with n,
 */
static Agraph_t *cl_bound(graph_t *g, node_t *n, node_t *adj) {
  graph_t *rv, *cl, *tcl, *hcl;
  edge_t *orig;

  rv = NULL;
  if (ND_node_type(n) == NORMAL)
    tcl = hcl = ND_clust(n);
  else {
    orig = ED_to_orig(ND_out(n).list[0]);
    tcl = ND_clust(agtail(orig));
    hcl = ND_clust(aghead(orig));
  }
  if (ND_node_type(adj) == NORMAL) {
    cl = REAL_CLUSTER(adj);
    if (cl && cl != tcl && cl != hcl)
      rv = cl;
  } else {
    orig = ED_to_orig(ND_out(adj).list[0]);
    cl = REAL_CLUSTER(agtail(orig));
    if (cl && cl != tcl && cl != hcl && cl_vninside(cl, adj))
      rv = cl;
    else {
      cl = REAL_CLUSTER(aghead(orig));
      if (cl && cl != tcl && cl != hcl && cl_vninside(cl, adj))
        rv = cl;
    }
  }
  return rv;
}

/* Return an initial bounding box to be used for building the
 * beginning or ending of the path of boxes.
 * Height reflects height of tallest node on rank.
 * The extra space provided by FUDGE allows begin/endpath to create a box
 * FUDGE-2 away from the node, so the routing can avoid the node and the
 * box is at least 2 wide.
 */
#define FUDGE 4

static boxf maximal_bbox(graph_t *g, const spline_info_t sp, node_t *vn,
                         edge_t *ie, edge_t *oe) {
  double b, nb;
  graph_t *left_cl, *right_cl;
  node_t *left, *right;
  boxf rv;

  left_cl = right_cl = NULL;

  /* give this node all the available space up to its neighbors */
  b = (double)(ND_coord(vn).x - ND_lw(vn) - FUDGE);
  if ((left = neighbor(g, vn, ie, oe, -1))) {
    if ((left_cl = cl_bound(g, vn, left)))
      nb = GD_bb(left_cl).UR.x + sp.Splinesep;
    else {
      nb = (double)(ND_coord(left).x + ND_mval(left));
      if (ND_node_type(left) == NORMAL)
        nb += GD_nodesep(g) / 2.;
      else
        nb += sp.Splinesep;
    }
    if (nb < b)
      b = nb;
    rv.LL.x = round(b);
  } else
    rv.LL.x = fmin(round(b), sp.LeftBound);

  /* we have to leave room for our own label! */
  if (ND_node_type(vn) == VIRTUAL && ND_label(vn))
    b = (double)(ND_coord(vn).x + 10);
  else
    b = (double)(ND_coord(vn).x + ND_rw(vn) + FUDGE);
  if ((right = neighbor(g, vn, ie, oe, 1))) {
    if ((right_cl = cl_bound(g, vn, right)))
      nb = GD_bb(right_cl).LL.x - sp.Splinesep;
    else {
      nb = ND_coord(right).x - ND_lw(right);
      if (ND_node_type(right) == NORMAL)
        nb -= GD_nodesep(g) / 2.;
      else
        nb -= sp.Splinesep;
    }
    if (nb > b)
      b = nb;
    rv.UR.x = round(b);
  } else
    rv.UR.x = fmax(round(b), sp.RightBound);

  if (ND_node_type(vn) == VIRTUAL && ND_label(vn)) {
    rv.UR.x -= ND_rw(vn);
    if (rv.UR.x < rv.LL.x)
      rv.UR.x = ND_coord(vn).x;
  }

  rv.LL.y = ND_coord(vn).y - GD_rank(g)[ND_rank(vn)].ht1;
  rv.UR.y = ND_coord(vn).y + GD_rank(g)[ND_rank(vn)].ht2;
  return rv;
}

static node_t *neighbor(graph_t *g, node_t *vn, edge_t *ie, edge_t *oe,
                        int dir) {
  int i;
  node_t *n, *rv = NULL;
  rank_t *rank = &(GD_rank(g)[ND_rank(vn)]);

  for (i = ND_order(vn) + dir; i >= 0 && i < rank->n; i += dir) {
    n = rank->v[i];
    if (ND_node_type(n) == VIRTUAL && ND_label(n)) {
      rv = n;
      break;
    }
    if (ND_node_type(n) == NORMAL) {
      rv = n;
      break;
    }
    if (!pathscross(n, vn, ie, oe)) {
      rv = n;
      break;
    }
  }
  return rv;
}

static bool pathscross(node_t *n0, node_t *n1, edge_t *ie1, edge_t *oe1) {
  edge_t *e0, *e1;
  node_t *na, *nb;
  int order, cnt;

  order = ND_order(n0) > ND_order(n1);
  if (ND_out(n0).size != 1 && ND_out(n1).size != 1)
    return false;
  e1 = oe1;
  if (ND_out(n0).size == 1 && e1) {
    e0 = ND_out(n0).list[0];
    for (cnt = 0; cnt < 2; cnt++) {
      if ((na = aghead(e0)) == (nb = aghead(e1)))
        break;
      if (order != (ND_order(na) > ND_order(nb)))
        return true;
      if (ND_out(na).size != 1 || ND_node_type(na) == NORMAL)
        break;
      e0 = ND_out(na).list[0];
      if (ND_out(nb).size != 1 || ND_node_type(nb) == NORMAL)
        break;
      e1 = ND_out(nb).list[0];
    }
  }
  e1 = ie1;
  if (ND_in(n0).size == 1 && e1) {
    e0 = ND_in(n0).list[0];
    for (cnt = 0; cnt < 2; cnt++) {
      if ((na = agtail(e0)) == (nb = agtail(e1)))
        break;
      if (order != (ND_order(na) > ND_order(nb)))
        return true;
      if (ND_in(na).size != 1 || ND_node_type(na) == NORMAL)
        break;
      e0 = ND_in(na).list[0];
      if (ND_in(nb).size != 1 || ND_node_type(nb) == NORMAL)
        break;
      e1 = ND_in(nb).list[0];
    }
  }
  return false;
}

#ifdef DEBUG
void showpath(path *p) {
  pointf LL, UR;

  fprintf(stderr, "%%!PS\n");
  for (size_t i = 0; i < p->nbox; i++) {
    LL = p->boxes[i].LL;
    UR = p->boxes[i].UR;
    fprintf(stderr,
            "newpath %.04f %.04f moveto %.04f %.04f lineto %.04f %.04f lineto "
            "%.04f %.04f lineto closepath stroke\n",
            LL.x, LL.y, UR.x, LL.y, UR.x, UR.y, LL.x, UR.y);
  }
  fprintf(stderr, "showpage\n");
}
#endif
