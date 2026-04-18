/**
 * @file
 * @brief
 * [transitive reduction](https://en.wikipedia.org/wiki/Transitive_reduction)
 * filter for directed graphs, API: cgraph.h,
 * implements @ref graphviz_tred,
 * used in cmd/tools/tred.c
 *
 * @ingroup cgraph_app
 */

/*
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Authors: Stephen North, Emden Gansner
 * Contributors: Details at https://graphviz.org
 */

#include "config.h"

#include <cgraph/cghdr.h>
#include <cgraph/node_set.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <util/alloc.h>
#include <util/list.h>

typedef struct {
  bool on_stack : 1;
  unsigned char dist;
} nodeinfo_t;

#define ON_STACK(ninfo, n) (ninfo[AGSEQ(n)].on_stack)
#define DISTANCE(ninfo, n) (ninfo[AGSEQ(n)].dist)
#define agrootof(n) ((n)->root)

static unsigned char uchar_min(unsigned char a, unsigned char b) {
  if (a < b)
    return a;
  return b;
}

typedef LIST(Agedge_t *) edge_stack_t;

static void push(edge_stack_t *sp, Agedge_t *ep, nodeinfo_t *ninfo) {

  // mark this edge on the stack
  ON_STACK(ninfo, aghead(ep)) = true;

  // insert the new edge
  LIST_PUSH_BACK(sp, ep);
}

static Agedge_t *pop(edge_stack_t *sp, nodeinfo_t *ninfo) {

  if (LIST_IS_EMPTY(sp)) {
    return NULL;
  }

  // remove the top
  Agedge_t *e = LIST_POP_BACK(sp);

  // mark it as no longer on the stack
  ON_STACK(ninfo, aghead(e)) = false;

  return e;
}

static Agedge_t *top(edge_stack_t *sp) {

  if (LIST_IS_EMPTY(sp)) {
    return NULL;
  }

  return *LIST_BACK(sp);
}

/* Main function for transitive reduction.
 * This does a DFS starting at node n. Each node records the length of
 * its largest simple path from n. We only care if the length is > 1. Node
 * n will have distance 0; outneighbors of n will have distance 1 or 2; all
 * others will have distance 2.
 *
 * During the DFS, we only push edges on the stack whose head has distance 0
 * (i.e., hasn't been visited yet), setting its distance to the distance of the
 * tail node plus one. If we find a head node with distance 1, we don't push the
 * edge, since it has already been in a DFS, but we update its distance. We also
 * check for back edges and report these.
 *
 * After the DFS, we check all outedges of n. Those edges whose head has
 * distance 2 we delete. We also delete all but one copy of any edges with the
 * same head.
 */
static int dfs(Agnode_t *n, nodeinfo_t *ninfo, int warn,
               const graphviz_tred_options_t *opts) {
  Agraph_t *g = agrootof(n);
  Agedgepair_t dummy;
  Agedge_t *link;
  Agedge_t *next;
  Agedge_t *prev;
  Agedge_t *e;
  Agedge_t *f;
  Agnode_t *v;
  Agnode_t *hd;
  Agnode_t *oldhd;
  int do_delete;

  dummy.out.base.tag.objtype = AGOUTEDGE;
  dummy.out.node = n;
  dummy.in.base.tag.objtype = AGINEDGE;
  dummy.in.node = NULL;

  edge_stack_t estk = {0};
  push(&estk, &dummy.out, ninfo);
  prev = 0;

  while ((link = top(&estk))) {
    v = aghead(link);
    if (prev)
      next = agnxtout(g, prev);
    else
      next = agfstout(g, v);
    for (; next; next = agnxtout(g, next)) {
      hd = aghead(next);
      if (hd == v)
        continue; // Skip a loop
      if (ON_STACK(ninfo, hd)) {
        if (!warn) {
          warn++;
          if (opts->err != NULL) {
            fprintf(
                opts->err,
                "warning: %s has cycle(s), transitive reduction not unique\n",
                agnameof(g));
            fprintf(opts->err, "cycle involves edge %s -> %s\n", agnameof(v),
                    agnameof(hd));
          }
        }
      } else if (DISTANCE(ninfo, hd) == 0) {
        DISTANCE(ninfo, hd) = uchar_min(1, DISTANCE(ninfo, v)) + 1;
        break;
      } else if (DISTANCE(ninfo, hd) == 1) {
        DISTANCE(ninfo, hd) = uchar_min(1, DISTANCE(ninfo, v)) + 1;
      }
    }
    if (next) {
      push(&estk, next, ninfo);
      prev = 0;
    } else {
      prev = pop(&estk, ninfo);
    }
  }
  oldhd = NULL;
  for (e = agfstout(g, n); e; e = f) {
    do_delete = 0;
    f = agnxtout(g, e);
    hd = aghead(e);
    if (oldhd == hd)
      do_delete = 1;
    else {
      oldhd = hd;
      if (DISTANCE(ninfo, hd) > 1)
        do_delete = 1;
    }
    if (do_delete) {
      if (opts->PrintRemovedEdges && opts->err != NULL)
        fprintf(opts->err, "removed edge: %s: \"%s\" -> \"%s\"\n", agnameof(g),
                agnameof(aghead(e)), agnameof(agtail(e)));
      agdelete(g, e);
    }
  }
  LIST_FREE(&estk);
  return warn;
}

/* Do a DFS for each vertex in graph g, so the time
 * complexity is O(|V||E|).
 */
void graphviz_tred(Agraph_t *g, const graphviz_tred_options_t *opts) {
  int cnt = 0;
  int warn = 0;
  time_t total_secs = 0;

  const size_t infosize = (node_set_size(g->n_id) + 1) * sizeof(nodeinfo_t);
  nodeinfo_t *const ninfo = gv_alloc(infosize);

  if (opts->Verbose && opts->err != NULL)
    fprintf(stderr, "Processing graph %s\n", agnameof(g));
  for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
    memset(ninfo, 0, infosize);
    const time_t start = time(NULL);
    warn = dfs(n, ninfo, warn, opts);
    if (opts->Verbose) {
      const time_t secs = time(NULL) - start;
      total_secs += secs;
      cnt++;
      if (cnt % 1000 == 0 && opts->err != NULL) {
        fprintf(opts->err, "[%d]\n", cnt);
      }
    }
  }
  if (opts->Verbose && opts->err != NULL)
    fprintf(opts->err, "Finished graph %s: %lld.00 secs.\n", agnameof(g),
            (long long)total_secs);
  free(ninfo);
  agwrite(g, opts->out);
  fflush(opts->out);
}
/**
 * @defgroup cgraph_app app
 * @brief uncoupled application specific functions
 * @ingroup cgraph
 */
