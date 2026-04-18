/**
 * @file
 * @brief make directed graph acyclic, implements @ref graphviz_acyclic, used
 * in cmd/tools/acyclic.c
 *
 * @ingroup cgraph_app
 *
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
#include <stdbool.h>
#include <stddef.h>

typedef struct {
  Agrec_t h;
  int mark;
  bool onstack : 1;
} Agnodeinfo_t;

#define ND_mark(n) (((Agnodeinfo_t *)((n)->base.data))->mark)
#define ND_onstack(n) (((Agnodeinfo_t *)((n)->base.data))->onstack)
#define graphName(g) (agnameof(g))

/* Add a reversed version of e. The new edge has the same key.
 * We also copy the attributes, reversing the roles of head and
 * tail ports.
 * This assumes we've already checked that such an edge does not exist.
 */
static void addRevEdge(Agraph_t *g, Agedge_t *e) {
  Agsym_t *sym;
  Agedge_t *f = agedge(g, aghead(e), agtail(e), agnameof(e), 1);

  agcopyattr(e, f);

  sym = agattr_text(g, AGEDGE, TAILPORT_ID, 0);
  if (sym)
    agsafeset(f, HEADPORT_ID, agxget(e, sym), "");
  sym = agattr_text(g, AGEDGE, HEADPORT_ID, 0);
  if (sym)
    agsafeset(f, TAILPORT_ID, agxget(e, sym), "");
}

/// Return true if the graph has a cycle.
static bool dfs(Agraph_t *g, Agnode_t *t, bool hasCycle, size_t *num_rev) {
  Agedge_t *e;
  Agedge_t *f;
  Agnode_t *h;

  ND_mark(t) = 1;
  ND_onstack(t) = true;
  for (e = agfstout(g, t); e; e = f) {
    f = agnxtout(g, e);
    if (agtail(e) == aghead(e))
      continue;
    h = aghead(e);
    if (ND_onstack(h)) {
      if (agisstrict(g)) {
        if (agedge(g, h, t, 0, 0) == 0) {
          addRevEdge(g, e);
          ++*num_rev;
        }
      } else {
        char *key = agnameof(e);
        if (!key || agedge(g, h, t, key, 0) == 0) {
          addRevEdge(g, e);
          ++*num_rev;
        }
      }
      agdelete(g, e);
      hasCycle = true;
    } else if (ND_mark(h) == 0)
      hasCycle |= dfs(g, h, hasCycle, num_rev);
  }
  ND_onstack(t) = false;
  return hasCycle;
}

bool graphviz_acyclic(Agraph_t *g, const graphviz_acyclic_options_t *opts,
                      size_t *num_rev) {
  bool has_cycle = false;
  aginit(g, AGNODE, "info", sizeof(Agnodeinfo_t), true);
  for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n)) {
    if (ND_mark(n) == 0) {
      has_cycle |= dfs(g, n, false, num_rev);
    }
  }
  if (opts->doWrite) {
    agwrite(g, opts->outFile);
    fflush(opts->outFile);
  }
  return has_cycle;
}
