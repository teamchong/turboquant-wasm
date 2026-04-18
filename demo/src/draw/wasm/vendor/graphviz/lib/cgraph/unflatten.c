/**
 * @file
 * @brief adjusts directed graphs to improve layout aspect ratio,
 * API: cgraph.h,
 * implements @ref graphviz_unflatten, used in cmd/tools/unflatten.c
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
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <util/itos.h>

static int myindegree(Agnode_t *n) { return agdegree(n->root, n, 1, 0); }

// need outdegree without selfarcs
static int myoutdegree(Agnode_t *n) {
  Agedge_t *e;
  int rv = 0;

  for (e = agfstout(n->root, n); e; e = agnxtout(n->root, e)) {
    if (agtail(e) != aghead(e))
      rv++;
  }
  return rv;
}

static bool isleaf(Agnode_t *n) { return myindegree(n) + myoutdegree(n) == 1; }

static bool ischainnode(Agnode_t *n) {
  return myindegree(n) == 1 && myoutdegree(n) == 1;
}

static void adjustlen(Agedge_t *e, Agsym_t *sym, int newlen) {
  agxset(e, sym, ITOS(newlen));
}

static Agsym_t *bindedgeattr(Agraph_t *g, char *str) {
  return agattr_text(g, AGEDGE, str, "");
}

void graphviz_unflatten(Agraph_t *g, const graphviz_unflatten_options_t *opts) {
  Agnode_t *n;
  Agedge_t *e;
  char *str;
  Agsym_t *m_ix, *s_ix;
  int cnt, d;
  int ChainSize = 0;
  Agnode_t *ChainNode = NULL;

  m_ix = bindedgeattr(g, "minlen");
  s_ix = bindedgeattr(g, "style");

  for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
    d = myindegree(n) + myoutdegree(n);
    if (d == 0) {
      if (opts->ChainLimit < 1)
        continue;
      if (ChainNode) {
        e = agedge(g, ChainNode, n, "", 1);
        agxset(e, s_ix, "invis");
        ChainSize++;
        if (ChainSize < opts->ChainLimit)
          ChainNode = n;
        else {
          ChainNode = NULL;
          ChainSize = 0;
        }
      } else
        ChainNode = n;
    } else if (d > 1) {
      if (opts->MaxMinlen < 1)
        continue;
      cnt = 0;
      for (e = agfstin(g, n); e; e = agnxtin(g, e)) {
        if (isleaf(agtail(e))) {
          str = agxget(e, m_ix);
          if (str[0] == 0) {
            adjustlen(e, m_ix, cnt % opts->MaxMinlen + 1);
            cnt++;
          }
        }
      }

      cnt = 0;
      for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
        if (isleaf(e->node) || (opts->Do_fans && ischainnode(e->node))) {
          str = agxget(e, m_ix);
          if (str[0] == 0)
            adjustlen(e, m_ix, cnt % opts->MaxMinlen + 1);
          cnt++;
        }
      }
    }
  }
}
