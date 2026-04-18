/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include "config.h"

#include <cgraph/cgraph.h>
#include <common/render.h>
#include <common/utils.h>
#include <limits.h>
#include <pack/pack.h>
#include <stdbool.h>
#include <stdlib.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/gv_ctype.h>
#include <util/list.h>
#include <util/prisize_t.h>

typedef LIST(Agnode_t *) node_stack_t;

typedef struct {
  node_stack_t data;
  void (*actionfn)(Agnode_t *, void *);
  bool (*markfn)(Agnode_t *, int);
} stk_t;

/// does `n` have a mark set?
static bool marked(const stk_t *stk, Agnode_t *n) { return stk->markfn(n, -1); }

/// set a mark on `n`
static void mark(const stk_t *stk, Agnode_t *n) { stk->markfn(n, 1); }

/// unset a mark on `n`
static void unmark(const stk_t *stk, Agnode_t *n) { stk->markfn(n, 0); }

static stk_t initStk(void (*actionfn)(Agnode_t *, void *),
                     bool (*markfn)(Agnode_t *, int)) {
  return (stk_t){.actionfn = actionfn, .markfn = markfn};
}

static void freeStk(stk_t *sp) { LIST_FREE(&sp->data); }

static void push(stk_t *sp, Agnode_t *np) {
  mark(sp, np);
  LIST_PUSH_BACK(&sp->data, np);
}

static Agnode_t *pop(stk_t *sp) {
  if (LIST_IS_EMPTY(&sp->data)) {
    return NULL;
  }

  return LIST_POP_BACK(&sp->data);
}

static size_t dfs(Agraph_t *g, Agnode_t *n, void *state, stk_t *stk) {
  Agedge_t *e;
  Agnode_t *other;
  size_t cnt = 0;

  push(stk, n);
  while ((n = pop(stk))) {
    cnt++;
    if (stk->actionfn)
      stk->actionfn(n, state);
    for (e = agfstedge(g, n); e; e = agnxtedge(g, e, n)) {
      if ((other = agtail(e)) == n)
        other = aghead(e);
      if (!marked(stk, other))
        push(stk, other);
    }
  }
  return cnt;
}

static bool isLegal(const char *p) {
  char c;

  while ((c = *p++)) {
    if (c != '_' && !gv_isalnum(c))
      return false;
  }

  return true;
}

static void insertFn(Agnode_t *n, void *state) { agsubnode(state, n, 1); }

static bool markFn(Agnode_t *n, int v) {
  if (v < 0)
    return ND_mark(n) != 0;
  const size_t ret = ND_mark(n);
  ND_mark(n) = v != 0;
  return ret != 0;
}

static void setPrefix(agxbuf *xb, const char *pfx) {
  if (!pfx || !isLegal(pfx)) {
    pfx = "_cc_";
  }
  agxbput(xb, pfx);
}

/* pccomps:
 * Return an array of subgraphs consisting of the connected
 * components of graph g. The number of components is returned in ncc.
 * All pinned nodes are in one component.
 * If pfx is non-null and a legal graph name, we use it as the prefix
 * for the name of the subgraphs created. If not, a simple default is used.
 * If pinned is non-null, *pinned set to 1 if pinned nodes found
 * and the first component is the one containing the pinned nodes.
 * Note that the component subgraphs do not contain any edges. These must
 * be obtained from the root graph.
 * Return NULL if graph is empty.
 */
Agraph_t **pccomps(Agraph_t *g, size_t *ncc, char *pfx, bool *pinned) {
  agxbuf name = {0};
  Agraph_t *out = NULL;
  Agnode_t *n;
  bool pin = false;

  if (agnnodes(g) == 0) {
    *ncc = 0;
    return NULL;
  }

  LIST(Agraph_t *) ccs = {0};

  stk_t stk = initStk(insertFn, markFn);
  for (n = agfstnode(g); n; n = agnxtnode(g, n))
    unmark(&stk, n);

  /* Component with pinned nodes */
  for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
    if (marked(&stk, n) || !isPinned(n))
      continue;
    if (!out) {
      setPrefix(&name, pfx);
      agxbprint(&name, "%" PRISIZE_T, LIST_SIZE(&ccs));
      out = agsubg(g, agxbuse(&name), 1);
      agbindrec(out, "Agraphinfo_t", sizeof(Agraphinfo_t),
                true); // node custom data
      LIST_APPEND(&ccs, out);
      pin = true;
    }
    dfs(g, n, out, &stk);
  }

  /* Remaining nodes */
  for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
    if (marked(&stk, n))
      continue;
    setPrefix(&name, pfx);
    agxbprint(&name, "%" PRISIZE_T, LIST_SIZE(&ccs));
    out = agsubg(g, agxbuse(&name), 1);
    agbindrec(out, "Agraphinfo_t", sizeof(Agraphinfo_t),
              true); // node custom data
    dfs(g, n, out, &stk);
    LIST_APPEND(&ccs, out);
  }
  freeStk(&stk);
  agxbfree(&name);
  *pinned = pin;
  Agraph_t **ret;
  LIST_DETACH(&ccs, &ret, ncc);
  return ret;
}

/* ccomps:
 * Return an array of subgraphs consisting of the connected
 * components of graph g. The number of components is returned in ncc.
 * If pfx is non-null and a legal graph name, we use it as the prefix
 * for the name of the subgraphs created. If not, a simple default is used.
 * Note that the component subgraphs do not contain any edges. These must
 * be obtained from the root graph.
 * Returns NULL on error or if graph is empty.
 */
Agraph_t **ccomps(Agraph_t *g, size_t *ncc, char *pfx) {
  agxbuf name = {0};
  Agraph_t *out;
  Agnode_t *n;

  if (agnnodes(g) == 0) {
    *ncc = 0;
    return NULL;
  }

  LIST(Agraph_t *) ccs = {0};
  stk_t stk = initStk(insertFn, markFn);
  for (n = agfstnode(g); n; n = agnxtnode(g, n))
    unmark(&stk, n);

  for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
    if (marked(&stk, n))
      continue;
    setPrefix(&name, pfx);
    agxbprint(&name, "%" PRISIZE_T, LIST_SIZE(&ccs));
    out = agsubg(g, agxbuse(&name), 1);
    agbindrec(out, "Agraphinfo_t", sizeof(Agraphinfo_t),
              true); // node custom data
    dfs(g, n, out, &stk);
    LIST_APPEND(&ccs, out);
  }
  freeStk(&stk);
  agxbfree(&name);
  Agraph_t **ret;
  LIST_DETACH(&ccs, &ret, ncc);
  return ret;
}

typedef struct {
  Agrec_t h;
  char cc_subg; /* true iff subgraph corresponds to a component */
} ccgraphinfo_t;

typedef struct {
  Agrec_t h;
  char mark;
  union {
    Agraph_t *g;
    Agnode_t *n;
    void *v;
  } ptr;
} ccgnodeinfo_t;

#define GRECNAME "ccgraphinfo"
#define NRECNAME "ccgnodeinfo"
#define GD_cc_subg(g) (((ccgraphinfo_t *)aggetrec(g, GRECNAME, 0))->cc_subg)
#ifdef DEBUG
Agnode_t *dnodeOf(Agnode_t *v) {
  ccgnodeinfo_t *ip = (ccgnodeinfo_t *)aggetrec(v, NRECNAME, 0);
  if (ip)
    return ip->ptr.n;
  fprintf(stderr, "nodeinfo undefined\n");
  return NULL;
}
void dnodeSet(Agnode_t *v, Agnode_t *n) {
  ((ccgnodeinfo_t *)aggetrec(v, NRECNAME, 0))->ptr.n = n;
}
#else
#define dnodeOf(v) (((ccgnodeinfo_t *)aggetrec(v, NRECNAME, 0))->ptr.n)
#define dnodeSet(v, w) (((ccgnodeinfo_t *)aggetrec(v, NRECNAME, 0))->ptr.n = w)
#endif

#define ptrOf(np) (((ccgnodeinfo_t *)((np)->base.data))->ptr.v)
#define nodeOf(np) (((ccgnodeinfo_t *)((np)->base.data))->ptr.n)
#define clustOf(np) (((ccgnodeinfo_t *)((np)->base.data))->ptr.g)
#define clMark(n) (((ccgnodeinfo_t *)(n->base.data))->mark)

/* deriveClusters:
 * Construct nodes in derived graph corresponding top-level clusters.
 * Since a cluster might be wrapped in a subgraph, we need to traverse
 * down into the tree of subgraphs
 */
static void deriveClusters(Agraph_t *dg, Agraph_t *g) {
  Agraph_t *subg;
  Agnode_t *dn;
  Agnode_t *n;

  for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
    if (is_a_cluster(subg)) {
      dn = agnode(dg, agnameof(subg), 1);
      agbindrec(dn, NRECNAME, sizeof(ccgnodeinfo_t), true);
      clustOf(dn) = subg;
      for (n = agfstnode(subg); n; n = agnxtnode(subg, n)) {
        if (dnodeOf(n)) {
          fprintf(stderr,
                  "Error: node \"%s\" belongs to two non-nested clusters "
                  "\"%s\" and \"%s\"\n",
                  agnameof(n), agnameof(subg), agnameof(dnodeOf(n)));
        }
        dnodeSet(n, dn);
      }
    } else {
      deriveClusters(dg, subg);
    }
  }
}

/* deriveGraph:
 * Create derived graph dg of g where nodes correspond to top-level nodes
 * or clusters, and there is an edge in dg if there is an edge in g
 * between any nodes in the respective clusters.
 */
static Agraph_t *deriveGraph(Agraph_t *g) {
  Agraph_t *dg;
  Agnode_t *dn;
  Agnode_t *n;

  dg = agopen("dg", Agstrictundirected, NULL);

  deriveClusters(dg, g);

  for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
    if (dnodeOf(n))
      continue;
    dn = agnode(dg, agnameof(n), 1);
    agbindrec(dn, NRECNAME, sizeof(ccgnodeinfo_t), true);
    nodeOf(dn) = n;
    dnodeSet(n, dn);
  }

  for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
    Agedge_t *e;
    Agnode_t *hd;
    Agnode_t *tl = dnodeOf(n);
    for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
      hd = aghead(e);
      hd = dnodeOf(hd);
      if (hd == tl)
        continue;
      if (hd > tl)
        agedge(dg, tl, hd, NULL, 1);
      else
        agedge(dg, hd, tl, NULL, 1);
    }
  }

  return dg;
}

/* unionNodes:
 * Add all nodes in cluster nodes of dg to g
 */
static void unionNodes(Agraph_t *dg, Agraph_t *g) {
  Agnode_t *n;
  Agnode_t *dn;
  Agraph_t *clust;

  for (dn = agfstnode(dg); dn; dn = agnxtnode(dg, dn)) {
    if (AGTYPE(ptrOf(dn)) == AGNODE) {
      agsubnode(g, nodeOf(dn), 1);
    } else {
      clust = clustOf(dn);
      for (n = agfstnode(clust); n; n = agnxtnode(clust, n))
        agsubnode(g, n, 1);
    }
  }
}

static bool clMarkFn(Agnode_t *n, int v) {
  int ret;
  if (v < 0)
    return clMark(n) != 0;
  ret = clMark(n);
  clMark(n) = (char)v;
  return ret != 0;
}

typedef struct {
  Agrec_t h;
  Agraph_t *orig;
} orig_t;

#define ORIG_REC "orig"

Agraph_t *mapClust(Agraph_t *cl) {
  orig_t *op = (orig_t *)aggetrec(cl, ORIG_REC, 0);
  assert(op);
  return op->orig;
}

/* projectG:
 * If any nodes of subg are in g, create a subgraph of g
 * and fill it with all nodes of subg in g and their induced
 * edges in subg. Copy the attributes of subg to g. Return the subgraph.
 * If not, return null.
 * If subg is a cluster, the new subgraph will contain a pointer to it
 * in the record "orig".
 */
static Agraph_t *projectG(Agraph_t *subg, Agraph_t *g, int inCluster) {
  Agraph_t *proj = NULL;
  Agnode_t *n;
  Agnode_t *m;
  orig_t *op;

  for (n = agfstnode(subg); n; n = agnxtnode(subg, n)) {
    if ((m = agfindnode(g, agnameof(n)))) {
      if (proj == NULL) {
        proj = agsubg(g, agnameof(subg), 1);
      }
      agsubnode(proj, m, 1);
    }
  }
  if (!proj && inCluster) {
    proj = agsubg(g, agnameof(subg), 1);
  }
  if (proj) {
    (void)graphviz_node_induce(proj, subg);
    agcopyattr(subg, proj);
    if (is_a_cluster(proj)) {
      op = agbindrec(proj, ORIG_REC, sizeof(orig_t), false);
      op->orig = subg;
    }
  }

  return proj;
}

/* subgInduce:
 * Project subgraphs of root graph on subgraph.
 * If non-empty, add to subgraph.
 */
static void subgInduce(Agraph_t *root, Agraph_t *g, int inCluster) {
  Agraph_t *subg;
  Agraph_t *proj;
  int in_cluster;

  for (subg = agfstsubg(root); subg; subg = agnxtsubg(subg)) {
    if (GD_cc_subg(subg))
      continue;
    if ((proj = projectG(subg, g, inCluster))) {
      in_cluster = (inCluster || is_a_cluster(subg));
      subgInduce(subg, proj, in_cluster);
    }
  }
}

static void subGInduce(Agraph_t *g, Agraph_t *out) { subgInduce(g, out, 0); }

/* cccomps:
 * Decompose g into "connected" components, where nodes are connected
 * either by an edge or by being in the same cluster. The components
 * are returned in an array of subgraphs. ncc indicates how many components
 * there are. The subgraphs use the prefix pfx in their names, if non-NULL.
 * Note that cluster subgraph of the main graph, corresponding to a component,
 * is cloned within the subgraph. Each cloned cluster contains a record pointing
 * to the real cluster.
 */
Agraph_t **cccomps(Agraph_t *g, size_t *ncc, char *pfx) {
  Agraph_t *dg;
  size_t n_cnt, e_cnt;
  agxbuf name = {0};
  Agraph_t *out;
  Agraph_t *dout;
  Agnode_t *dn;
  int sz = (int)sizeof(ccgraphinfo_t);

  if (agnnodes(g) == 0) {
    *ncc = 0;
    return NULL;
  }

  /* Bind ccgraphinfo to graph and all subgraphs */
  aginit(g, AGRAPH, GRECNAME, -sz, false);

  /* Bind ccgraphinfo to graph and all subgraphs */
  aginit(g, AGNODE, NRECNAME, sizeof(ccgnodeinfo_t), false);

  dg = deriveGraph(g);

  size_t ccs_length = (size_t)agnnodes(dg);
  LIST(Agraph_t *) ccs = {0};
  LIST_RESERVE(&ccs, ccs_length);
  stk_t stk = initStk(insertFn, clMarkFn);

  for (dn = agfstnode(dg); dn; dn = agnxtnode(dg, dn)) {
    if (marked(&stk, dn))
      continue;
    setPrefix(&name, pfx);
    agxbprint(&name, "%" PRISIZE_T, LIST_SIZE(&ccs));
    char *name_str = agxbuse(&name);
    dout = agsubg(dg, name_str, 1);
    out = agsubg(g, name_str, 1);
    agbindrec(out, GRECNAME, sizeof(ccgraphinfo_t), false);
    GD_cc_subg(out) = 1;
    n_cnt = dfs(dg, dn, dout, &stk);
    unionNodes(dout, out);
    e_cnt = graphviz_node_induce(out, NULL);
    subGInduce(g, out);
    LIST_APPEND(&ccs, out);
    agdelete(dg, dout);
    if (Verbose)
      fprintf(stderr,
              "(%4" PRISIZE_T ") %7" PRISIZE_T " nodes %7" PRISIZE_T " edges\n",
              LIST_SIZE(&ccs) - 1, n_cnt, e_cnt);
  }

  if (Verbose)
    fprintf(stderr,
            "       %7d nodes %7d edges %7" PRISIZE_T " components %s\n",
            agnnodes(g), agnedges(g), LIST_SIZE(&ccs), agnameof(g));

  agclose(dg);
  agclean(g, AGRAPH, GRECNAME);
  agclean(g, AGNODE, NRECNAME);
  freeStk(&stk);
  agxbfree(&name);
  Agraph_t **ret;
  LIST_DETACH(&ccs, &ret, ncc);
  return ret;
}

/* isConnected:
 * Returns 1 if the graph is connected.
 * Returns 0 if the graph is not connected.
 * Returns -1 if the graph is error.
 */
int isConnected(Agraph_t *g) {
  if (agnnodes(g) == 0)
    return 1;

  stk_t stk = initStk(NULL, markFn);
  for (Agnode_t *n = agfstnode(g); n; n = agnxtnode(g, n))
    unmark(&stk, n);

  const size_t cnt = dfs(g, agfstnode(g), NULL, &stk);
  freeStk(&stk);
  if (cnt != (size_t)agnnodes(g))
    return 0;
  return 1;
}
