/**
 * @file
 * @ingroup cgraph_app
 */

#include "config.h"

#include <assert.h>
#include <cgraph/cghdr.h>
#include <stddef.h>

size_t graphviz_node_induce(Agraph_t *g, Agraph_t *edgeset) {

  assert(g != NULL);

  if (edgeset == NULL) {
    edgeset = agroot(g);
  }

  // if the graph and the edge set are the same, by implication the graph
  // already contains all the edges we would consider adding
  if (g == edgeset) {
    return 0;
  }

  size_t count = 0;
  for (Agnode_t *n = agfstnode(g); n != NULL; n = agnxtnode(g, n)) {
    for (Agedge_t *e = agfstout(edgeset, n); e != NULL;
         e = agnxtout(edgeset, e)) {
      if (agsubnode(g, aghead(e), 0)) {
        agsubedge(g, e, 1);
        ++count;
      }
    }
  }

  return count;
}
