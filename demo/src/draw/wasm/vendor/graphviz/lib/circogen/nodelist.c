/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include	<circogen/nodelist.h>
#include	<circogen/circular.h>
#include	<assert.h>
#include	<limits.h>
#include	<stddef.h>
#include	<string.h>

void appendNodelist(nodelist_t *list, size_t one, Agnode_t *n) {
  assert(one <= nodelist_size(list));

  // expand the list by one element
  nodelist_append(list, NULL);

  // shuffle everything past where we will insert
  nodelist_sync(list);
  size_t to_move = sizeof(node_t*) * (nodelist_size(list) - one - 1);
  if (to_move > 0) {
    memmove(nodelist_at(list, one + 1), nodelist_at(list, one), to_move);
  }

  // insert the new node
  nodelist_set(list, one, n);
}

void realignNodelist(nodelist_t *list, size_t np) {
  assert(np < nodelist_size(list));
  for (size_t i = np; i != 0; --i) {
    // rotate the list by 1
    node_t *const head = nodelist_pop_front(list);
    nodelist_push_back(list, head);
  }
}

void
insertNodelist(nodelist_t * list, Agnode_t * cn, Agnode_t * neighbor,
	       int pos)
{
  nodelist_remove(list, cn);

  for (size_t i = 0; i < nodelist_size(list); ++i) {
    Agnode_t *here = nodelist_get(list, i);
    if (here == neighbor) {
      if (pos == 0) {
        appendNodelist(list, i, cn);
      } else {
        appendNodelist(list, i + 1, cn);
      }
      break;
    }
  }
}

/// attach l2 to l1.
static void concatNodelist(nodelist_t * l1, nodelist_t * l2)
{
  for (size_t i = 0; i < nodelist_size(l2); ++i) {
    nodelist_append(l1, nodelist_get(l2, i));
  }
}

void reverseAppend(nodelist_t * l1, nodelist_t * l2)
{
    nodelist_reverse(l2);
    concatNodelist(l1, l2);
    nodelist_free(l2);
}

#ifdef DEBUG
void printNodelist(nodelist_t * list)
{
    for (size_t i = 0; i < nodelist_size(list); ++i) {
      fprintf(stderr, "%s ", agnameof(nodelist_get(list, i)));
    }
    fputs("\n", stderr);
}
#endif
