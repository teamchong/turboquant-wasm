/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include  <cgraph/list.h>
#include  <render.h>
#include  <stddef.h>

    DEFINE_LIST(nodelist, node_t*)

    /// Add node after one.
    extern void appendNodelist(nodelist_t*, size_t, Agnode_t *n);

    /// Make np new front of list, with current last hooked to current first.
    extern void realignNodelist(nodelist_t *list, size_t np);

    /// Remove cn. Then, insert cn before neighbor if pos == 0 and after
    /// neighbor otherwise.
    extern void insertNodelist(nodelist_t *, Agnode_t *, Agnode_t *, int);

    /// Create l1 @ (rev l2) Destroys and frees l2.
    extern void reverseAppend(nodelist_t *, nodelist_t *);

#ifdef DEBUG
    extern void printNodelist(nodelist_t * list);
#endif

#ifdef __cplusplus
}
#endif
