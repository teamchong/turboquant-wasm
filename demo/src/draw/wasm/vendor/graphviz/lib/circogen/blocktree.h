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

#include <render.h>
#include <circogen/circular.h>

    extern block_t *createBlocktree(Agraph_t * g, circ_state * state);
    extern void freeBlocktree(block_t *);
#ifdef DEBUG
    extern void print_blocktree(block_t * sn, int depth);
#endif

#ifdef __cplusplus
}
#endif
