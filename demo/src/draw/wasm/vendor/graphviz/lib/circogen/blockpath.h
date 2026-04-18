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

#include <circogen/circular.h>
#include <circogen/nodelist.h>

nodelist_t layout_block(Agraph_t *g, block_t *sn, double, circ_state *state);

#ifdef DEBUG
    extern void prTree(Agraph_t * g);
#endif

#ifdef __cplusplus
}
#endif
