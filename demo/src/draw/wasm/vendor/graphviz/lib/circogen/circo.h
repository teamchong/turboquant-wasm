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

#include "render.h"

#ifdef __cplusplus
extern "C" {
#endif

    extern void circo_layout(Agraph_t * g);
    extern void circoLayout(Agraph_t * g);
    extern void circo_cleanup(Agraph_t * g);
    extern void circo_init_graph(graph_t * g);

#ifdef __cplusplus
}
#endif
