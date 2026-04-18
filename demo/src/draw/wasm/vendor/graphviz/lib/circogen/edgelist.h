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

#include  <render.h>

    typedef struct edgelistitem {
	Dtlink_t link;
	Agedge_t *edge;
    } edgelistitem;

    typedef Dt_t edgelist;

    extern edgelist *init_edgelist(void);
    extern void add_edge(edgelist * list, Agedge_t * e);
    extern void remove_edge(edgelist * list, Agedge_t * e);
    extern void free_edgelist(edgelist * list);
#ifdef DEBUG
    extern void print_edge(edgelist *);
#endif

#ifdef __cplusplus
}
#endif
