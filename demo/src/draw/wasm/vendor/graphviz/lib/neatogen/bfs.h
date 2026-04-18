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

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <neatogen/defs.h>

    typedef struct {
	int *data;
	int queueSize;
	int end;
	int start;
    } Queue;

    extern void mkQueue(Queue *, int);
    extern void freeQueue(Queue *);
    extern void initQueue(Queue *, int startVertex);
    extern bool deQueue(Queue *, int *);
    extern bool enQueue(Queue *, int);

    extern void bfs(int, vtx_data*, int, DistType*);

#ifdef __cplusplus
}
#endif
