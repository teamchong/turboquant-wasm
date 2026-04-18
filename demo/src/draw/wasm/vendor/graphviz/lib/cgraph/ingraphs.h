/**
 * @file
 * @brief supports user-supplied data
 *
 * The ingraphs library works with libcgraph with all user-supplied data.
 * For this to work, the include file relies upon its context to supply
 * a definition of @ref Agraph_t.
 *
 * @ingroup cgraph_app
 */
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

#include "config.h"
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#ifdef EXPORT_CGHDR
#define CGHDR_API __declspec(dllexport)
#else
#define CGHDR_API __declspec(dllimport)
#endif
#endif

#ifndef CGHDR_API
#define CGHDR_API /* nothing */
#endif

    typedef struct {
	union {
	    char**     Files;
	    Agraph_t** Graphs;
	} u;
	int ctr;
	int ingraphs;
	void *fp;
	Agraph_t *(*readf)(const char *, void *);
	bool heap;
	unsigned errors;
    } ingraph_state;

CGHDR_API ingraph_state *newIngraph(ingraph_state *, char **);
CGHDR_API ingraph_state *newIng(ingraph_state *, char **,
                                Agraph_t *(*readf)(const char *, void *));
CGHDR_API ingraph_state *newIngGraphs(ingraph_state *, Agraph_t **,
                                      Agraph_t *(*readf)(const char *, void *));
CGHDR_API void closeIngraph(ingraph_state * sp);
CGHDR_API Agraph_t *nextGraph(ingraph_state *);
CGHDR_API char *fileName(ingraph_state *);

#ifdef __cplusplus
}
#endif
