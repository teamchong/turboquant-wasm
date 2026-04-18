/// @file
/// @ingroup common_utils
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

#include <util/agxbuf.h>

#ifdef __cplusplus
extern "C" {
#endif

    union HTMLSTYPE;
    typedef struct htmlparserstate_s htmlparserstate_t;
    typedef struct htmlscan_s htmlscan_t;

    extern int initHTMLlexer(htmlscan_t *, char *, agxbuf *, htmlenv_t *);
    extern int htmllex(union HTMLSTYPE *, htmlscan_t *);
    extern unsigned long htmllineno(htmlscan_t *);
    extern int clearHTMLlexer(htmlscan_t *);
    void htmlerror(htmlscan_t *, const char *);

#ifdef __cplusplus
}
#endif
