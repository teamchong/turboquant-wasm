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

#include <neatogen/site.h>

    typedef struct Edge {
	double a, b, c;		/* edge on line ax + by = c */
	Site *ep[2];		/* endpoints (vertices) of edge; initially NULL */
	Site *reg[2];		/* sites forming edge */
    } Edge;

#define le 0
#define re 1

    extern double pxmin, pxmax, pymin, pymax;	/* clipping window */
    extern void edgeinit(void);
    extern void endpoint(Edge *, int, Site *);
    extern void clip_line(Edge * e);
    extern Edge *gvbisect(Site *, Site *);

#ifdef __cplusplus
}
#endif
