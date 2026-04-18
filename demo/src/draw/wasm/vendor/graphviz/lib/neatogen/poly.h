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

#include <neatogen/geometry.h>

    typedef struct {
	Point origin;
	Point corner;
	int nverts;
	Point *verts;
	int kind;
    } Poly;

    extern void polyFree(void);
    extern int polyOverlap(Point, Poly *, Point, Poly *);
int makePoly(Poly *, Agnode_t *, double, double);
int makeAddPoly(Poly *, Agnode_t *, double, double);
    extern void breakPoly(Poly *);

#ifdef __cplusplus
}
#endif
