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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_POINTF_S
    typedef pointf Point;
#else
    typedef struct Point {
	double x, y;
    } Point;
#endif

    extern double xmin, xmax, ymin, ymax;	/* extreme x,y values of sites */
    extern double deltax;	// xmax - xmin

    extern size_t nsites; // Number of sites
    extern int sqrt_nsites;

    extern void geominit(void);
    extern double dist_2(Point, Point); ///< distance squared between two points
    extern void subpt(Point * a, Point b, Point c);
    extern void addpt(Point * a, Point b, Point c);
    extern double area_2(Point a, Point b, Point c);
    extern int leftOf(Point a, Point b, Point c);
    extern int intersection(Point a, Point b, Point c, Point d, Point * p);

#ifdef __cplusplus
}
#endif
