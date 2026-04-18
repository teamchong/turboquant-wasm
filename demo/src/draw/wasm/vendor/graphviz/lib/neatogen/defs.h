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

#include <neatogen/neato.h>

#include <neatogen/sparsegraph.h>

#ifdef DIGCOLA
#ifdef IPSEPCOLA
    typedef struct cluster_data {
	int nvars;         /* total count of vars in clusters */
        int nclusters;     /* number of clusters */
        int *clustersizes; /* number of vars in each cluster */
        int **clusters;    /* list of var indices for constituents of each c */
	int ntoplevel;     /* number of nodes not in any cluster */
	int *toplevel;     /* array of nodes not in any cluster */
	boxf *bb;	   /* bounding box of each cluster */
    } cluster_data;
#endif
#endif


#ifdef __cplusplus
}
#endif
