/**
 * @file
 * @brief API cghdr.h
 * @ingroup cgraph_utils
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

#include "config.h"

#include <cgraph/cghdr.h>
#include <stdlib.h>

static Agraph_t *Ag_dictop_G;

Dict_t *agdtopen(Dtdisc_t *disc, Dtmethod_t *method) {
    return dtopen(disc, method);
}

int agdtdelete(Agraph_t * g, Dict_t * dict, void *obj)
{
    Ag_dictop_G = g;
    return dtdelete(dict, obj) != NULL;
}

int agdtclose(Agraph_t * g, Dict_t * dict)
{
    dtdisc(dict, NULL);
    Ag_dictop_G = g;
    if (dtclose(dict))
	return 1;
    Ag_dictop_G = NULL;
    return 0;
}

void agdtdisc(Agraph_t * g, Dict_t * dict, Dtdisc_t * disc)
{
    (void)g; /* unused */
    if (disc && dtdisc(dict, NULL) != disc) {
	dtdisc(dict, disc);
    }
    /* else unchanged, disc is same as old disc */
}
/// @defgroup cgraph_utils utilities
/// @brief low level cgraph utilities
/// @ingroup cgraph
