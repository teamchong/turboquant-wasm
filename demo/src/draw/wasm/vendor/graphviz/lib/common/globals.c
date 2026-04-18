/// @file
/// @brief @ref fdp_parms
/// @ingroup common_render
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

#define EXTERN
#include <common/types.h>
#include <common/globals.h>
#include <fdpgen/fdp.h>
#include <util/list.h>

show_boxes_t Show_boxes = {.dtor = LIST_DTOR_FREE};

/* Default layout values, possibly set via command line; -1 indicates unset */
static fdpParms_t fdpParms = {
    1,                          /* useGrid */
    1,                          /* useNew */
    -1,                         /* numIters */
    50,                         /* unscaled */
    0.0,                        /* C */
    1.0,                        /* Tfact */
    -1.0,                       /* K - set in initParams; used in init_edge */
    -1.0,                       /* T0 */
};

struct fdpParms_s* fdp_parms = &fdpParms;
