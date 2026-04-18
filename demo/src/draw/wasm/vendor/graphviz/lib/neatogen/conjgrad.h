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

#include <neatogen/defs.h>
#include <stdbool.h>

/*************************
 * C.G. method - SPARSE  *
 ************************/

    extern int conjugate_gradient(vtx_data *, double *, double *, int,
				   double, int);

/*************************
 * C.G. method - DENSE   *
 ************************/

    extern int conjugate_gradient_f(float **, double *, double *, int,
				     double, int, bool);

    extern int conjugate_gradient_mkernel(float *, float *, float *, int,
					   double, int);

#ifdef __cplusplus
}
#endif
