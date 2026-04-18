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

    extern void PCA_alloc(DistType **, int, int, double **, int);
    extern bool iterativePCA_1D(double **, int, int, double *);

#ifdef __cplusplus
}
#endif
