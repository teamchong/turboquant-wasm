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

#define MODEL_SHORTPATH      0
#define MODEL_CIRCUIT        1
#define MODEL_SUBSET         2
#define MODEL_MDS            3

#define MODE_KK          0
#define MODE_MAJOR       1
#define MODE_HIER        2
#define MODE_IPSEP       3
#define MODE_SGD         4

#define INIT_ERROR       -1
#define INIT_SELF        0
#define INIT_REGULAR     1
#define INIT_RANDOM      2

#include	"render.h"
#include	"pathplan.h"
#include	<neatogen/neatoprocs.h>
#include	<neatogen/adjust.h>

int lu_decompose(double **a, int n);
void lu_solve(double *x, double *b, int n);
