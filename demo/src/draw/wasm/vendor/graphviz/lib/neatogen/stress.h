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

#define tolerance_cg 1e-3

#define DFLT_ITERATIONS 200

#define DFLT_TOLERANCE 1e-4

    /* some possible values for 'num_pivots_stress' */
#define num_pivots_stress 40

#define opt_smart_init 0x4
#define opt_exp_flag   0x3

    /* Full dense stress optimization (equivalent to Kamada-Kawai's energy) */
    /* Slowest and most accurate optimization */
    extern int stress_majorization_kD_mkernel(vtx_data * graph,	/* Input graph in sparse representation */
					      int n,	/* Number of nodes */
					      double **coords,	/* coordinates of nodes (output layout)  */
					      node_t **nodes,	/* original nodes  */
					      int dim,	/* dimemsionality of layout */
					      int opts,	/* option flags */
					      int model,	/* model */
					      int maxi	/* max iterations */
	);

extern float *compute_apsp_packed(vtx_data * graph, int n);
extern float *compute_apsp_artificial_weights_packed(vtx_data *graph, int n);
extern float* circuitModel(vtx_data * graph, int nG);
extern float* mdsModel (vtx_data * graph, int nG);
extern int initLayout(int n, int dim, double **coords, node_t **nodes);

#ifdef __cplusplus
}
#endif
