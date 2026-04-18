/**
 * Authors:
 *   Tim Dwyer <tgdwyer@gmail.com>
 *
 * Copyright (C) 2005 Authors
 *
 * This version is released under the CPL (Common Public License) with
 * the Graphviz distribution.
 * A version is also available under the LGPL as part of the Adaptagrams
 * project: http://sourceforge.net/projects/adaptagrams.  
 * If you make improvements or bug fixes to this code it would be much
 * appreciated if you could also contribute those changes back to the
 * Adaptagrams repository.
 */

/**********************************************************
*      Written by Tim Dwyer for the graphviz package      *
*                  https://graphviz.org                   *
*                                                         *
**********************************************************/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DIGCOLA

#include <neatogen/defs.h>
#include <neatogen/digcola.h>
#include <stdbool.h>

typedef struct CMajEnvVPSC {
	float **A;
	int nv;   /* number of actual vars */
	int nldv; /* number of dummy nodes included in lap matrix */
	int ndv;  /* number of dummy nodes not included in lap matrix */
	Variable **vs;
	int m; /* total number of constraints for next iteration */
	int gm; /* number of global constraints */
	Constraint **cs;
	/* global constraints are persistent throughout optimisation process */
	Constraint **gcs;
	VPSC *vpsc;
	float *fArray1; /* utility arrays - reusable memory */
	float *fArray2;
	float *fArray3;
} CMajEnvVPSC;

extern CMajEnvVPSC* initCMajVPSC(int n, float *packedMat, vtx_data* graph, ipsep_options *opt, int diredges);

extern int constrained_majorization_vpsc(CMajEnvVPSC*, float*, float*, int);

extern void deleteCMajEnvVPSC(CMajEnvVPSC *e);
extern void generateNonoverlapConstraints(
        CMajEnvVPSC* e,
        float nsizeScale,
        float** coords,
        int k,
	bool transitiveClosure,
	ipsep_options* opt
);

extern void removeoverlaps(int,float**,ipsep_options*);

typedef struct {
	int *nodes;
	int num_nodes;
} DigColaLevel;

/*
 * unpack the "ordering" array into an array of DigColaLevel (as defined above)
 */
extern DigColaLevel* assign_digcola_levels(int *ordering, int n, int *level_inds, int num_divisions);
int get_num_digcola_constraints(DigColaLevel *levels, int num_levels);
#endif 

#ifdef __cplusplus
}
#endif
