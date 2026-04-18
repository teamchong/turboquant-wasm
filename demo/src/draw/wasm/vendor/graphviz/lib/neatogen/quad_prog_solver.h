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

#ifdef DIGCOLA

typedef struct {
	float **A;
	int n;
	float *fArray1;
	float *fArray2;
	float *fArray3;
	float *fArray4;
	int *ordering;
	int *levels;
	int num_levels;
}CMajEnv;

extern CMajEnv* initConstrainedMajorization(float *, int, int*, int*, int);

extern void constrained_majorization_new_with_gaps(CMajEnv*, float*, float**, 
                                                   int, int, float);
extern void deleteCMajEnv(CMajEnv *e);

extern float** unpackMatrix(float * packedMat, int n);

#endif 

#ifdef __cplusplus
}
#endif
