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

/*-----------------------------------------------------------------------------
| Definitions and global variables.
-----------------------------------------------------------------------------*/
#include <label/rectangle.h>
#include <label/index.h>
#include <stdint.h>

#ifndef METHODS
#define METHODS 1
#endif /*METHODS*/
/* variables for finding a partition */
    struct PartitionVars {
    int partition[NODECARD + 1];
    int taken[NODECARD + 1];
    int count[2];
    struct Rect cover[2];
    uint64_t area[2];
};

typedef struct split_q_s {
    struct Branch BranchBuf[NODECARD + 1];
    struct Rect CoverSplit;
    uint64_t CoverSplitArea;
    struct PartitionVars Partitions[METHODS];
} SplitQ_t;

void SplitNode(RTree_t *, Node_t *, Branch_t *, Node_t **);

#ifdef __cplusplus
}
#endif
