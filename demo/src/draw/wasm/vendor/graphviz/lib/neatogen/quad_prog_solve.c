/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <cgraph/list.h>
#include <neatogen/digcola.h>
#include <stdint.h>
#include <util/alloc.h>
#ifdef DIGCOLA
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include <neatogen/matrix_ops.h>
#include <neatogen/kkutils.h>
#include <neatogen/quad_prog_solver.h>

/// are two values equal, within a reasonable tolerance?
static bool equals(float a, float b) {
  const float TOLERANCE = 1e-2;
  return fabsf(a - b) < TOLERANCE;
}

float **unpackMatrix(float *packedMat, int n)
{
    int i, j, k;

    float **mat = gv_calloc(n, sizeof(float *));
    mat[0] = gv_calloc(n * n, sizeof(float));
    set_vector_valf(n * n, 0, mat[0]);
    for (i = 1; i < n; i++) {
	mat[i] = mat[0] + i * n;
    }

    for (i = 0, k = 0; i < n; i++) {
	for (j = i; j < n; j++, k++) {
	    mat[j][i] = mat[i][j] = packedMat[k];
	}
    }
    return mat;
}

static void
ensureMonotonicOrderingWithGaps(float *place, int n, int *ordering,
				int *levels, int num_levels,
				float levels_gap)
{
    /* ensure that levels are separated in the initial layout and that 
     * places are monotonic increasing within layer
     */

    int i;
    int node, level, max_in_level;
    float lower_bound = -1e9f;

    level = -1;
    max_in_level = 0;
    for (i = 0; i < n; i++) {
	if (i >= max_in_level) {
	    /* we are entering a new level */
	    level++;
	    if (level == num_levels) {
		/* last_level */
		max_in_level = n;
	    } else {
		max_in_level = levels[level];
	    }
	    lower_bound =
		i > 0 ? place[ordering[i - 1]] + levels_gap : -1e9f;
	    quicksort_placef(place, ordering, i, max_in_level - 1);
	}

	node = ordering[i];
	if (place[node] < lower_bound) {
	    place[node] = lower_bound;
	}
    }
}

DEFINE_LIST(ints, int)

void
constrained_majorization_new_with_gaps(CMajEnv * e, float *b,
				       float **coords,
				       int cur_axis, int max_iterations,
				       float levels_gap)
{
    float *place = coords[cur_axis];
    int n = e->n;
    float **lap = e->A;
    int *ordering = e->ordering;
    int *levels = e->levels;
    int num_levels = e->num_levels;
    float new_place_i;
    bool converged = false;
    float upper_bound, lower_bound;
    int node;
    int left, right;
    float cur_place;
    float des_place_block;
    float block_deg;
    float toBlockConnectivity;
    float *lap_node;
    float *desired_place;
    float *prefix_desired_place;
    float *suffix_desired_place;
    int first_next_level;
    int level = -1, max_in_level = 0;
    float *gap;
    float target_place;

    if (max_iterations <= 0) {
	return;
    }

    ensureMonotonicOrderingWithGaps(place, n, ordering, levels, num_levels,
				    levels_gap);
    /* it is important that in 'ordering' nodes are always sorted by layers, 
     * and within a layer by 'place'
     */

    /* the desired place of each individual node in the current block */
    desired_place = e->fArray1;
    /* the desired place of each prefix of current block */
    prefix_desired_place = e->fArray2;
    /* the desired place of each suffix of current block */
    suffix_desired_place = e->fArray3;
    /* current block (nodes connected by active constraints) */
    ints_t block = {0};

    int *lev = gv_calloc(n, sizeof(int)); // level of each node
    for (int i = 0; i < n; i++) {
	if (i >= max_in_level) {
	    /* we are entering a new level */
	    level++;
	    if (level == num_levels) {
		/* last_level */
		max_in_level = n;
	    } else {
		max_in_level = levels[level];
	    }
	}
	node = ordering[i];
	lev[node] = level;
    }

    /* displacement of block's nodes from block's reference point */
    gap = e->fArray4;

    for (int counter = 0; counter < max_iterations && !converged; counter++) {
	converged = true;
	lower_bound = -1e9;	/* no lower bound for first level */
	for (left = 0; left < n; left = right) {
	    double max_movement;
	    double movement;
	    float prefix_des_place, suffix_des_place;
	    /* compute a block 'ordering[left]...ordering[right-1]' of 
	     * nodes connected with active constraints
	     */
	    cur_place = place[ordering[left]];
	    target_place = cur_place;
	    gap[ordering[left]] = 0;
	    for (right = left + 1; right < n; right++) {
		if (lev[right] > lev[right - 1]) {
		    /* we are entering a new level */
		    target_place += levels_gap;	/* note that 'levels_gap' may be negative */
		}
		node = ordering[right];
		    /* not sure if this is better than 'place[node]!=target_place' */
		    if (fabs(place[node] - target_place) > 1e-9) {
			break;
		    }
		gap[node] = place[node] - cur_place;
	    }

	    /* compute desired place of block's reference point according 
	     * to each node in the block:
	     */
	    for (int i = left; i < right; i++) {
		node = ordering[i];
		new_place_i = -b[node];
		lap_node = lap[node];
		for (int j = 0; j < n; j++) {
		    if (j == node) {
			continue;
		    }
		    new_place_i += lap_node[j] * place[j];
		}
		desired_place[node] =
		    new_place_i / (-lap_node[node]) - gap[node];
	    }

	    /* reorder block by levels, and within levels 
	     * by "relaxed" desired position
	     */
	    ints_clear(&block);
	    first_next_level = 0;
	    for (int i = left; i < right; i = first_next_level) {
		level = lev[ordering[i]];
		if (level == num_levels) {
		    /* last_level */
		    first_next_level = right;
		} else {
		    first_next_level = MIN(right, levels[level]);
		}

		/* First, collect all nodes with desired places smaller 
		 * than current place
		 */
		for (int j = i; j < first_next_level; j++) {
		    node = ordering[j];
		    if (desired_place[node] < cur_place) {
			ints_append(&block, node);
		    }
		}
		/* Second, collect all nodes with desired places equal 
		 * to current place
		 */
		for (int j = i; j < first_next_level; j++) {
		    node = ordering[j];
		    if (desired_place[node] == cur_place) {
			ints_append(&block, node);
		    }
		}
		/* Third, collect all nodes with desired places greater 
		 * than current place
		 */
		for (int j = i; j < first_next_level; j++) {
		    node = ordering[j];
		    if (desired_place[node] > cur_place) {
			ints_append(&block, node);
		    }
		}
	    }

	    /* loop through block and compute desired places of its prefixes */
	    des_place_block = 0;
	    block_deg = 0;
	    for (size_t i = 0; i < ints_size(&block); i++) {
		node = ints_get(&block, i);
		toBlockConnectivity = 0;
		lap_node = lap[node];
		for (size_t j = 0; j < i; j++) {
		    toBlockConnectivity -= lap_node[ints_get(&block, j)];
		}
		toBlockConnectivity *= 2;
		/* update block stats */
		des_place_block =
		    (block_deg * des_place_block +
		     (-lap_node[node]) * desired_place[node] +
		     toBlockConnectivity * cur_place) / (block_deg -
							 lap_node[node] +
							 toBlockConnectivity);
		prefix_desired_place[i] = des_place_block;
		block_deg += toBlockConnectivity - lap_node[node];
	    }

	    if (n >= 0 && ints_size(&block) == (size_t)n) {
		/* fix is needed since denominator was 0 in this case */
		prefix_desired_place[n - 1] = cur_place;	/* a "neutral" value */
	    }

	    /* loop through block and compute desired places of its suffixes */
	    des_place_block = 0;
	    block_deg = 0;
	    for (size_t i = ints_size(&block) - 1; i != SIZE_MAX; i--) {
		node = ints_get(&block, i);
		toBlockConnectivity = 0;
		lap_node = lap[node];
		for (size_t j = i + 1; j < ints_size(&block); j++) {
		    toBlockConnectivity -= lap_node[ints_get(&block, j)];
		}
		toBlockConnectivity *= 2;
		/* update block stats */
		des_place_block =
		    (block_deg * des_place_block +
		     (-lap_node[node]) * desired_place[node] +
		     toBlockConnectivity * cur_place) / (block_deg -
							 lap_node[node] +
							 toBlockConnectivity);
		suffix_desired_place[i] = des_place_block;
		block_deg += toBlockConnectivity - lap_node[node];
	    }

	    if (n >= 0 && ints_size(&block) == (size_t)n) {
		/* fix is needed since denominator was 0 in this case */
		suffix_desired_place[0] = cur_place;	/* a "neutral" value? */
	    }


	    /* now, find best place to split block */
	    size_t best_i = SIZE_MAX;
	    max_movement = 0;
	    for (size_t i = 0; i < ints_size(&block); i++) {
		suffix_des_place = suffix_desired_place[i];
		prefix_des_place =
		    i > 0 ? prefix_desired_place[i - 1] : suffix_des_place;
		/* limit moves to ensure that the prefix is placed before the suffix */
		if (suffix_des_place < prefix_des_place) {
		    if (suffix_des_place < cur_place) {
			if (prefix_des_place > cur_place) {
			    prefix_des_place = cur_place;
			}
			suffix_des_place = prefix_des_place;
		    } else if (prefix_des_place > cur_place) {
			prefix_des_place = suffix_des_place;
		    }
		}
		movement =
		    (float)(ints_size(&block) - i) * fabs(suffix_des_place - cur_place) +
		    (float)i * fabs(prefix_des_place - cur_place);
		if (movement > max_movement) {
		    max_movement = movement;
		    best_i = i;
		}
	    }
	    /* Actually move prefix and suffix */
	    if (best_i != SIZE_MAX) {
		suffix_des_place = suffix_desired_place[best_i];
		prefix_des_place =
		    best_i >
		    0 ? prefix_desired_place[best_i -
					     1] : suffix_des_place;

		/* compute right border of feasible move */
		if (right >= n) {
		    /* no nodes after current block */
		    upper_bound = 1e9;
		} else {
		    /* notice that we have to deduct 'gap[block[block_len-1]]'
		     * since all computations should be relative to 
		     * the block's reference point
		     */
		    if (lev[ordering[right]] > lev[ordering[right - 1]]) {
			upper_bound =
			    place[ordering[right]] - levels_gap -
			    gap[ints_get(&block, ints_size(&block) - 1)];
		    } else {
			upper_bound =
			    place[ordering[right]] -
			    gap[ints_get(&block, ints_size(&block) - 1)];
		    }
		}
		suffix_des_place = fminf(suffix_des_place, upper_bound);
		prefix_des_place = fmaxf(prefix_des_place, lower_bound);

		/* limit moves to ensure that the prefix is placed before the suffix */
		if (suffix_des_place < prefix_des_place) {
		    if (suffix_des_place < cur_place) {
			if (prefix_des_place > cur_place) {
			    prefix_des_place = cur_place;
			}
			suffix_des_place = prefix_des_place;
		    } else if (prefix_des_place > cur_place) {
			prefix_des_place = suffix_des_place;
		    }
		}

		/* move prefix: */
		for (size_t i = 0; i < best_i; i++) {
		    place[ints_get(&block, i)] = prefix_des_place + gap[ints_get(&block, i)];
		}
		/* move suffix: */
		for (size_t i = best_i; i < ints_size(&block); i++) {
		    place[ints_get(&block, i)] = suffix_des_place + gap[ints_get(&block, i)];
		}


		/* compute lower bound for next block */
		if (right < n
		    && lev[ordering[right]] > lev[ordering[right - 1]]) {
		    lower_bound = place[ints_get(&block, ints_size(&block) - 1)] + levels_gap;
		} else {
		    lower_bound = place[ints_get(&block, ints_size(&block) - 1)];
		}


		/* reorder 'ordering' to reflect change of places
		 * Note that it is enough to reorder the level where 
		 * the split was done
		 */
		for (int i = left; i < right; i++) {
		    ordering[i] = ints_get(&block, i - (size_t)left);
		}
		converged &= equals(prefix_des_place, cur_place)
		    && equals(suffix_des_place, cur_place);


	    } else {
		/* no movement */
		/* compute lower bound for next block */
		if (right < n
		    && lev[ordering[right]] > lev[ordering[right - 1]]) {
		    lower_bound = place[ints_get(&block, ints_size(&block) - 1)] + levels_gap;
		} else {
		    lower_bound = place[ints_get(&block, ints_size(&block) - 1)];
		}
	    }
	}
	orthog1f(n, place);	/* for numerical stability, keep ||place|| small */
    }
    free(lev);
    ints_free(&block);
}

void deleteCMajEnv(CMajEnv * e)
{
    free(e->A[0]);
    free(e->A);
    free(e->fArray1);
    free(e->fArray2);
    free(e->fArray3);
    free(e->fArray4);
    free(e);
}

CMajEnv *initConstrainedMajorization(float *packedMat, int n,
				     int *ordering, int *levels,
				     int num_levels)
{
    CMajEnv *e = gv_alloc(sizeof(CMajEnv));
    e->n = n;
    e->ordering = ordering;
    e->levels = levels;
    e->num_levels = num_levels;
    e->A = unpackMatrix(packedMat, n);
    e->fArray1 = gv_calloc(n, sizeof(float));
    e->fArray2 = gv_calloc(n, sizeof(float));
    e->fArray3 = gv_calloc(n, sizeof(float));
    e->fArray4 = gv_calloc(n, sizeof(float));
    return e;
}
#endif				/* DIGCOLA */
