/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <float.h>
#include <neatogen/neato.h>
#include <neatogen/dijkstra.h>
#include <neatogen/bfs.h>
#include <neatogen/pca.h>
#include <neatogen/matrix_ops.h>
#include <neatogen/conjgrad.h>
#include <neatogen/embed_graph.h>
#include <neatogen/kkutils.h>
#include <neatogen/stress.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <util/alloc.h>

// the terms in the stress energy are normalized by dᵢⱼ¯²

/* dimensionality of subspace; relevant 
 * when optimizing within subspace) 
 */
#define stress_pca_dim 50

 /* a structure used for storing sparse distance matrix */
typedef struct {
    size_t nedges;
    int *edges;
    DistType *edist;
    bool free_mem;
} dist_data;

static double compute_stressf(float **coords, float *lap, int dim, int n, int exp)
{
    /* compute the overall stress */

    int i, j, l, neighbor, count;
    double sum, dist, Dij;
    sum = 0;
    for (count = 0, i = 0; i < n - 1; i++) {
	count++;		/* skip diagonal entry */
	for (j = 1; j < n - i; j++, count++) {
	    dist = 0;
	    neighbor = i + j;
	    for (l = 0; l < dim; l++) {
		dist +=
		    (coords[l][i] - coords[l][neighbor]) * (coords[l][i] -
							    coords[l]
							    [neighbor]);
	    }
	    dist = sqrt(dist);
	    if (exp == 2) {
		Dij = 1.0 / sqrt(lap[count]);
		sum += (Dij - dist) * (Dij - dist) * lap[count];
	    } else {
		Dij = 1.0 / lap[count];
		sum += (Dij - dist) * (Dij - dist) * lap[count];
	    }
	}
    }

    return sum;
}

static double
compute_stress1(double **coords, dist_data * distances, int dim, int n, int exp)
{
    /* compute the overall stress */

    int i, l, node;
    double sum, dist, Dij;
    sum = 0;
    if (exp == 2) {
	for (i = 0; i < n; i++) {
	    for (size_t j = 0; j < distances[i].nedges; j++) {
		node = distances[i].edges[j];
		if (node <= i) {
		    continue;
		}
		dist = 0;
		for (l = 0; l < dim; l++) {
		    dist +=
			(coords[l][i] - coords[l][node]) * (coords[l][i] -
							    coords[l]
							    [node]);
		}
		dist = sqrt(dist);
		Dij = distances[i].edist[j];
		sum += (Dij - dist) * (Dij - dist) / (Dij * Dij);
	    }
	}
    } else {
	for (i = 0; i < n; i++) {
	    for (size_t j = 0; j < distances[i].nedges; j++) {
		node = distances[i].edges[j];
		if (node <= i) {
		    continue;
		}
		dist = 0;
		for (l = 0; l < dim; l++) {
		    dist +=
			(coords[l][i] - coords[l][node]) * (coords[l][i] -
							    coords[l]
							    [node]);
		}
		dist = sqrt(dist);
		Dij = distances[i].edist[j];
		sum += (Dij - dist) * (Dij - dist) / Dij;
	    }
	}
    }

    return sum;
}

/* initLayout:
 * Initialize node coordinates. If the node already has
 * a position, use it.
 * Return true if some node is fixed.
 */
int initLayout(int n, int dim, double **coords, node_t **nodes) {
    node_t *np;
    double *xp;
    double *yp;
    double *pt;
    int i, d;
    int pinned = 0;

    xp = coords[0];
    yp = coords[1];
    for (i = 0; i < n; i++) {
	np = nodes[i];
	if (hasPos(np)) {
	    pt = ND_pos(np);
	    *xp++ = *pt++;
	    *yp++ = *pt++;
	    if (dim > 2) {
		for (d = 2; d < dim; d++)
		    coords[d][i] = *pt++;
	    }
	    if (isFixed(np))
		pinned = 1;
	} else {
	    *xp++ = drand48();
	    *yp++ = drand48();
	    if (dim > 2) {
		for (d = 2; d < dim; d++)
		    coords[d][i] = drand48();
	    }
	}
    }

    for (d = 0; d < dim; d++)
	orthog1(n, coords[d]);

    return pinned;
}

float *circuitModel(vtx_data * graph, int nG)
{
    int i, j, rv, count;
    float *Dij = gv_calloc(nG * (nG + 1) / 2, sizeof(float));
    double **Gm;
    double **Gm_inv;

    Gm = new_array(nG, nG, 0.0);
    Gm_inv = new_array(nG, nG, 0.0);

    /* set non-diagonal entries */
    if (graph->ewgts) {
	for (i = 0; i < nG; i++) {
	    for (size_t e = 1; e < graph[i].nedges; e++) {
		j = graph[i].edges[e];
		/* conductance is 1/resistance */
		Gm[i][j] = Gm[j][i] = -1.0 / graph[i].ewgts[e];	/* negate */
	    }
	}
    } else {
	for (i = 0; i < nG; i++) {
	    for (size_t e = 1; e < graph[i].nedges; e++) {
		j = graph[i].edges[e];
		/* conductance is 1/resistance */
		Gm[i][j] = Gm[j][i] = -1.0;	/* ewgts are all 1 */
	    }
	}
    }

    rv = solveCircuit(nG, Gm, Gm_inv);

    if (rv) {
	float v;
	count = 0;
	for (i = 0; i < nG; i++) {
	    for (j = i; j < nG; j++) {
		if (i == j)
		    v = 0.0;
		else
		    v = (float) (Gm_inv[i][i] + Gm_inv[j][j] -
				 2.0 * Gm_inv[i][j]);
		Dij[count++] = v;
	    }
	}
    } else {
	free(Dij);
	Dij = NULL;
    }
    free_array(Gm);
    free_array(Gm_inv);
    return Dij;
}

/* sparse_stress_subspace_majorization_kD:
 * Optimization of the stress function using sparse distance matrix, within a vector-space
 * Fastest and least accurate method
 *
 * NOTE: We use integral shortest path values here, assuming
 * this is only to get an initial layout. In general, if edge lengths
 * are involved, we may end up with 0 length edges. 
 */
static int sparse_stress_subspace_majorization_kD(vtx_data * graph,	/* Input graph in sparse representation */
						  int n,	/* Number of nodes */
						  double **coords,	/* coordinates of nodes (output layout)  */
						  int dim,	/* dimemsionality of layout */
						  int smart_ini,	/* smart initialization */
						  int exp,	/* scale exponent */
						  int reweight_graph,	/* difference model */
						  int n_iterations,	/* max #iterations */
						  int num_centers	/* #pivots in sparse distance matrix  */
    )
{
    int iterations;		/* output: number of iteration of the process */

    double conj_tol = tolerance_cg;	/* tolerance of Conjugate Gradient */

	/*************************************************
	** Computation of pivot-based, sparse, subspace-restricted **
	** k-D  stress minimization by majorization                **    
	*************************************************/

    int i, k, node;

	/*************************************************
	** First compute the subspace in which we optimize     **
	** The subspace is  the high-dimensional embedding     **
	*************************************************/

    int subspace_dim = MIN(stress_pca_dim, n);	/* overall dimensionality of subspace */
    double **subspace = gv_calloc(subspace_dim, sizeof(double *));
    double *d_storage = gv_calloc(subspace_dim * n, sizeof(double));
    int num_centers_local;
    DistType **full_coords;
    /* if i is a pivot than CenterIndex[i] is its index, otherwise CenterIndex[i]= -1 */
    int *invCenterIndex;	/* list the pivot nodes  */
    float *old_weights;
    /* this matrix stores the distance between  each node and each "center" */
    DistType **Dij;
    /* this vector stores the distances of each node to the selected "centers" */
    DistType max_dist;
    DistType *storage;
    int *visited_nodes;
    dist_data *distances;
    int available_space;
    int *storage1 = NULL;
    DistType *storage2 = NULL;
    int num_visited_nodes;
    int num_neighbors;
    int index;
    DistType *dist_list;
    vtx_data *lap;
    int *edges;
    float *ewgts;
    double degree;
    double **directions;
    float **tmp_mat;
    float **matrix;
    double dist_ij;
    double *b;
    double *b_restricted;
    double L_ij;
    double old_stress, new_stress;
    bool converged;

    for (i = 0; i < subspace_dim; i++) {
	subspace[i] = d_storage + i * n;
    }

    /* compute PHDE: */
    num_centers_local = MIN(n, MAX(2 * subspace_dim, 50));
    full_coords = NULL;
    /* High dimensional embedding */
    embed_graph(graph, n, num_centers_local, &full_coords, reweight_graph);
    /* Centering coordinates */
    center_coordinate(full_coords, n, num_centers_local);
    /* PCA */
    PCA_alloc(full_coords, num_centers_local, n, subspace, subspace_dim);

    free(full_coords[0]);
    free(full_coords);

	/*************************************************
	** Compute the sparse-shortest-distances matrix 'distances' **
	*************************************************/

    int *CenterIndex = gv_calloc(n, sizeof(int));
    for (i = 0; i < n; i++) {
	CenterIndex[i] = -1;
    }
    invCenterIndex = NULL;

    old_weights = graph[0].ewgts;

    if (reweight_graph) {
	/* weight graph to separate high-degree nodes */
	/* in the future, perform slower Dijkstra-based computation */
	compute_new_weights(graph, n);
    }

    /* compute sparse distance matrix */
    /* first select 'num_centers' pivots from which we compute distance */
    /* to all other nodes */

    Dij = NULL;
    DistType *dist = gv_calloc(n, sizeof(DistType));
    if (num_centers == 0) {	/* no pivots, skip pivots-to-nodes distance calculation */
	goto after_pivots_selection;
    }

    invCenterIndex = gv_calloc(num_centers, sizeof(int));

    storage = gv_calloc(n * num_centers, sizeof(DistType));
    Dij = gv_calloc(num_centers, sizeof(DistType *));
    for (i = 0; i < num_centers; i++)
	Dij[i] = storage + i * n;

    /* select 'num_centers' pivots that are uniformaly spread over the graph */

    /* the first pivots is selected randomly */
    node = rand() % n;
    CenterIndex[node] = 0;
    invCenterIndex[0] = node;

    if (reweight_graph) {
	dijkstra(node, graph, n, Dij[0]);
    } else {
	bfs(node, graph, n, Dij[0]);
    }

    /* find the most distant node from first pivot */
    max_dist = 0;
    for (i = 0; i < n; i++) {
	dist[i] = Dij[0][i];
	if (dist[i] > max_dist) {
	    node = i;
	    max_dist = dist[i];
	}
    }
    /* select other dim-1 nodes as pivots */
    for (i = 1; i < num_centers; i++) {
	CenterIndex[node] = i;
	invCenterIndex[i] = node;
	if (reweight_graph) {
	    dijkstra(node, graph, n, Dij[i]);
	} else {
	    bfs(node, graph, n, Dij[i]);
	}
	max_dist = 0;
	for (int j = 0; j < n; j++) {
	    dist[j] = MIN(dist[j], Dij[i][j]);
	    if (dist[j] > max_dist
		|| (dist[j] == max_dist && rand() % (j + 1) == 0)) {
		node = j;
		max_dist = dist[j];
	    }
	}
    }

  after_pivots_selection:

    /* Construct a sparse distance matrix 'distances' */

    /* initialize dist to -1, important for 'bfs_bounded(..)' */
    for (i = 0; i < n; i++) {
	dist[i] = -1;
    }

    visited_nodes = gv_calloc(n, sizeof(int));
    distances = gv_calloc(n, sizeof(dist_data));
    available_space = 0;
    size_t nedges = 0;
    for (i = 0; i < n; i++) {
	if (CenterIndex[i] >= 0) {	/* a pivot node */
	    distances[i].edges = gv_calloc(n - 1, sizeof(int));
	    distances[i].edist = gv_calloc(n - 1, sizeof(DistType));
	    distances[i].nedges = (size_t)n - 1;
	    nedges += (size_t)n - 1;
	    distances[i].free_mem = true;
	    index = CenterIndex[i];
	    for (int j = 0; j < i; j++) {
		distances[i].edges[j] = j;
		distances[i].edist[j] = Dij[index][j];
	    }
	    for (int j = i + 1; j < n; j++) {
		distances[i].edges[j - 1] = j;
		distances[i].edist[j - 1] = Dij[index][j];
	    }
	    continue;
	}

	/* a non pivot node */

	num_visited_nodes = 0;
	num_neighbors = num_visited_nodes + num_centers;
	if (num_neighbors > available_space) {
	    available_space = n;
	    storage1 = gv_calloc(available_space, sizeof(int));
	    storage2 = gv_calloc(available_space, sizeof(DistType));
	    distances[i].free_mem = true;
	} else {
	    distances[i].free_mem = false;
	}
	distances[i].edges = storage1;
	distances[i].edist = storage2;
	distances[i].nedges = (size_t)num_neighbors;
	nedges += (size_t)num_neighbors;
	for (int j = 0; j < num_visited_nodes; j++) {
	    storage1[j] = visited_nodes[j];
	    storage2[j] = dist[visited_nodes[j]];
	    dist[visited_nodes[j]] = -1;
	}
	/* add all pivots: */
	for (int j = num_visited_nodes; j < num_neighbors; j++) {
	    index = j - num_visited_nodes;
	    storage1[j] = invCenterIndex[index];
	    storage2[j] = Dij[index][i];
	}

	storage1 += num_neighbors;
	storage2 += num_neighbors;
	available_space -= num_neighbors;
    }

    free(dist);
    free(visited_nodes);

    if (Dij != NULL) {
	free(Dij[0]);
	free(Dij);
    }

	/*************************************************
	** Laplacian computation **
	*************************************************/

    lap = gv_calloc(n, sizeof(vtx_data));
    edges = gv_calloc(nedges + n, sizeof(int));
    ewgts = gv_calloc(nedges + n, sizeof(float));
    for (i = 0; i < n; i++) {
	lap[i].edges = edges;
	lap[i].ewgts = ewgts;
	lap[i].nedges = distances[i].nedges + 1;	/*add the self loop */
	dist_list = distances[i].edist - 1;	/* '-1' since edist[0] goes for number '1' entry in the lap */
	degree = 0;
	if (exp == 2) {
	    for (size_t j = 1; j < lap[i].nedges; j++) {
		edges[j] = distances[i].edges[j - 1];
		ewgts[j] = (float) -1.0 / ((float) dist_list[j] * (float) dist_list[j]);	/* cast to float to prevent overflow */
		degree -= ewgts[j];
	    }
	} else {
	    for (size_t j = 1; j < lap[i].nedges; j++) {
		edges[j] = distances[i].edges[j - 1];
		ewgts[j] = -1.0f / (float) dist_list[j];
		degree -= ewgts[j];
	    }
	}
	edges[0] = i;
	ewgts[0] = (float) degree;
	edges += lap[i].nedges;
	ewgts += lap[i].nedges;
    }

	/*************************************************
	** initialize direction vectors  **
	** to get an initial layout       **
	*************************************************/

    /* the layout is subspace*directions */
    directions = gv_calloc(dim, sizeof(double *));
    directions[0] = gv_calloc(dim * subspace_dim, sizeof(double));
    for (i = 1; i < dim; i++) {
	directions[i] = directions[0] + i * subspace_dim;
    }

    if (smart_ini) {
	/* smart initialization */
	for (k = 0; k < dim; k++) {
	    for (i = 0; i < subspace_dim; i++) {
		directions[k][i] = 0;
	    }
	}
	if (dim != 2) {
	    /* use the first vectors in the eigenspace */
	    /* each direction points to its "principal axes" */
	    for (k = 0; k < dim; k++) {
		directions[k][k] = 1;
	    }
	} else {
	    /* for the frequent 2-D case we prefer iterative-PCA over PCA */
	    /* Note that we don't want to mix the Lap's eigenspace with the HDE */
	    /* in the computation since they have different scales */

	    directions[0][0] = 1;	/* first pca projection vector */
	    if (!iterativePCA_1D(subspace, subspace_dim, n, directions[1])) {
		for (k = 0; k < subspace_dim; k++) {
		    directions[1][k] = 0;
		}
		directions[1][1] = 1;
	    }
	}

    } else {
	/* random initialization */
	for (k = 0; k < dim; k++) {
	    for (i = 0; i < subspace_dim; i++) {
		directions[k][i] = (double) rand() / RAND_MAX;
	    }
	}
    }

    /* compute initial k-D layout */

    for (k = 0; k < dim; k++) {
	right_mult_with_vector_transpose(subspace, n, subspace_dim,
					 directions[k], coords[k]);
    }

	/*************************************************
	** compute restriction of the laplacian to subspace: **	
	*************************************************/

    tmp_mat = NULL;
    matrix = NULL;
    mult_sparse_dense_mat_transpose(lap, subspace, n, subspace_dim,
				    &tmp_mat);
    mult_dense_mat(subspace, tmp_mat, subspace_dim, n, subspace_dim,
		   &matrix);
    free(tmp_mat[0]);
    free(tmp_mat);

	/*************************************************
	** Layout optimization  **
	*************************************************/

    b = gv_calloc(n, sizeof(double));
    b_restricted = gv_calloc(subspace_dim, sizeof(double));
    old_stress = compute_stress1(coords, distances, dim, n, exp);
    for (converged = false, iterations = 0;
	 iterations < n_iterations && !converged; iterations++) {

	/* Axis-by-axis optimization: */
	for (k = 0; k < dim; k++) {
	    /* compute the vector b */
	    /* multiply on-the-fly with distance-based laplacian */
	    /* (for saving storage we don't construct this Lap explicitly) */
	    for (i = 0; i < n; i++) {
		degree = 0;
		b[i] = 0;
		dist_list = distances[i].edist - 1;
		edges = lap[i].edges;
		ewgts = lap[i].ewgts;
		for (size_t j = 1; j < lap[i].nedges; j++) {
		    node = edges[j];
		    dist_ij = distance_kD(coords, dim, i, node);
		    if (dist_ij > 1e-30) {	/* skip zero distances */
			L_ij = -ewgts[j] * dist_list[j] / dist_ij;	/* L_ij=w_{ij}*d_{ij}/dist_{ij} */
			degree -= L_ij;
			b[i] += L_ij * coords[k][node];
		    }
		}
		b[i] += degree * coords[k][i];
	    }
	    right_mult_with_vector_d(subspace, subspace_dim, n, b,
				     b_restricted);
	    if (conjugate_gradient_f(matrix, directions[k], b_restricted,
				 subspace_dim, conj_tol, subspace_dim,
				 false)) {
		iterations = -1;
		goto finish0;
	    }
	    right_mult_with_vector_transpose(subspace, n, subspace_dim,
					     directions[k], coords[k]);
	}

	if (iterations % 2 == 0) { // check for convergence each two iterations
	    new_stress = compute_stress1(coords, distances, dim, n, exp);
	    converged = fabs(new_stress - old_stress) / (new_stress + 1e-10) < Epsilon;
	    old_stress = new_stress;
	}
    }
finish0:
    free(b_restricted);
    free(b);

    if (reweight_graph) {
	restore_old_weights(graph, n, old_weights);
    }

    for (i = 0; i < n; i++) {
	if (distances[i].free_mem) {
	    free(distances[i].edges);
	    free(distances[i].edist);
	}
    }

    free(distances);
    free(lap[0].edges);
    free(lap[0].ewgts);
    free(lap);
    free(CenterIndex);
    free(invCenterIndex);
    free(directions[0]);
    free(directions);
    if (matrix != NULL) {
	free(matrix[0]);
	free(matrix);
    }
    free(subspace[0]);
    free(subspace);

    return iterations;
}

/* compute_weighted_apsp_packed:
 * Edge lengths can be any float > 0
 */
static float *compute_weighted_apsp_packed(vtx_data * graph, int n)
{
    int i, j, count;
    float *Dij = gv_calloc(n * (n + 1) / 2, sizeof(float));

    float *Di = gv_calloc(n, sizeof(float));

    count = 0;
    for (i = 0; i < n; i++) {
	dijkstra_f(i, graph, n, Di);
	for (j = i; j < n; j++) {
	    Dij[count++] = Di[j];
	}
    }
    free(Di);
    return Dij;
}


/* mdsModel:
 * Update matrix with actual edge lengths
 */
float *mdsModel(vtx_data * graph, int nG)
{
    int i, j;
    float *Dij;
    int shift = 0;
    double delta = 0.0;

    if (graph->ewgts == NULL)
	return 0;

    /* first, compute shortest paths to fill in non-edges */
    Dij = compute_weighted_apsp_packed(graph, nG);

    /* then, replace edge entries will user-supplied len */
    for (i = 0; i < nG; i++) {
	shift += i;
	for (size_t e = 1; e < graph[i].nedges; e++) {
	    j = graph[i].edges[e];
	    if (j < i)
		continue;
	    delta += fabsf(Dij[i * nG + j - shift] - graph[i].ewgts[e]);
	    Dij[i * nG + j - shift] = graph[i].ewgts[e];
	}
    }
    if (Verbose) {
	fprintf(stderr, "mdsModel: delta = %f\n", delta);
    }
    return Dij;
}

/* compute_apsp_packed:
 * Assumes integral weights > 0.
 */
float *compute_apsp_packed(vtx_data * graph, int n)
{
    int i, j, count;
    float *Dij = gv_calloc(n * (n + 1) / 2, sizeof(float));

    DistType *Di = gv_calloc(n, sizeof(DistType));

    count = 0;
    for (i = 0; i < n; i++) {
	bfs(i, graph, n, Di);
	for (j = i; j < n; j++) {
	    Dij[count++] = (float)Di[j];
	}
    }
    free(Di);
    return Dij;
}

float *compute_apsp_artificial_weights_packed(vtx_data *graph, int n) {
    /* compute all-pairs-shortest-path-length while weighting the graph */
    /* so high-degree nodes are distantly located */

    float *Dij;
    int i;
    float *old_weights = graph[0].ewgts;
    size_t nedges = 0;
    size_t deg_i, deg_j;
    int neighbor;

    for (i = 0; i < n; i++) {
	nedges += graph[i].nedges;
    }

    float *weights = gv_calloc(nedges, sizeof(float));
    int *vtx_vec = gv_calloc(n, sizeof(int));

    if (graph->ewgts) {
	for (i = 0; i < n; i++) {
	    fill_neighbors_vec_unweighted(graph, i, vtx_vec);
	    deg_i = graph[i].nedges - 1;
	    for (size_t j = 1; j <= deg_i; j++) {
		neighbor = graph[i].edges[j];
		deg_j = graph[neighbor].nedges - 1;
		weights[j] = fmaxf((float)(deg_i + deg_j -
			 2 * common_neighbors(graph, neighbor, vtx_vec)), graph[i].ewgts[j]);
	    }
	    empty_neighbors_vec(graph, i, vtx_vec);
	    graph[i].ewgts = weights;
	    weights += graph[i].nedges;
	}
	Dij = compute_weighted_apsp_packed(graph, n);
    } else {
	for (i = 0; i < n; i++) {
	    graph[i].ewgts = weights;
	    fill_neighbors_vec_unweighted(graph, i, vtx_vec);
	    deg_i = graph[i].nedges - 1;
	    for (size_t j = 1; j <= deg_i; j++) {
		neighbor = graph[i].edges[j];
		deg_j = graph[neighbor].nedges - 1;
		weights[j] =
		    (float)(deg_i + deg_j - 2 * common_neighbors(graph, neighbor, vtx_vec));
	    }
	    empty_neighbors_vec(graph, i, vtx_vec);
	    weights += graph[i].nedges;
	}
	Dij = compute_apsp_packed(graph, n);
    }

    free(vtx_vec);
    free(graph[0].ewgts);
    graph[0].ewgts = NULL;
    if (old_weights != NULL) {
	for (i = 0; i < n; i++) {
	    graph[i].ewgts = old_weights;
	    old_weights += graph[i].nedges;
	}
    }
    return Dij;
}

#if defined(DEBUG) && DEBUG > 1
static void dumpMatrix(float *Dij, int n)
{
    int i, j, count = 0;
    for (i = 0; i < n; i++) {
	for (j = i; j < n; j++) {
	    fprintf(stderr, "%.02f  ", Dij[count++]);
	}
	fputs("\n", stderr);
    }
}
#endif

/* Accumulator type for diagonal of Laplacian. Needs to be as large
 * as possible. Use long double; configure to double if necessary.
 */
#define DegType long double

/* stress_majorization_kD_mkernel:
 * At present, if any nodes have pos set, smart_ini is false.
 */
int stress_majorization_kD_mkernel(vtx_data * graph,	/* Input graph in sparse representation */
				   int n,	/* Number of nodes */
				   double **d_coords,	/* coordinates of nodes (output layout) */
				   node_t ** nodes,	/* original nodes */
				   int dim,	/* dimemsionality of layout */
				   int opts,    /* options */
				   int model,	/* model */
				   int maxi	/* max iterations */
    )
{
    int iterations;		/* output: number of iteration of the process */

    double conj_tol = tolerance_cg;	/* tolerance of Conjugate Gradient */
    float *Dij = NULL;
    int i, j, k;
    float **coords = NULL;
    float *f_storage = NULL;
    float constant_term;
    int count;
    DegType degree;
    int lap_length;
    float *lap2 = NULL;
    DegType *degrees = NULL;
    int step;
    float val;
    double old_stress, new_stress;
    bool converged;
    float **b = NULL;
    float *tmp_coords = NULL;
    float *dist_accumulator = NULL;
    float *lap1 = NULL;
    int smart_ini = opts & opt_smart_init;
    int exp = opts & opt_exp_flag;
    int len;
    int havePinned;		/* some node is pinned */

	/*************************************************
	** Computation of full, dense, unrestricted k-D ** 
	** stress minimization by majorization          **    
	*************************************************/

	/****************************************************
	** Compute the all-pairs-shortest-distances matrix **
	****************************************************/

    if (maxi < 0)
	return 0;

    if (Verbose)
	start_timer();

    if (model == MODEL_SUBSET) {
	/* weight graph to separate high-degree nodes */
	/* and perform slower Dijkstra-based computation */
	if (Verbose)
	    fprintf(stderr, "Calculating subset model");
	Dij = compute_apsp_artificial_weights_packed(graph, n);
    } else if (model == MODEL_CIRCUIT) {
	Dij = circuitModel(graph, n);
	if (!Dij) {
	    agwarningf(
		  "graph is disconnected. Hence, the circuit model\n");
	    agerr(AGPREV,
		  "is undefined. Reverting to the shortest path model.\n");
	}
    } else if (model == MODEL_MDS) {
	if (Verbose)
	    fprintf(stderr, "Calculating MDS model");
	Dij = mdsModel(graph, n);
    }
    if (!Dij) {
	if (Verbose)
	    fprintf(stderr, "Calculating shortest paths");
	if (graph->ewgts)
	    Dij = compute_weighted_apsp_packed(graph, n);
	else
	    Dij = compute_apsp_packed(graph, n);
    }

    if (Verbose) {
	fprintf(stderr, ": %.2f sec\n", elapsed_sec());
	fprintf(stderr, "Setting initial positions");
	start_timer();
    }

	/**************************
	** Layout initialization **
	**************************/

    if (smart_ini && n > 1) {
	havePinned = 0;
	/* optimize layout quickly within subspace */
	/* perform at most 50 iterations within 30-D subspace to 
	   get an estimate */
	if (sparse_stress_subspace_majorization_kD(graph, n,
					       d_coords, dim, smart_ini, exp,
					       model == MODEL_SUBSET, 50,
					       num_pivots_stress) < 0) {
	    iterations = -1;
	    goto finish1;
	}

	for (i = 0; i < dim; i++) {
	    /* for numerical stability, scale down layout */
	    double max = 1;
	    for (j = 0; j < n; j++) {
		if (fabs(d_coords[i][j]) > max) {
		    max = fabs(d_coords[i][j]);
		}
	    }
	    for (j = 0; j < n; j++) {
		d_coords[i][j] /= max;
	    }
	    /* add small random noise */
	    for (j = 0; j < n; j++) {
		d_coords[i][j] += 1e-6 * (drand48() - 0.5);
	    }
	    orthog1(n, d_coords[i]);
	}
    } else {
	havePinned = initLayout(n, dim, d_coords, nodes);
    }
    if (Verbose)
	fprintf(stderr, ": %.2f sec", elapsed_sec());
    if (n == 1 || maxi == 0) {
	free(Dij);
	return 0;
    }

    if (Verbose) {
	fprintf(stderr, ": %.2f sec\n", elapsed_sec());
	fprintf(stderr, "Setting up stress function");
	start_timer();
    }
    coords = gv_calloc(dim, sizeof(float *));
    f_storage = gv_calloc(dim * n, sizeof(float));
    for (i = 0; i < dim; i++) {
	coords[i] = f_storage + i * n;
	for (j = 0; j < n; j++) {
	    coords[i][j] = (float)d_coords[i][j];
	}
    }

    /* compute constant term in stress sum */
    /* which is \sum_{i<j} w_{ij}d_{ij}^2 */
    assert(exp == 1 || exp == 2);
    constant_term = (float)n * (n - 1) / 2;

	/**************************
	** Laplacian computation **
	**************************/

    lap_length = n * (n + 1) / 2;
    lap2 = Dij;
    if (exp == 2) {
    square_vec(lap_length, lap2);
    }
    /* compute off-diagonal entries */
    invert_vec(lap_length, lap2);

    /* compute diagonal entries */
    count = 0;
    degrees = gv_calloc(n, sizeof(DegType));
    for (i = 0; i < n - 1; i++) {
	degree = 0;
	count++;		/* skip main diag entry */
	for (j = 1; j < n - i; j++, count++) {
	    val = lap2[count];
	    degree += val;
	    degrees[i + j] -= val;
	}
	degrees[i] -= degree;
    }
    for (step = n, count = 0, i = 0; i < n; i++, count += step, step--) {
	lap2[count] = degrees[i];
    }

	/*************************
	** Layout optimization  **
	*************************/

    b = gv_calloc(dim, sizeof(float *));
    b[0] = gv_calloc(dim * n, sizeof(float));
    for (k = 1; k < dim; k++) {
	b[k] = b[0] + k * n;
    }

    tmp_coords = gv_calloc(n, sizeof(float));
    dist_accumulator = gv_calloc(n, sizeof(float));
    lap1 = gv_calloc(lap_length, sizeof(float));


    old_stress = DBL_MAX; // at least one iteration
    if (Verbose) {
	fprintf(stderr, ": %.2f sec\n", elapsed_sec());
	fprintf(stderr, "Solving model: ");
	start_timer();
    }

    for (converged = false, iterations = 0;
	 iterations < maxi && !converged; iterations++) {

	/* First, construct Laplacian of 1/(d_ij*|p_i-p_j|)  */
	/* set_vector_val(n, 0, degrees); */
	memset(degrees, 0, n * sizeof(DegType));
	if (exp == 2) {
	    sqrt_vecf(lap_length, lap2, lap1);
	}
	for (count = 0, i = 0; i < n - 1; i++) {
	    len = n - i - 1;
	    /* init 'dist_accumulator' with zeros */
	    set_vector_valf(len, 0, dist_accumulator);

	    /* put into 'dist_accumulator' all squared distances between 'i' and 'i'+1,...,'n'-1 */
	    for (k = 0; k < dim; k++) {
		size_t x;
		for (x = 0; x < (size_t)len; ++x) {
		    float tmp = coords[k][i] + -1.0f * (coords[k] + i + 1)[x];
		    dist_accumulator[x] += tmp * tmp;
		}
	    }

	    /* convert to 1/d_{ij} */
	    invert_sqrt_vec(len, dist_accumulator);
	    /* detect overflows */
	    for (j = 0; j < len; j++) {
		if (dist_accumulator[j] >= FLT_MAX || dist_accumulator[j] < 0) {
		    dist_accumulator[j] = 0;
		}
	    }

	    count++;		/* save place for the main diagonal entry */
	    degree = 0;
	    if (exp == 2) {
		for (j = 0; j < len; j++, count++) {
		    val = lap1[count] *= dist_accumulator[j];
		    degree += val;
		    degrees[i + j + 1] -= val;
		}
	    } else {
		for (j = 0; j < len; j++, count++) {
		    val = lap1[count] = dist_accumulator[j];
		    degree += val;
		    degrees[i + j + 1] -= val;
		}
	    }
	    degrees[i] -= degree;
	}
	for (step = n, count = 0, i = 0; i < n; i++, count += step, step--) {
	    lap1[count] = degrees[i];
	}

	/* Now compute b[] */
	for (k = 0; k < dim; k++) {
	    /* b[k] := lap1*coords[k] */
	    right_mult_with_vector_ff(lap1, n, coords[k], b[k]);
	}


	/* compute new stress  */
	/* remember that the Laplacians are negated, so we subtract instead of add and vice versa */
	new_stress = 0;
	for (k = 0; k < dim; k++) {
	    new_stress += vectors_inner_productf(n, coords[k], b[k]);
	}
	new_stress *= 2;
	new_stress += constant_term;	/* only after mult by 2 */
	for (k = 0; k < dim; k++) {
	    right_mult_with_vector_ff(lap2, n, coords[k], tmp_coords);
	    new_stress -= vectors_inner_productf(n, coords[k], tmp_coords);
	}
	/* Invariant: old_stress > 0. In theory, old_stress >= new_stress
	 * but we use fabs in case of numerical error.
	 */
	{
	    double diff = old_stress - new_stress;
	    double change = fabs(diff);
	    converged = change / old_stress < Epsilon || new_stress < Epsilon;
	}
	old_stress = new_stress;

	for (k = 0; k < dim; k++) {
	    node_t *np;
	    if (havePinned) {
		copy_vectorf(n, coords[k], tmp_coords);
		if (conjugate_gradient_mkernel(lap2, tmp_coords, b[k], n,
					   conj_tol, n) < 0) {
		    iterations = -1;
		    goto finish1;
		}
		for (i = 0; i < n; i++) {
		    np = nodes[i];
		    if (!isFixed(np))
			coords[k][i] = tmp_coords[i];
		}
	    } else {
		if (conjugate_gradient_mkernel(lap2, coords[k], b[k], n,
					   conj_tol, n) < 0) {
		    iterations = -1;
		    goto finish1;
		}
	    }
	}
	if (Verbose && iterations % 5 == 0) {
	    fprintf(stderr, "%.3f ", new_stress);
	    if ((iterations + 5) % 50 == 0)
		fprintf(stderr, "\n");
	}
    }
    if (Verbose) {
	fprintf(stderr, "\nfinal e = %f %d iterations %.2f sec\n",
		compute_stressf(coords, lap2, dim, n, exp),
		iterations, elapsed_sec());
    }

    for (i = 0; i < dim; i++) {
	for (j = 0; j < n; j++) {
	    d_coords[i][j] = coords[i][j];
	}
    }
finish1:
    free(f_storage);
    free(coords);

    free(lap2);
    if (b) {
	free(b[0]);
	free(b);
    }
    free(tmp_coords);
    free(dist_accumulator);
    free(degrees);
    free(lap1);
    return iterations;
}
