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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <pathplan/vis.h>
#include <util/alloc.h>

typedef Ppoint_t ilcoord_t;

#ifdef DEBUG
static void printVconfig(vconfig_t * cp);
static void printVis(char *lbl, COORD * vis, int n);
static void printDad(int *vis, int n);
#endif

vconfig_t *Pobsopen(Ppoly_t ** obs, int n_obs)
{
    vconfig_t *rv;
    int poly_i, pt_i, i;
    int start, end;

    rv = malloc(sizeof(vconfig_t));
    if (!rv) {
	return NULL;
    }

    /* get storage */
    size_t n = 0;
    for (poly_i = 0; poly_i < n_obs; poly_i++) {
	n += obs[poly_i]->pn;
    }
    if (n > INT_MAX) { // will this overflow rv->N?
	free(rv);
	return NULL;
    }
    rv->P = calloc(n, sizeof(Ppoint_t));
    assert(n_obs >= 0);
    rv->start = calloc((size_t)n_obs + 1, sizeof(int));
    rv->next = calloc(n, sizeof(int));
    rv->prev = calloc(n, sizeof(int));
    rv->N = (int)n;
    rv->Npoly = n_obs;

    // bail out if any above allocations failed
    if (rv->start == NULL || (n > 0 && (rv->P == NULL ||
                                        rv->next == NULL ||
                                        rv->prev == NULL))) {
	free(rv->prev);
	free(rv->next);
	free(rv->start);
	free(rv->P);
	free(rv);
	return NULL;
    }

    /* build arrays */
    i = 0;
    for (poly_i = 0; poly_i < n_obs; poly_i++) {
	start = i;
	rv->start[poly_i] = start;
	assert(obs[poly_i]->pn <= INT_MAX);
	end = start + (int)obs[poly_i]->pn - 1;
	for (pt_i = 0; pt_i < (int)obs[poly_i]->pn; pt_i++) {
	    rv->P[i] = obs[poly_i]->ps[pt_i];
	    rv->next[i] = i + 1;
	    rv->prev[i] = i - 1;
	    i++;
	}
	rv->next[end] = start;
	rv->prev[start] = end;
    }
    rv->start[poly_i] = i;
    visibility(rv);
    return rv;
}

void Pobsclose(vconfig_t * config)
{
    free(config->P);
    free(config->start);
    free(config->next);
    free(config->prev);
    if (config->vis) {
	free(config->vis[0]);
	free(config->vis);
    }
    free(config);
}

void Pobspath(vconfig_t *config, Ppoint_t p0, int poly0, Ppoint_t p1, int poly1,
              Ppolyline_t *output_route) {
    int i, *dad;
    size_t opn;
    Ppoint_t *ops;
    COORD *ptvis0, *ptvis1;

    ptvis0 = ptVis(config, poly0, p0);
    ptvis1 = ptVis(config, poly1, p1);

    dad = makePath(p0, poly0, ptvis0, p1, poly1, ptvis1, config);

    opn = 1;
    for (i = dad[config->N]; i != config->N + 1; i = dad[i])
	opn++;
    opn++;
    ops = gv_calloc(opn, sizeof(Ppoint_t));

    size_t j = opn - 1;
    ops[j--] = p1;
    for (i = dad[config->N]; i != config->N + 1; i = dad[i])
	ops[j--] = config->P[i];
    ops[j] = p0;
    assert(j == 0);

#ifdef DEBUG
    printVconfig(config);
    printVis("p", ptvis0, config->N + 1);
    printVis("q", ptvis1, config->N + 1);
    printDad(dad, config->N + 1);
#endif

    free(ptvis0);
    free(ptvis1);

    output_route->pn = opn;
    output_route->ps = ops;
    free(dad);
}

#ifdef DEBUG
static void printVconfig(vconfig_t * cp)
{
    int i, j;
    int *next, *prev;
    Ppoint_t *pts;
    array2 arr;

    next = cp->next;
    prev = cp->prev;
    pts = cp->P;
    arr = cp->vis;

    printf("this next prev point\n");
    for (i = 0; i < cp->N; i++)
	printf("%3d  %3d  %3d    (%3g,%3g)\n", i, next[i], prev[i],
	       pts[i].x, pts[i].y);

    printf("\n\n");

    for (i = 0; i < cp->N; i++) {
	for (j = 0; j < cp->N; j++)
	    printf("%4.1f ", arr[i][j]);
	printf("\n");
    }
}

static void printVis(char *lbl, COORD * vis, int n)
{
    int i;

    printf("%s: ", lbl);
    for (i = 0; i < n; i++)
	printf("%4.1f ", vis[i]);
    printf("\n");
}

static void printDad(int *vis, int n)
{
    int i;

    printf("     ");
    for (i = 0; i < n; i++) {
	printf("%3d ", i);
    }
    printf("\n");
    printf("dad: ");
    for (i = 0; i < n; i++) {
	printf("%3d ", vis[i]);
    }
    printf("\n");
}
#endif
