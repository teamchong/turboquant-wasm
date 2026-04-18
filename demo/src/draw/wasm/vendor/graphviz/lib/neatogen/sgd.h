#pragma once

#include <stddef.h>
#include <util/bitarray.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct term_sgd {
    int i, j;
    float d, w;
} term_sgd;

typedef struct graph_sgd {
    size_t n; // number of nodes
    size_t *sources; // index of first edge in *targets for each node (length n+1)
    bitarray_t pinneds; // whether a node is fixed or not

    size_t *targets; // index of targets for each node (length sources[n])
    float *weights; // weights of edges (length sources[n])
} graph_sgd;

extern void sgd(graph_t *, int);

#ifdef __cplusplus
}
#endif
