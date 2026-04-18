/**
 * Implementations for functions referenced by neato code paths we don't use.
 *
 * We only use neato in nop2 mode (pre-positioned edge routing).
 * The full neato layout path references constraint adjustment functions
 * (from constraint.c which depends on DEFINE_LIST) and xdotBB (from emit.c
 * which we don't compile). These functions are never called in nop2 mode
 * but the linker needs them resolved.
 *
 * Additionally, closest_pairs2graph (from closest.c) is referenced by pca.c
 * but never called in the nop2 path.
 */

#include "config.h"
#include <cgraph/cgraph.h>
#include <common/render.h>

/* Constraint-based overlap adjustment — used by full neato layout only */
int scAdjust(graph_t *g, int equal) {
    (void)g; (void)equal;
    return 0;
}

int cAdjust(graph_t *g, int mode) {
    (void)g; (void)mode;
    return 0;
}

/* xdot bounding box computation — used when reading xdot attributes */
boxf xdotBB(graph_t *g) {
    (void)g;
    boxf b = {{0, 0}, {0, 0}};
    return b;
}

/* Closest pairs for sparse stress model — used by full neato layout only */
typedef struct { int *edges; double *ewgts; } vtx_data;
void closest_pairs2graph(double *place, int n, int num_pairs, vtx_data **graph) {
    (void)place; (void)n; (void)num_pairs; (void)graph;
}
