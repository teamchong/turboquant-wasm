#include <assert.h>
#include <limits.h>
#include <neatogen/neato.h>
#include <neatogen/sgd.h>
#include <neatogen/dijkstra.h>
#include <neatogen/randomkit.h>
#include <neatogen/neatoprocs.h>
#include <math.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/bitarray.h>

static float calculate_stress(float *pos, term_sgd *terms, int n_terms) {
    float stress = 0;
    int ij;
    for (ij=0; ij<n_terms; ij++) {
        float dx = pos[2*terms[ij].i] - pos[2*terms[ij].j];
        float dy = pos[2*terms[ij].i+1] - pos[2*terms[ij].j+1];
        float r = hypotf(dx, dy) - terms[ij].d;
        stress += terms[ij].w * (r * r);
    }
    return stress;
}
// it is much faster to shuffle term rather than pointers to term, even though the swap is more expensive
static void fisheryates_shuffle(term_sgd *terms, int n_terms, rk_state *rstate) {
    int i;
    for (i=n_terms-1; i>=1; i--) {
        int j = rk_interval(i, rstate);

        term_sgd temp = terms[i];
        terms[i] = terms[j];
        terms[j] = temp;
    }
}

// graph_sgd data structure exists only to make dijkstras faster
static graph_sgd * extract_adjacency(graph_t *G, int model) {
    node_t *np;
    edge_t *ep;
    size_t n_nodes = 0, n_edges = 0;
    for (np = agfstnode(G); np; np = agnxtnode(G,np)) {
        assert(ND_id(np) == n_nodes);
        n_nodes++;
        for (ep = agfstedge(G, np); ep; ep = agnxtedge(G, ep, np)) {
            if (agtail(ep) != aghead(ep)) { // ignore self-loops and double edges
                n_edges++;
            }
        }
    }
    graph_sgd *graph = gv_alloc(sizeof(graph_sgd));
    graph->sources = gv_calloc(n_nodes + 1, sizeof(size_t));
    graph->pinneds = bitarray_new(n_nodes);
    graph->targets = gv_calloc(n_edges, sizeof(size_t));
    graph->weights = gv_calloc(n_edges, sizeof(float));

    graph->n = n_nodes;
    assert(n_edges <= INT_MAX);
    graph->sources[graph->n] = n_edges; // to make looping nice

    n_nodes = 0, n_edges = 0;
    for (np = agfstnode(G); np; np = agnxtnode(G,np)) {
        assert(n_edges <= INT_MAX);
        graph->sources[n_nodes] = n_edges;
        bitarray_set(&graph->pinneds, n_nodes, isFixed(np));
        for (ep = agfstedge(G, np); ep; ep = agnxtedge(G, ep, np)) {
            if (agtail(ep) == aghead(ep)) { // ignore self-loops and double edges
                continue;
            }
            node_t *target = (agtail(ep) == np) ? aghead(ep) : agtail(ep); // in case edge is reversed
            graph->targets[n_edges] = (size_t)ND_id(target);
            graph->weights[n_edges] = ED_dist(ep);
            assert(graph->weights[n_edges] > 0);
            n_edges++;
        }
        n_nodes++;
    }
    assert(n_nodes == graph->n);
    assert(n_edges <= INT_MAX);
    assert(n_edges == graph->sources[graph->n]);
    graph->sources[n_nodes] = n_edges;

    if (model == MODEL_SHORTPATH) {
        // do nothing
    } else if (model == MODEL_SUBSET) {
        // i,j,k refer to actual node indices, while x,y refer to edge indices in graph->targets
        // initialise to no neighbours
        bitarray_t neighbours_i = bitarray_new(graph->n);
        bitarray_t neighbours_j = bitarray_new(graph->n);
        for (size_t i = 0; i < graph->n; i++) {
            int deg_i = 0;
            for (size_t x = graph->sources[i]; x < graph->sources[i + 1]; x++) {
                size_t j = graph->targets[x];
                if (!bitarray_get(neighbours_i, j)) { // ignore multiedges
                    bitarray_set(&neighbours_i, j, true); // set up sort of hashset
                    deg_i++;
                }
            }
            for (size_t x = graph->sources[i]; x < graph->sources[i + 1]; x++) {
                size_t j = graph->targets[x];
                int intersect = 0;
                int deg_j = 0;
                for (size_t y = graph->sources[j]; y < graph->sources[j + 1];
                     y++) {
                    size_t k = graph->targets[y];
                    if (!bitarray_get(neighbours_j, k)) { // ignore multiedges
                        bitarray_set(&neighbours_j, k, true); // set up sort of hashset
                        deg_j++;
                        if (bitarray_get(neighbours_i, k)) {
                            intersect++;
                        }
                    }
                }
                graph->weights[x] = deg_i + deg_j - (2*intersect);
                assert(graph->weights[x] > 0);
                for (size_t y = graph->sources[j]; y < graph->sources[j + 1];
                     y++) {
                    size_t k = graph->targets[y];
                    bitarray_set(&neighbours_j, k, false); // reset sort of hashset
                }
            }
            for (size_t x = graph->sources[i]; x < graph->sources[i + 1]; x++) {
                size_t j = graph->targets[x];
                bitarray_set(&neighbours_i, j, false); // reset sort of hashset
            }
        }
        bitarray_reset(&neighbours_i);
        bitarray_reset(&neighbours_j);
    } else {
        // TODO: model == MODEL_MDS and MODEL_CIRCUIT
        assert(false); // mds and circuit model not supported
    }
    return graph;
}
static void free_adjacency(graph_sgd *graph) {
    free(graph->sources);
    bitarray_reset(&graph->pinneds);
    free(graph->targets);
    free(graph->weights);
    free(graph);
}


void sgd(graph_t *G, /* input graph */
        int model /* distance model */)
{
    if (model == MODEL_CIRCUIT) {
        agwarningf("circuit model not yet supported in Gmode=sgd, reverting to shortpath model\n");
        model = MODEL_SHORTPATH;
    }
    if (model == MODEL_MDS) {
        agwarningf("mds model not yet supported in Gmode=sgd, reverting to shortpath model\n");
        model = MODEL_SHORTPATH;
    }
    int n = agnnodes(G);

    if (Verbose) {
        fprintf(stderr, "calculating shortest paths and setting up stress terms:");
        start_timer();
    }
    // calculate how many terms will be needed as fixed nodes can be ignored
    int i, n_fixed = 0, n_terms = 0;
    for (i=0; i<n; i++) {
        if (!isFixed(GD_neato_nlist(G)[i])) {
            n_fixed++;
            n_terms += n-n_fixed;
        }
    }
    term_sgd *terms = gv_calloc(n_terms, sizeof(term_sgd));
    // calculate term values through shortest paths
    int offset = 0;
    graph_sgd *graph = extract_adjacency(G, model);
    for (i=0; i<n; i++) {
        if (!isFixed(GD_neato_nlist(G)[i])) {
            offset += dijkstra_sgd(graph, i, terms+offset);
        }
    }
    assert(offset == n_terms);
    free_adjacency(graph);
    if (Verbose) {
        fprintf(stderr, " %.2f sec\n", elapsed_sec());
    }

    // initialise annealing schedule
    float w_min = terms[0].w, w_max = terms[0].w;
    int ij;
    for (ij=1; ij<n_terms; ij++) {
        w_min = fminf(w_min, terms[ij].w);
        w_max = fmaxf(w_max, terms[ij].w);
    }
    // note: Epsilon is different from MODE_KK and MODE_MAJOR as it is a minimum step size rather than energy threshold
    //       MaxIter is also different as it is a fixed number of iterations rather than a maximum
    float eta_max = 1 / w_min;
    float eta_min = Epsilon / w_max;
    float lambda = log(eta_max/eta_min) / (MaxIter-1);

    // initialise starting positions (from neatoprocs)
    initial_positions(G, n);
    // copy initial positions and state into temporary space for speed
    float *pos = gv_calloc(2 * n, sizeof(float));
    bool *unfixed = gv_calloc(n, sizeof(bool));
    for (i=0; i<n; i++) {
        node_t *node = GD_neato_nlist(G)[i];
        pos[2*i] = ND_pos(node)[0];
        pos[2*i+1] = ND_pos(node)[1];
        unfixed[i] = !isFixed(node);
    }

    // perform optimisation
    if (Verbose) {
        fprintf(stderr, "solving model:");
        start_timer();
    }
    int t;
    rk_state rstate;
    rk_seed(0, &rstate); // TODO: get seed from graph
    for (t=0; t<MaxIter; t++) {
        fisheryates_shuffle(terms, n_terms, &rstate);
        float eta = eta_max * exp(-lambda * t);
        for (ij=0; ij<n_terms; ij++) {
            // cap step size
            float mu = eta * terms[ij].w;
            if (mu > 1)
                mu = 1;

            float dx = pos[2*terms[ij].i] - pos[2*terms[ij].j];
            float dy = pos[2*terms[ij].i+1] - pos[2*terms[ij].j+1];
            float mag = hypotf(dx, dy);

            float r = (mu * (mag-terms[ij].d)) / (2*mag);
            float r_x = r * dx;
            float r_y = r * dy;

            if (unfixed[terms[ij].i]) {
                pos[2*terms[ij].i] -= r_x;
                pos[2*terms[ij].i+1] -= r_y;
            }
            if (unfixed[terms[ij].j]) {
                pos[2*terms[ij].j] += r_x;
                pos[2*terms[ij].j+1] += r_y;
            }
        }
        if (Verbose) {
            fprintf(stderr, " %.3f", calculate_stress(pos, terms, n_terms));
        }
    }
    if (Verbose) {
        fprintf(stderr, "\nfinished in %.2f sec\n", elapsed_sec());
    }
    free(terms);

    // copy temporary positions back into graph_t
    for (i=0; i<n; i++) {
        node_t *node = GD_neato_nlist(G)[i];
        ND_pos(node)[0] = pos[2*i];
        ND_pos(node)[1] = pos[2*i+1];
    }
    free(pos);
    free(unfixed);
}
