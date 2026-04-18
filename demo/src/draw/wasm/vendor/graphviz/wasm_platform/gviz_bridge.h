/**
 * C bridge for Zig FFI — exposes Graphviz functions with simple types
 * that Zig can handle (no bitfield structs).
 */
#ifndef GVIZ_BRIDGE_H
#define GVIZ_BRIDGE_H

#include <stddef.h>

/* Opaque handle types (Zig treats these as *anyopaque) */
typedef void* gviz_graph_t;
typedef void* gviz_node_t;
typedef void* gviz_edge_t;
typedef void* gviz_context_t;

/* Spline point */
typedef struct {
    double x;
    double y;
} gviz_point_t;

/* Spline data for an edge */
typedef struct {
    size_t point_count;
    const gviz_point_t *points;
    int has_start_point;
    gviz_point_t start_point;
    int has_end_point;
    gviz_point_t end_point;
} gviz_spline_t;

/* Bounding box */
typedef struct {
    double ll_x, ll_y;
    double ur_x, ur_y;
} gviz_bbox_t;

/* Initialize Graphviz with built-in dot layout plugin */
gviz_context_t gviz_context_new(void);
void gviz_context_free(gviz_context_t ctx);

/* Graph construction (programmatic, no DOT parser needed) */
gviz_graph_t gviz_graph_new(const char *name);
gviz_node_t gviz_add_node(gviz_graph_t g, const char *name);
gviz_edge_t gviz_add_edge(gviz_graph_t g, gviz_node_t tail, gviz_node_t head, const char *name);
void gviz_set_default_node_attr(gviz_graph_t g, const char *name, const char *value);
void gviz_set_graph_attr(gviz_graph_t g, const char *name, const char *value);
void gviz_set_node_attr(gviz_graph_t g, gviz_node_t n, const char *name, const char *value);
void gviz_set_default_edge_attr(gviz_graph_t g, const char *name, const char *value);
void gviz_set_edge_attr(gviz_graph_t g, gviz_edge_t e, const char *name, const char *value);

/* Subgraph (cluster) construction */
void gviz_set_subgraph_attr(gviz_graph_t g, const char *name, const char *value);
gviz_graph_t gviz_add_subgraph(gviz_graph_t g, const char *name);
gviz_node_t gviz_subgraph_add_node(gviz_graph_t subg, gviz_node_t n);

/* Graph operations */
gviz_graph_t gviz_parse_dot(const char *dot_string);
void gviz_graph_close(gviz_graph_t g);
int gviz_layout(gviz_context_t ctx, gviz_graph_t g);
int gviz_layout_nop2(gviz_context_t ctx, gviz_graph_t g);
int gviz_layout_circo(gviz_context_t ctx, gviz_graph_t g);
void gviz_free_layout(gviz_context_t ctx, gviz_graph_t g);

/* Node iteration */
gviz_node_t gviz_first_node(gviz_graph_t g);
gviz_node_t gviz_next_node(gviz_graph_t g, gviz_node_t n);
const char* gviz_node_name(gviz_node_t n);
void gviz_node_coord(gviz_node_t n, double *x, double *y);

/* Edge iteration */
gviz_edge_t gviz_first_out_edge(gviz_graph_t g, gviz_node_t n);
gviz_edge_t gviz_next_out_edge(gviz_graph_t g, gviz_edge_t e);
gviz_node_t gviz_edge_head(gviz_edge_t e);
gviz_node_t gviz_edge_tail(gviz_edge_t e);

/* Edge spline data */
int gviz_edge_spline(gviz_edge_t e, gviz_spline_t *out);

/* Edge label position (after layout) — returns 1 if label exists, 0 otherwise */
int gviz_edge_label_pos(gviz_edge_t e, double *x, double *y);

/* Graph bounding box */
gviz_bbox_t gviz_graph_bbox(gviz_graph_t g);

#endif /* GVIZ_BRIDGE_H */
