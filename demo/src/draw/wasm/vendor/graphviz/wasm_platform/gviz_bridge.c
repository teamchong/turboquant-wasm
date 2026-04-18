/**
 * C bridge implementation — wraps Graphviz internals into simple types
 * that Zig can import without @cImport bitfield issues.
 */

#include "config.h"
#include <cgraph/cgraph.h>
#include <gvc/gvc.h>
#include <common/types.h>
#include "gviz_bridge.h"

/* Plugin registration */
extern gvplugin_library_t gvplugin_dot_layout_LTX_library;
extern gvplugin_library_t gvplugin_neato_layout_LTX_library;
extern gvplugin_library_t gvplugin_circo_layout_LTX_library;

static lt_symlist_t builtin_plugins[] = {
    { "gvplugin_dot_layout_LTX_library", (void*)&gvplugin_dot_layout_LTX_library },
    { "gvplugin_neato_layout_LTX_library", (void*)&gvplugin_neato_layout_LTX_library },
    { "gvplugin_circo_layout_LTX_library", (void*)&gvplugin_circo_layout_LTX_library },
    { 0, 0 }
};

gviz_context_t gviz_context_new(void) {
    return (gviz_context_t)gvContextPlugins(builtin_plugins, 0);
}

void gviz_context_free(gviz_context_t ctx) {
    if (ctx) gvFreeContext((GVC_t*)ctx);
}

gviz_graph_t gviz_graph_new(const char *name) {
    return (gviz_graph_t)agopen((char*)name, Agdirected, NULL);
}

gviz_node_t gviz_add_node(gviz_graph_t g, const char *name) {
    return (gviz_node_t)agnode((Agraph_t*)g, (char*)name, 1);
}

gviz_edge_t gviz_add_edge(gviz_graph_t g, gviz_node_t tail, gviz_node_t head, const char *name) {
    return (gviz_edge_t)agedge((Agraph_t*)g, (Agnode_t*)tail, (Agnode_t*)head, (char*)name, 1);
}

void gviz_set_default_node_attr(gviz_graph_t g, const char *name, const char *value) {
    agattr((Agraph_t*)g, AGNODE, (char*)name, (char*)value);
}

void gviz_set_graph_attr(gviz_graph_t g, const char *name, const char *value) {
    agattr((Agraph_t*)g, AGRAPH, (char*)name, (char*)value);
    agsafeset((Agraph_t*)g, (char*)name, (char*)value, (char*)value);
}

void gviz_set_node_attr(gviz_graph_t g, gviz_node_t n, const char *name, const char *value) {
    (void)g;
    agsafeset((Agnode_t*)n, (char*)name, (char*)value, (char*)"");
}

void gviz_set_default_edge_attr(gviz_graph_t g, const char *name, const char *value) {
    agattr((Agraph_t*)g, AGEDGE, (char*)name, (char*)value);
}

void gviz_set_edge_attr(gviz_graph_t g, gviz_edge_t e, const char *name, const char *value) {
    (void)g;
    agsafeset((Agedge_t*)e, (char*)name, (char*)value, (char*)"");
}

void gviz_set_subgraph_attr(gviz_graph_t g, const char *name, const char *value) {
    /* Set attribute on subgraph only — unlike gviz_set_graph_attr, this does NOT
     * call agattr() which would set the default value for ALL subgraphs.
     * Uses agsafeset which creates the attribute if needed via the root graph
     * but sets the value only on the given subgraph. */
    agsafeset((Agraph_t*)g, (char*)name, (char*)value, (char*)"");
}

gviz_graph_t gviz_add_subgraph(gviz_graph_t g, const char *name) {
    return (gviz_graph_t)agsubg((Agraph_t*)g, (char*)name, 1);
}

gviz_node_t gviz_subgraph_add_node(gviz_graph_t subg, gviz_node_t n) {
    return (gviz_node_t)agsubnode((Agraph_t*)subg, (Agnode_t*)n, 1);
}

gviz_graph_t gviz_parse_dot(const char *dot_string) {
    return (gviz_graph_t)agmemread(dot_string);
}

void gviz_graph_close(gviz_graph_t g) {
    if (g) agclose((Agraph_t*)g);
}

int gviz_layout(gviz_context_t ctx, gviz_graph_t g) {
    return gvLayout((GVC_t*)ctx, (Agraph_t*)g, "dot");
}

int gviz_layout_nop2(gviz_context_t ctx, gviz_graph_t g) {
    return gvLayout((GVC_t*)ctx, (Agraph_t*)g, "nop2");
}

int gviz_layout_circo(gviz_context_t ctx, gviz_graph_t g) {
    return gvLayout((GVC_t*)ctx, (Agraph_t*)g, "circo");
}

void gviz_free_layout(gviz_context_t ctx, gviz_graph_t g) {
    gvFreeLayout((GVC_t*)ctx, (Agraph_t*)g);
}

gviz_node_t gviz_first_node(gviz_graph_t g) {
    return (gviz_node_t)agfstnode((Agraph_t*)g);
}

gviz_node_t gviz_next_node(gviz_graph_t g, gviz_node_t n) {
    return (gviz_node_t)agnxtnode((Agraph_t*)g, (Agnode_t*)n);
}

const char* gviz_node_name(gviz_node_t n) {
    return agnameof(n);
}

void gviz_node_coord(gviz_node_t n, double *x, double *y) {
    Agnode_t *node = (Agnode_t*)n;
    *x = ND_coord(node).x;
    *y = ND_coord(node).y;
}

gviz_edge_t gviz_first_out_edge(gviz_graph_t g, gviz_node_t n) {
    return (gviz_edge_t)agfstout((Agraph_t*)g, (Agnode_t*)n);
}

gviz_edge_t gviz_next_out_edge(gviz_graph_t g, gviz_edge_t e) {
    return (gviz_edge_t)agnxtout((Agraph_t*)g, (Agedge_t*)e);
}

gviz_node_t gviz_edge_head(gviz_edge_t e) {
    return (gviz_node_t)aghead((Agedge_t*)e);
}

gviz_node_t gviz_edge_tail(gviz_edge_t e) {
    return (gviz_node_t)agtail((Agedge_t*)e);
}

int gviz_edge_label_pos(gviz_edge_t e, double *x, double *y) {
    Agedge_t *edge = (Agedge_t*)e;
    textlabel_t *lp = ED_label(edge);
    if (!lp) return 0;
    *x = lp->pos.x;
    *y = lp->pos.y;
    return 1;
}

int gviz_edge_spline(gviz_edge_t e, gviz_spline_t *out) {
    Agedge_t *edge = (Agedge_t*)e;
    splines *spl = ED_spl(edge);
    if (!spl || spl->size == 0) return 0;

    bezier *bz = &spl->list[0];
    out->point_count = bz->size;
    out->points = (const gviz_point_t*)bz->list;
    out->has_start_point = (bz->sflag != 0);
    out->start_point.x = bz->sp.x;
    out->start_point.y = bz->sp.y;
    out->has_end_point = (bz->eflag != 0);
    out->end_point.x = bz->ep.x;
    out->end_point.y = bz->ep.y;
    return 1;
}

gviz_bbox_t gviz_graph_bbox(gviz_graph_t g) {
    Agraph_t *graph = (Agraph_t*)g;
    boxf bb = GD_bb(graph);
    gviz_bbox_t result;
    result.ll_x = bb.LL.x;
    result.ll_y = bb.LL.y;
    result.ur_x = bb.UR.x;
    result.ur_y = bb.UR.y;
    return result;
}

/* ── Functions referenced by Graphviz internals but not needed for layout-only use ──
 *
 * The dot layout engine calls into render/emit/HTML-label systems during
 * spline computation and label sizing. We provide these so the linker
 * is satisfied. They are no-ops because we only extract positions and
 * spline coordinates after layout, never render output.
 */

#include <gvc/gvcjob.h>
#include <string.h>

/* Include gvcproc.h for correct function prototypes */
#include <gvc/gvcint.h>
#include <gvc/gvcproc.h>
#include <gvc/gvplugin.h>

/* Render functions — called by arrows.c, shapes.c, labels.c, splines.c */
void gvrender_polygon(GVJ_t *j, pointf *a, size_t n, int f) { (void)j;(void)a;(void)n;(void)f; }
void gvrender_polyline(GVJ_t *j, pointf *a, size_t n) { (void)j;(void)a;(void)n; }
void gvrender_ellipse(GVJ_t *j, pointf *a, int f) { (void)j;(void)a;(void)f; }
void gvrender_beziercurve(GVJ_t *j, pointf *a, size_t n, int f) { (void)j;(void)a;(void)n;(void)f; }
void gvrender_box(GVJ_t *j, boxf b, int f) { (void)j;(void)b;(void)f; }
void gvrender_begin_anchor(GVJ_t *j, char *h, char *t, char *ta, char *id) { (void)j;(void)h;(void)t;(void)ta;(void)id; }
void gvrender_end_anchor(GVJ_t *j) { (void)j; }
void gvrender_begin_label(GVJ_t *j, label_type t) { (void)j;(void)t; }
void gvrender_end_label(GVJ_t *j) { (void)j; }
void gvrender_set_pencolor(GVJ_t *j, char *c) { (void)j;(void)c; }
void gvrender_set_fillcolor(GVJ_t *j, char *c) { (void)j;(void)c; }
void gvrender_set_gradient_vals(GVJ_t *j, char *s, int t, double a) { (void)j;(void)s;(void)t;(void)a; }
void gvrender_set_penwidth(GVJ_t *j, double w) { (void)j;(void)w; }
void gvrender_set_style(GVJ_t *j, char **s) { (void)j;(void)s; }
void gvrender_textspan(GVJ_t *j, pointf p, textspan_t *s) { (void)j;(void)p;(void)s; }
void gvrender_usershape(GVJ_t *j, char *n, pointf *a, size_t s, bool f, char *iu, char *ip) { (void)j;(void)n;(void)a;(void)s;(void)f;(void)iu;(void)ip; }

/* Text layout — fixed-width metrics matching SDK estimates so Graphviz
 * sizes edge labels correctly and avoids overlapping placements. */
bool gvtextlayout(GVC_t *gvc, textspan_t *span, char **fp) {
    (void)gvc;
    if (!span || !span->str || !span->font) return false;
    double fontsize = span->font->size;
    if (fontsize <= 0) fontsize = 14.0;
    size_t len = strlen(span->str);
    /* Match SDK: width ≈ len * 8, using fontsize * 0.6 per char */
    span->size.x = (double)len * fontsize * 0.6;
    if (span->size.x < 16.0) span->size.x = 16.0;
    span->size.y = fontsize * 1.2;
    span->yoffset_layout = fontsize;
    span->yoffset_centerline = 0.1 * fontsize;
    span->layout = NULL;
    span->free_layout = NULL;
    if (fp) *fp = NULL;
    return true;
}

/* User shape sizing — returns zero point (no image loading) */
point gvusershape_size(graph_t *g, char *name) { (void)g;(void)name; point p = {0,0}; return p; }

/* HTML label functions — not supported (no expat parser) */
int make_html_label(void *obj, textlabel_t *lp) { (void)obj; if(lp) lp->dimen.x = lp->dimen.y = 0; return 0; }
void free_html_label(textlabel_t *lp, int root) { (void)lp;(void)root; }
void emit_html_label(GVJ_t *j, htmllabel_t *lp, textlabel_t *tp) { (void)j;(void)lp;(void)tp; }
int html_port(node_t *n, char *pname, port *pp) { (void)n;(void)pname;(void)pp; return 0; }

/* Emit functions */
int emit_once(char *s) { (void)s; return 0; }
void emit_once_reset(void) {}

/* EPS functions */
void epsf_init(node_t *n) { (void)n; }
void epsf_free(node_t *n) { (void)n; }

/* Style parsing */
char **parse_style(char *s) { (void)s; return NULL; }

/* Color gradient stop parsing — signature: bool findStopColor(const char*, char*[2], double*) */
bool findStopColor(const char *colorlist, char *clrs[2], double *frac) {
    (void)colorlist;(void)clrs;(void)frac; return false;
}

/* Striped/wedged shapes — rendering-only, not used during layout positioning */
int stripedBox(GVJ_t *j, pointf *a, const char *c, int f) { (void)j;(void)a;(void)c;(void)f; return 0; }
int wedgedEllipse(GVJ_t *j, pointf *a, char *c) { (void)j;(void)a;(void)c; return 0; }

/* Spline bounding box update */
void update_bb_bz(boxf *bb, pointf *cp) { (void)bb;(void)cp; }

/* xdot initialization — returns void* per render.h */
void* init_xdot(Agraph_t *g) { (void)g; return NULL; }

/* Locale handling */
void gv_fixLocale(int set) { (void)set; }

/* Plugin config loading — install builtin plugins, skip filesystem scanning */

/* Forward declarations from common/textspan.c */
extern void textfont_dict_open(GVC_t *gvc);

static gvplugin_package_t * gvplugin_package_record(GVC_t *gvc,
    const char *package_path, const char *name) {
    gvplugin_package_t *package = calloc(1, sizeof(gvplugin_package_t));
    if (!package) return NULL;
    package->path = package_path ? strdup(package_path) : NULL;
    package->name = strdup(name);
    package->next = gvc->packages;
    gvc->packages = package;
    return package;
}

void gvconfig_plugin_install_from_library(GVC_t *gvc, char *package_path,
                                          gvplugin_library_t *library) {
    gvplugin_api_t *apis;
    gvplugin_installed_t *types;
    gvplugin_package_t *package;

    package = gvplugin_package_record(gvc, package_path, library->packagename);
    for (apis = library->apis; (types = apis->types); apis++) {
        for (int i = 0; types[i].type; i++) {
            gvplugin_install(gvc, apis->api, types[i].type,
                types[i].quality, package, &types[i]);
        }
    }
}

int gvtextlayout_select(GVC_t *gvc) { (void)gvc; return 0; }

void gvconfig(GVC_t *gvc, bool demand_loading) {
    (void)demand_loading;
    /* Install builtin plugins */
    if (gvc->common.builtins) {
        const lt_symlist_t *s;
        for (s = gvc->common.builtins; s->name; s++) {
            if (s->name[0] == 'g' && strstr(s->name, "_LTX_library"))
                gvconfig_plugin_install_from_library(gvc, NULL,
                    (gvplugin_library_t *)s->address);
        }
    }
    gvtextlayout_select(gvc);
    textfont_dict_open(gvc);
}

/* Graph reading from FILE* — not needed, we use agmemread */
Agraph_t *agread(void *chan, Agdisc_t *disc) { (void)chan;(void)disc; return NULL; }
Agraph_t *agconcat(Agraph_t *g, const char *filename, void *chan, Agdisc_t *disc) { (void)g;(void)filename;(void)chan;(void)disc; return NULL; }

/* Thread-safe stdio (WASI is single-threaded) — return int to match caller expectations */
int flockfile(void *f) { (void)f; return 0; }
int funlockfile(void *f) { (void)f; return 0; }

/* WASI has no process concept — there is always exactly one process.
 * Returning 1 is the documented WASI behavior. Used by neatoinit's random-
 * seed path (getpid XOR time), which circo's init chain pulls in. */
int getpid(void) { return 1; }

/* Implement clock() via WASI's clock_gettime(CLOCK_MONOTONIC). Zig's
 * wasi-libc exposes clock_gettime but not the legacy `clock()` wrapper.
 * WASI doesn't expose CLOCK_PROCESS_CPUTIME_ID, so we measure wall-clock
 * monotonic time — accurate enough for graphviz's internal profiling
 * prints (which we never read, but the symbol still needs to link). */
#include <time.h>
clock_t clock(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (clock_t)-1;
  return (clock_t)(ts.tv_sec * CLOCKS_PER_SEC + ts.tv_nsec / (1000000000L / CLOCKS_PER_SEC));
}

/* Required by WASI libc startup */
int main(void) {
    return 0;
}
