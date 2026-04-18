/// @file
/// @ingroup common_utils
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

#include "config.h"

#include <stdbool.h>
#include <stddef.h>
#include <util/agxbuf.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#ifdef GVC_EXPORTS
#define UTILS_API __declspec(dllexport)
#else
#define UTILS_API __declspec(dllimport)
#endif
#endif

#ifndef UTILS_API
#define UTILS_API /* nothing */
#endif

UTILS_API pointf Bezier(const pointf *, double, pointf *, pointf *);

/// @return Computed ymin + ymax
UTILS_API double attach_attrs_and_arrows(graph_t *, bool *, bool *);

UTILS_API void write_plain(GVJ_t *job, graph_t *g, void *f, bool extend);

/// @param yOff Displacement to apply to `y`
UTILS_API double yDir(double y, double yOff);

/// translate string for emitting into a PostScript file
///
/// Internally, strings are always UTF-8. If `chset` is `CHAR_LATIN1`, we know
/// all of the values can be represented by Latin-1; if `chset` is `CHAR_UTF8`,
/// we use the string as is; otherwise, we test to see if the string is ASCII,
/// Latin-1 or non-Latin, and translate to Latin-l if possible.
///
/// The caller must free the returned string.
///
/// @param s String to translate
/// @param chset Character set
/// @return Translated string
UTILS_API char *ps_string(char *s, int chset);

UTILS_API char *strdup_and_subst_obj(char *str, void *obj);
UTILS_API void epsf_emit_body(GVJ_t *job, usershape_t *us);
UTILS_API void epsf_define(GVJ_t *job);
UTILS_API void undoClusterEdges(graph_t *g);
UTILS_API Dt_t *mkClustMap(Agraph_t *g);
UTILS_API Agraph_t *findCluster(Dt_t *map, char *name);
UTILS_API attrsym_t *safe_dcl(graph_t *g, int obj_kind, char *name,
                              char *defaultValue);

UTILS_API int late_int(void *obj, Agsym_t *attr, int defaultValue, int minimum);
UTILS_API double late_double(void *obj, Agsym_t *attr, double defaultValue,
                             double minimum);
UTILS_API char *late_nnstring(void *obj, Agsym_t *attr, char *defaultValue);
UTILS_API char *late_string(void *obj, Agsym_t *attr, char *defaultValue);
UTILS_API bool late_bool(void *obj, Agsym_t *attr, bool defaultValue);
UTILS_API double get_inputscale(graph_t *g);

// routines for supporting “union-find”, a.k.a. “disjoint-set forest”
// https://en.wikipedia.org/wiki/Disjoint-set_data_structure
UTILS_API Agnode_t *UF_find(Agnode_t *);
UTILS_API Agnode_t *UF_union(Agnode_t *, Agnode_t *);
UTILS_API void UF_singleton(Agnode_t *);
UTILS_API void UF_setname(Agnode_t *, Agnode_t *);

UTILS_API const char *safefile(const char *filename);

UTILS_API bool mapBool(const char *p, bool defaultValue);
UTILS_API bool mapbool(const char *p);
UTILS_API int maptoken(char *, char **, int *);

UTILS_API bool findStopColor(const char *colorlist, char *clrs[2],
                             double *frac);
UTILS_API int test_toggle(void);

UTILS_API void common_init_node(node_t *n);
UTILS_API void common_init_edge(edge_t *e);

UTILS_API void updateBB(graph_t *g, textlabel_t *lp);
UTILS_API void compute_bb(Agraph_t *);
UTILS_API boxf polyBB(polygon_t *poly);
UTILS_API bool overlap_node(node_t *n, boxf b);
UTILS_API bool overlap_label(textlabel_t *lp, boxf b);
UTILS_API bool overlap_edge(edge_t *e, boxf b);

UTILS_API void get_gradient_points(pointf *A, pointf *G, size_t n, double angle,
                                   int flags);

UTILS_API void processClusterEdges(graph_t *g);

UTILS_API char *latin1ToUTF8(char *);
UTILS_API char *htmlEntityUTF8(char *, graph_t *g);
UTILS_API char *utf8ToLatin1(char *ins);
UTILS_API char *scanEntity(char *t, agxbuf *xb);

UTILS_API pointf dotneato_closest(splines *spl, pointf p);

UTILS_API Agsym_t *setAttr(graph_t *, void *, char *name, char *value,
                           Agsym_t *);
UTILS_API void setEdgeType(graph_t *g, int defaultValue);
UTILS_API bool is_a_cluster(Agraph_t *g);

/* from postproc.c */
UTILS_API void gv_nodesize(Agnode_t *n, bool flip);

#ifndef HAVE_DRAND48
UTILS_API double drand48(void);
#endif

/* from timing.c */
UTILS_API void start_timer(void);
UTILS_API double elapsed_sec(void);

/* from psusershape.c */
UTILS_API void cat_libfile(GVJ_t *job, const char **arglib,
                           const char **stdlib);

#undef UTILS_API

#ifdef __cplusplus
}
#endif
