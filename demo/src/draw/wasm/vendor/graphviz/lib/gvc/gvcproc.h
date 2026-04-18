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

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/* these are intended to be private entry points - see gvc.h for the public ones */

/* configuration */

    char *gvconfig_libdir(GVC_t * gvc);
    void gvconfig(GVC_t * gvc, bool rescan);
    char *gvhostname(void);

/* plugins */

    bool gvplugin_install(GVC_t *gvc, api_t api,
		    const char *typestr, int quality, gvplugin_package_t *package,
		    gvplugin_installed_t * typeptr);
/// load a plugin
///
/// @param debug An optional sink to print debug comments to explaining failures
gvplugin_available_t *gvplugin_load(GVC_t *gvc, api_t api, const char *type,
                                    FILE *debug);
    gvplugin_library_t *gvplugin_library_load(GVC_t *gvc, const char *pathname);
    api_t gvplugin_api(const char *str);
    char * gvplugin_api_name(api_t api);
    void gvplugin_write_status(GVC_t * gvc);
    char *gvplugin_list(GVC_t * gvc, api_t api, const char *str);

    Agraph_t * gvplugin_graph(GVC_t * gvc);

/* job */

    void gvjobs_output_filename(GVC_t * gvc, const char *name);
    bool gvjobs_output_langname(GVC_t * gvc, const char *name);
    GVJ_t *gvjobs_first(GVC_t * gvc);
    GVJ_t *gvjobs_next(GVC_t * gvc);
    void gvjobs_delete(GVC_t * gvc);

/* emit */
    void gvemit_graph(GVC_t * gvc, graph_t * g);

/* textlayout */

    int gvtextlayout_select(GVC_t * gvc);
    bool gvtextlayout(GVC_t *gvc, textspan_t *span, char **fontpath);

/* loadimage */
    void gvloadimage(GVJ_t *job, usershape_t *us, boxf b, bool filled, const char *target);
    
/* usershapes */
    point gvusershape_size_dpi(usershape_t *us, pointf dpi);
    point gvusershape_size(graph_t *g, char *name);
    usershape_t *gvusershape_find(const char *name);

/* device */
    int gvdevice_initialize(GVJ_t * job);
    void gvdevice_format(GVJ_t * job);
    void gvdevice_finalize(GVJ_t * job);

/* render */

#ifdef GVDLL
#ifdef GVC_EXPORTS
    __declspec(dllexport) pointf gvrender_ptf(GVJ_t *job, pointf p);
#else
    __declspec(dllimport) pointf gvrender_ptf(GVJ_t *job, pointf p);
#endif
#else
    pointf gvrender_ptf(GVJ_t *job, pointf p);
#endif
    pointf *gvrender_ptf_A(GVJ_t *job, pointf *af, pointf *AF, size_t n);

    int gvrender_begin_job(GVJ_t * job);
    void gvrender_end_job(GVJ_t * job);
    int gvrender_select(GVJ_t * job, const char *lang);
    int gvrender_features(GVJ_t * job);
    void gvrender_begin_graph(GVJ_t *job);
    void gvrender_end_graph(GVJ_t * job);
    void gvrender_begin_page(GVJ_t * job);
    void gvrender_end_page(GVJ_t * job);
    void gvrender_begin_layer(GVJ_t * job);
    void gvrender_end_layer(GVJ_t * job);
    void gvrender_begin_cluster(GVJ_t *job);
    void gvrender_end_cluster(GVJ_t *job);
    void gvrender_begin_nodes(GVJ_t * job);
    void gvrender_end_nodes(GVJ_t * job);
    void gvrender_begin_edges(GVJ_t * job);
    void gvrender_end_edges(GVJ_t * job);
    void gvrender_begin_node(GVJ_t *job);
    void gvrender_end_node(GVJ_t * job);
    void gvrender_begin_edge(GVJ_t *job);
    void gvrender_end_edge(GVJ_t * job);
    void gvrender_begin_anchor(GVJ_t * job,
		char *href, char *tooltip, char *target, char *id);
    void gvrender_end_anchor(GVJ_t * job);
    void gvrender_begin_label(GVJ_t * job, label_type type);
    void gvrender_end_label(GVJ_t * job);
    void gvrender_textspan(GVJ_t * job, pointf p, textspan_t * span);
    void gvrender_set_pencolor(GVJ_t * job, char *name);
    void gvrender_set_penwidth(GVJ_t * job, double penwidth);
    void gvrender_set_fillcolor(GVJ_t * job, char *name);
    void gvrender_set_gradient_vals(GVJ_t *job, char *stopcolor, int angle,
                                    double frac);

    void gvrender_set_style(GVJ_t * job, char **s);
    void gvrender_ellipse(GVJ_t * job, pointf * AF, int filled);
    void gvrender_polygon(GVJ_t *job, pointf *af, size_t n, int filled);
    void gvrender_box(GVJ_t * job, boxf BF, int filled);
    void gvrender_beziercurve(GVJ_t *job, pointf *AF, size_t n, int filled);
    void gvrender_polyline(GVJ_t * job, pointf * AF, size_t n);
    void gvrender_comment(GVJ_t * job, char *str);
    void gvrender_usershape(GVJ_t *job, char *name, pointf *AF, size_t n,
                            bool filled, char *imagescale, char *imagepos);

/* layout */

    int gvlayout_select(GVC_t * gvc, const char *str);
