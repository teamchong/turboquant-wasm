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
#include <gvc/gvplugin_layout.h>

typedef enum { LAYOUT_CIRCO, } layout_type;

// Use `circo_layout` (the full plugin entry) rather than `circoLayout` (the
// raw positioning routine). `circo_layout` calls circo_init_graph first, then
// circoLayout, then spline_edges + dotneato_postprocess. Calling circoLayout
// directly leaves ND_alg uninitialized and crashes in edge routing.
extern void circo_layout(graph_t * g);
extern void circo_cleanup(graph_t * g);

static gvlayout_engine_t circogen_engine = {
    circo_layout,
    circo_cleanup,
};

static gvlayout_features_t circogen_features = {
    0,
};

gvplugin_installed_t gvlayout_circo_layout[] = {
    {LAYOUT_CIRCO, "circo", 0, &circogen_engine, &circogen_features},
    {0, NULL, 0, NULL, NULL}
};
