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

#include <stdio.h>

#include <gvc/gvplugin_layout.h>

// FIXME - globals.h is needed for Nop
#include <common/globals.h>

typedef enum {
  LAYOUT_NEATO,
  LAYOUT_FDP,
  LAYOUT_SFDP,
  LAYOUT_TWOPI,
  LAYOUT_CIRCO,
  LAYOUT_PATCHWORK,
  LAYOUT_CLUSTER,
  LAYOUT_NOP1,
  LAYOUT_NOP2,
} layout_type;

extern void neato_layout(graph_t *g);
extern void neato_cleanup(graph_t *g);

static void nop1_layout(graph_t *g) {
  Nop = 1;
  neato_layout(g);
  Nop = 0;
}

static void nop2_layout(graph_t *g) {
  Nop = 2;
  neato_layout(g);
  Nop = 0;
}

gvlayout_engine_t neatogen_engine = {
    neato_layout,
    neato_cleanup,
};

gvlayout_engine_t nop1gen_engine = {
    nop1_layout,
    neato_cleanup,
};

gvlayout_engine_t nop2gen_engine = {
    nop2_layout,
    neato_cleanup,
};

gvlayout_features_t neatogen_features = {
    0,
};

gvplugin_installed_t gvlayout_neato_types[] = {
    {LAYOUT_NEATO, "neato", 0, &neatogen_engine, &neatogen_features},
    {LAYOUT_NOP1, "nop", 0, &nop1gen_engine, &neatogen_features},
    {LAYOUT_NOP1, "nop1", 0, &nop1gen_engine, &neatogen_features},
    {LAYOUT_NOP2, "nop2", 0, &nop2gen_engine, &neatogen_features},
    {0, NULL, 0, NULL, NULL}};
