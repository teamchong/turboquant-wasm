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
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {MAX_COLOR = 1001};

extern bool knownColorScheme(const char *);
  /* return a list of rgb in hex form: "#ff0000,#00ff00,..." */
const char *color_palettes_get(const char *color_palette_name);

extern const float palette_pastel[1001][3];
extern const float palette_blue_to_yellow[1001][3];
extern const float palette_grey_to_red[1001][3];
extern const float palette_white_to_red[1001][3];
extern const float palette_grey[1001][3];
extern const float palette_primary[1001][3];
extern const float palette_sequential_singlehue_red[1001][3];
extern const float palette_sequential_singlehue_red_lighter[1001][3];
extern const float palette_adam_blend[1001][3];
extern const float palette_adam[11][3];

#ifdef __cplusplus
}
#endif
