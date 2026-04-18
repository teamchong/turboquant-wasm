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

#include "color.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#ifdef GVC_EXPORTS
#define COLORPROCS_API __declspec(dllexport)
#else
#define COLORPROCS_API __declspec(dllimport)
#endif
#endif

#ifndef COLORPROCS_API
#define COLORPROCS_API /* nothing */
#endif

/// set current color scheme for resolving names
///
/// Callers should eventually free the returned pointer from this function.
///
/// @param s Color scheme to set
/// @return Previous color scheme
COLORPROCS_API char *setColorScheme(const char *s);

COLORPROCS_API int colorxlate(const char *str, gvcolor_t *color,
                              color_type_t target_type);

#undef COLORPROCS_API

#ifdef __cplusplus
}
#endif
