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

#include "pathplan.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#ifdef PATHPLAN_EXPORTS
#define PATHUTIL_API __declspec(dllexport)
#else
#define PATHUTIL_API __declspec(dllimport)
#endif
#endif

#ifndef PATHUTIL_API
#define PATHUTIL_API /* nothing */
#endif
typedef double COORD;
PATHUTIL_API COORD area2(Ppoint_t, Ppoint_t, Ppoint_t);
PATHUTIL_API int wind(Ppoint_t a, Ppoint_t b, Ppoint_t c);
PATHUTIL_API COORD dist2(Ppoint_t, Ppoint_t);

PATHUTIL_API bool in_poly(const Ppoly_t poly, Ppoint_t q);

#undef PATHUTIL_API
#ifdef __cplusplus
}
#endif
