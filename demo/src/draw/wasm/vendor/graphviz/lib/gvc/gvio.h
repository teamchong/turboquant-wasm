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

#include "gvcjob.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#ifdef GVC_EXPORTS
#define GVIO_API __declspec(dllexport)
#else
#define GVIO_API __declspec(dllimport)
#endif
#endif

#ifndef GVIO_API
#define GVIO_API /* nothing */
#endif

    GVIO_API size_t gvwrite (GVJ_t * job, const char *s, size_t len);
    GVIO_API int gvferror (FILE *stream);
    GVIO_API int gvputc(GVJ_t * job, int c);
    GVIO_API int gvputs(GVJ_t * job, const char *s);

    // `gvputs`, but XML-escape the input string
    GVIO_API int gvputs_xml(GVJ_t* job, const char *s);

    // `gvputs`, C-escaping '\' and non-ASCII bytes
    GVIO_API void gvputs_nonascii(GVJ_t* job, const char *s);

    GVIO_API int gvflush (GVJ_t * job);

// support for extra API misuse warnings if available
#ifdef __GNUC__
#define GV_PRINTF_LIKE(index, first) __attribute__((format(printf, index, first)))
#else
#define GV_PRINTF_LIKE(index, first) /* nothing */
#endif

    GVIO_API GV_PRINTF_LIKE(2, 3) void gvprintf(GVJ_t * job, const char *format, ...);

#undef GV_PRINTF_LIKE

    GVIO_API void gvprintdouble(GVJ_t * job, double num);
    GVIO_API void gvprintpointf(GVJ_t * job, pointf p);
    GVIO_API void gvprintpointflist(GVJ_t *job, pointf *p, size_t n);

#undef GVIO_API

#ifdef __cplusplus
}
#endif
