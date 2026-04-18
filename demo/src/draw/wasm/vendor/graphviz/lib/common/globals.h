/**
 * @file
 * @ingroup common_utils
 * @ingroup common_render
 */
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

#include <cgraph/cgraph.h>
#include <stdbool.h>
#include <stdlib.h>
#include <util/list.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#ifdef GVC_EXPORTS
#define GLOBALS_API __declspec(dllexport)
#else
#define GLOBALS_API __declspec(dllimport)
#endif
#endif

/// @cond
#ifndef GLOBALS_API
#define GLOBALS_API /* nothing */
#endif

#ifndef EXTERN
#define EXTERN extern
#endif
/// @endcond

typedef LIST(char *) show_boxes_t;

    GLOBALS_API EXTERN const char **Lib;		/* from command line */
    GLOBALS_API EXTERN char *Gvfilepath;  /* Per-process path of files allowed in image attributes (also ps libs) */
    GLOBALS_API EXTERN char *Gvimagepath; /* Per-graph path of files allowed in image attributes  (also ps libs) */

    GLOBALS_API EXTERN unsigned char Verbose;
    GLOBALS_API EXTERN bool Reduce;
    GLOBALS_API EXTERN char *HTTPServerEnVar;
    GLOBALS_API EXTERN int graphviz_errors;
    GLOBALS_API EXTERN int Nop;
    GLOBALS_API EXTERN double PSinputscale;
    GLOBALS_API extern show_boxes_t Show_boxes; // emit code for correct box coordinates
    GLOBALS_API EXTERN int CL_type;		/* NONE, LOCAL, GLOBAL */
    GLOBALS_API EXTERN bool Concentrate; /// if parallel edges should be merged
    GLOBALS_API EXTERN double Epsilon;	/* defined in input_graph */
    GLOBALS_API EXTERN int MaxIter;
    GLOBALS_API EXTERN unsigned short Ndim;
    GLOBALS_API EXTERN int State;		/* last finished phase */
    GLOBALS_API EXTERN int EdgeLabelsDone;	/* true if edge labels have been positioned */
    GLOBALS_API EXTERN double Initial_dist;
    GLOBALS_API EXTERN double Damping;
    GLOBALS_API EXTERN bool Y_invert; ///< invert y in dot & plain output
    GLOBALS_API EXTERN int GvExitOnUsage;   /* gvParseArgs() should exit on usage or error */

    GLOBALS_API EXTERN Agsym_t
	*G_ordering, *G_peripheries, *G_penwidth,
	*G_gradientangle, *G_margin;
    GLOBALS_API EXTERN Agsym_t
	*N_height, *N_width, *N_shape, *N_color, *N_fillcolor,
	*N_fontsize, *N_fontname, *N_fontcolor,
	*N_label, *N_xlabel, *N_nojustify, *N_style, *N_showboxes,
	*N_sides, *N_peripheries, *N_ordering, *N_orientation,
	*N_skew, *N_distortion, *N_fixed, *N_imagescale, *N_imagepos, *N_layer,
	*N_group, *N_comment, *N_vertices, *N_z,
	*N_penwidth, *N_gradientangle;
    GLOBALS_API EXTERN Agsym_t
	*E_weight, *E_minlen, *E_color, *E_fillcolor,
	*E_fontsize, *E_fontname, *E_fontcolor,
	*E_label, *E_xlabel, *E_dir, *E_style, *E_decorate,
	*E_showboxes, *E_arrowsz, *E_constr, *E_layer,
	*E_comment, *E_label_float,
	*E_samehead, *E_sametail,
	*E_headlabel, *E_taillabel,
	*E_labelfontsize, *E_labelfontname, *E_labelfontcolor,
	*E_labeldistance, *E_labelangle,
	*E_tailclip, *E_headclip,
	*E_penwidth;

    GLOBALS_API extern struct fdpParms_s* fdp_parms;

#undef EXTERN
#undef GLOBALS_API

#ifdef __cplusplus
}
#endif
