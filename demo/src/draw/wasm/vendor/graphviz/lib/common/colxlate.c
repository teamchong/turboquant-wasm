/**
 * @file
 * @brief API colorprocs.h
 * @ingroup common_utils
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

#include "config.h"

#include <stdio.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <common/arith.h>
#include <common/color.h>
#include <common/colorprocs.h>
#include <common/colortbl.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/gv_ctype.h>
#include <util/gv_math.h>
#include <util/strcasecmp.h>
#include <util/unreachable.h>

static _Thread_local char *colorscheme;

static void hsv2rgb(double h, double s, double v,
			double *r, double *g, double *b)
{
    int i;
    double f, p, q, t;

    if (s <= 0.0) {		/* achromatic */
	*r = v;
	*g = v;
	*b = v;
    } else {
	if (h >= 1.0)
	    h = 0.0;
	h = 6.0 * h;
	i = (int) h;
	f = h - i;
	p = v * (1 - s);
	q = v * (1 - s * f);
	t = v * (1 - s * (1 - f));
	switch (i) {
	case 0:
	    *r = v;
	    *g = t;
	    *b = p;
	    break;
	case 1:
	    *r = q;
	    *g = v;
	    *b = p;
	    break;
	case 2:
	    *r = p;
	    *g = v;
	    *b = t;
	    break;
	case 3:
	    *r = p;
	    *g = q;
	    *b = v;
	    break;
	case 4:
	    *r = t;
	    *g = p;
	    *b = v;
	    break;
	case 5:
	    *r = v;
	    *g = p;
	    *b = q;
	    break;
	default:
	    UNREACHABLE();
	}
    }
}

static void rgb2hsv(double r, double g, double b,
		double *h, double *s, double *v)
{

    double rgbmin, rgbmax;
    double rc, bc, gc;
    double ht = 0.0, st = 0.0;

    rgbmin = fmin(r, fmin(g, b));
    rgbmax = fmax(r, fmax(g, b));

    if (rgbmax > 0.0)
	st = (rgbmax - rgbmin) / rgbmax;

    if (st > 0.0) {
	rc = (rgbmax - r) / (rgbmax - rgbmin);
	gc = (rgbmax - g) / (rgbmax - rgbmin);
	bc = (rgbmax - b) / (rgbmax - rgbmin);
	if (is_exactly_equal(r, rgbmax))
	    ht = bc - gc;
	else if (is_exactly_equal(g, rgbmax))
	    ht = 2 + rc - bc;
	else if (is_exactly_equal(b, rgbmax))
	    ht = 4 + gc - rc;
	ht = ht * 60.0;
	if (ht < 0.0)
	    ht += 360.0;
    }
    *h = ht / 360.0;
    *v = rgbmax;
    *s = st;
}

static int colorcmpf(const void *p0, const void *p1)
{
  return strcasecmp(p0, ((const hsvrgbacolor_t *)p1)->name);
}

/* fullColor:
 * Return "/prefix/str"
 */
static char *fullColor(agxbuf *xb, const char *prefix, const char *str) {
  agxbprint(xb, "/%s/%s", prefix, str);
  return agxbuse(xb);
}

/* resolveColor:
 * Resolve input color str allowing color scheme namespaces.
 *  0) "black" => "black" 
 *     "white" => "white" 
 *     "lightgrey" => "lightgrey" 
 *    NB: This is something of a hack due to the remaining codegen.
 *        Once these are gone, this case could be removed and all references
 *        to "black" could be replaced by "/X11/black".
 *  1) No initial / => 
 *          if colorscheme is defined and no "X11", return /colorscheme/str
 *          else return str
 *  2) One initial / => return str+1
 *  3) Two initial /'s =>
 *       a) If colorscheme is defined and not "X11", return /colorscheme/(str+2)
 *       b) else return (str+2)
 *  4) Two /'s, not both initial => return str.
 *
 * Note that 1), 2), and 3b) allow the default X11 color scheme. 
 *
 * In other words,
 *   xxx => /colorscheme/xxx     if colorscheme is defined and not "X11"
 *   xxx => xxx                  otherwise
 *   /xxx => xxx
 *   /X11/yyy => yyy
 *   /xxx/yyy => /xxx/yyy
 *   //yyy => /colorscheme/yyy   if colorscheme is defined and not "X11"
 *   //yyy => yyy                otherwise
 * 
 * At present, no other error checking is done. For example, 
 * yyy could be "". This will be caught later.
 */

#define DFLT_SCHEME "X11/"      /* Must have final '/' */
#define DFLT_SCHEME_LEN ((sizeof(DFLT_SCHEME)-1)/sizeof(char))
#define ISNONDFLT(s) ((s) && *(s) && strncasecmp(DFLT_SCHEME, s, DFLT_SCHEME_LEN-1))

static char *resolveColor(const char *str) {
    const char *s;

    if (!strcmp(str, "black")) return strdup(str);
    if (!strcmp(str, "white")) return strdup(str);
    if (!strcmp(str, "lightgrey")) return strdup(str);
    agxbuf xb = {0};
    if (*str == '/') {   /* if begins with '/' */
	const char *const c2 = str + 1; // second char
        const char *const ss = strchr(c2, '/'); // second slash
        if (ss != NULL) { // if has second '/'
	    if (*c2 == '/') {    /* if second '/' is second character */
		    /* Do not compare against final '/' */
		if (ISNONDFLT(colorscheme))
		    s = fullColor(&xb, colorscheme, c2+1);
		else
		    s = c2+1;
	    }
	    else if (strncasecmp(DFLT_SCHEME, c2, DFLT_SCHEME_LEN)) s = str;
	    else s = ss + 1;
	}
	else s = c2;
    }
    else if (ISNONDFLT(colorscheme)) s = fullColor(&xb, colorscheme, str);
    else s = str;
    char *on_heap = strdup(s);
    agxbfree(&xb);
    return on_heap;
}

int colorxlate(const char *str, gvcolor_t *color, color_type_t target_type) {
    char c;
    double H, S, V, A, R, G, B;
    unsigned int r, g, b;

    color->type = target_type;

    int rc = COLOR_OK;
    for (; *str == ' '; str++);	/* skip over any leading whitespace */
    const char *p = str;

    /* test for rgb value such as: "#ff0000"
       or rgba value such as "#ff000080" */
    unsigned a = 255; // default alpha channel value=opaque in case not supplied
    bool is_rgb = sscanf(p, "#%2x%2x%2x%2x", &r, &g, &b, &a) >= 3;
    if (!is_rgb) { // try 3 letter form
	is_rgb = strlen(p) == 4 && sscanf(p, "#%1x%1x%1x", &r, &g, &b) == 3;
	if (is_rgb) {
	    r |= r << 4;
	    g |= g << 4;
	    b |= b << 4;
	}
    }
    if (is_rgb) {
	switch (target_type) {
	case HSVA_DOUBLE:
	    R = (double) r / 255.0;
	    G = (double) g / 255.0;
	    B = (double) b / 255.0;
	    A = (double) a / 255.0;
	    rgb2hsv(R, G, B, &H, &S, &V);
	    color->u.HSVA[0] = H;
	    color->u.HSVA[1] = S;
	    color->u.HSVA[2] = V;
	    color->u.HSVA[3] = A;
	    break;
	case RGBA_BYTE:
	    color->u.rgba[0] = (unsigned char)r;
	    color->u.rgba[1] = (unsigned char)g;
	    color->u.rgba[2] = (unsigned char)b;
	    color->u.rgba[3] = (unsigned char)a;
	    break;
	case RGBA_WORD:
	    color->u.rrggbbaa[0] = (int)(r * 65535 / 255);
	    color->u.rrggbbaa[1] = (int)(g * 65535 / 255);
	    color->u.rrggbbaa[2] = (int)(b * 65535 / 255);
	    color->u.rrggbbaa[3] = (int)(a * 65535 / 255);
	    break;
	case RGBA_DOUBLE:
	    color->u.RGBA[0] = (double) r / 255.0;
	    color->u.RGBA[1] = (double) g / 255.0;
	    color->u.RGBA[2] = (double) b / 255.0;
	    color->u.RGBA[3] = (double) a / 255.0;
	    break;
	case COLOR_STRING:
	    break;
	case COLOR_INDEX:
	    break;
	default:
	    UNREACHABLE();
	}
	return rc;
    }

    /* test for hsv value such as: ".6,.5,.3" */
    if ((c = *p) == '.' || gv_isdigit(c)) {
	agxbuf canon = {0};
	while ((c = *p++)) {
	    agxbputc(&canon, c == ',' ? ' ' : c);
	}

	A = 1.0; // default
	if (sscanf(agxbuse(&canon), "%lf%lf%lf%lf", &H, &S, &V, &A) >= 3) {
	    /* clip to reasonable values */
	    H = fmax(fmin(H, 1.0), 0.0);
	    S = fmax(fmin(S, 1.0), 0.0);
	    V = fmax(fmin(V, 1.0), 0.0);
	    A = fmax(fmin(A, 1.0), 0.0);
	    switch (target_type) {
	    case HSVA_DOUBLE:
		color->u.HSVA[0] = H;
		color->u.HSVA[1] = S;
		color->u.HSVA[2] = V;
		color->u.HSVA[3] = A;
		break;
	    case RGBA_BYTE:
		hsv2rgb(H, S, V, &R, &G, &B);
		color->u.rgba[0] = (unsigned char)(R * 255);
		color->u.rgba[1] = (unsigned char)(G * 255);
		color->u.rgba[2] = (unsigned char)(B * 255);
		color->u.rgba[3] = (unsigned char)(A * 255);
		break;
	    case RGBA_WORD:
		hsv2rgb(H, S, V, &R, &G, &B);
		color->u.rrggbbaa[0] = (int) (R * 65535);
		color->u.rrggbbaa[1] = (int) (G * 65535);
		color->u.rrggbbaa[2] = (int) (B * 65535);
		color->u.rrggbbaa[3] = (int) (A * 65535);
		break;
	    case RGBA_DOUBLE:
		hsv2rgb(H, S, V, &R, &G, &B);
		color->u.RGBA[0] = R;
		color->u.RGBA[1] = G;
		color->u.RGBA[2] = B;
		color->u.RGBA[3] = A;
		break;
	    case COLOR_STRING:
		break;
	    case COLOR_INDEX:
		break;
	    default:
		UNREACHABLE();
	    }
	    agxbfree(&canon);
	    return rc;
	}
	agxbfree(&canon);
    }

    /* test for known color name (generic, not renderer specific known names) */
    char *name = resolveColor(str);
    if (!name)
	return COLOR_MALLOC_FAIL;
    const hsvrgbacolor_t *known = bsearch(name, color_lib,
                                          sizeof(color_lib) / sizeof(hsvrgbacolor_t),
                                          sizeof(color_lib[0]), colorcmpf);
    free(name);
    if (known != NULL) {
	switch (target_type) {
	case HSVA_DOUBLE:
	    color->u.HSVA[0] = (double)known->h / 255.0;
	    color->u.HSVA[1] = (double)known->s / 255.0;
	    color->u.HSVA[2] = (double)known->v / 255.0;
	    color->u.HSVA[3] = (double)known->a / 255.0;
	    break;
	case RGBA_BYTE:
	    color->u.rgba[0] = known->r;
	    color->u.rgba[1] = known->g;
	    color->u.rgba[2] = known->b;
	    color->u.rgba[3] = known->a;
	    break;
	case RGBA_WORD:
	    color->u.rrggbbaa[0] = known->r * 65535 / 255;
	    color->u.rrggbbaa[1] = known->g * 65535 / 255;
	    color->u.rrggbbaa[2] = known->b * 65535 / 255;
	    color->u.rrggbbaa[3] = known->a * 65535 / 255;
	    break;
	case RGBA_DOUBLE:
	    color->u.RGBA[0] = known->r / 255.0;
	    color->u.RGBA[1] = known->g / 255.0;
	    color->u.RGBA[2] = known->b / 255.0;
	    color->u.RGBA[3] = known->a / 255.0;
	    break;
	case COLOR_STRING:
	    break;
	case COLOR_INDEX:
	    break;
	default:
	    UNREACHABLE();
	}
	return rc;
    }

    /* if we're still here then we failed to find a valid color spec */
    rc = COLOR_UNKNOWN;
    switch (target_type) {
    case HSVA_DOUBLE:
	color->u.HSVA[0] = color->u.HSVA[1] = color->u.HSVA[2] = 0.0;
	color->u.HSVA[3] = 1.0; /* opaque */
	break;
    case RGBA_BYTE:
	color->u.rgba[0] = color->u.rgba[1] = color->u.rgba[2] = 0;
	color->u.rgba[3] = 255;	/* opaque */
	break;
    case RGBA_WORD:
	color->u.rrggbbaa[0] = color->u.rrggbbaa[1] = color->u.rrggbbaa[2] = 0;
	color->u.rrggbbaa[3] = 65535;	/* opaque */
	break;
    case RGBA_DOUBLE:
	color->u.RGBA[0] = color->u.RGBA[1] = color->u.RGBA[2] = 0.0;
	color->u.RGBA[3] = 1.0;	/* opaque */
	break;
    case COLOR_STRING:
	break;
    case COLOR_INDEX:
	break;
    default:
	UNREACHABLE();
    }
    return rc;
}

char *setColorScheme(const char *s) {
  char *previous = colorscheme;
  colorscheme = s == NULL ? NULL : gv_strdup(s);
  return previous;
}
