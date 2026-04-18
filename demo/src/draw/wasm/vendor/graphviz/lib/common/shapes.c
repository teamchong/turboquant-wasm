/// @file
/// @brief [node shapes](https://graphviz.org/doc/info/shapes.html)
/// @ingroup common_render
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

#include <assert.h>
#include <common/render.h>
#include <common/htmltable.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <util/alloc.h>
#include <util/gv_math.h>
#include <util/streq.h>
#include <util/unreachable.h>

#define RBCONST 12
#define RBCURVE .5

typedef struct {
    pointf (*size_gen) (pointf);
    void (*vertex_gen) (pointf*, pointf*);
} poly_desc_t;

static port Center = {.theta = -1, .clip = true};

#define ATTR_SET(a,n) ((a) && (*(agxget(n,a->index)) != '\0'))
  /* Default point size = 0.05 inches or 3.6 points */
#define DEF_POINT 0.05
  /* Minimum point size = 0.0003 inches or 0.02 points
   * This will make the radius 0.01 points, which is the smallest
   * non-zero number output by gvprintdouble in gvdevice.c
   */
#define MIN_POINT 0.0003
  /* extra null character needed to avoid style emitter from thinking
   * there are arguments.
   */
static char *point_style[3] = { "invis\0", "filled\0", 0 };

/* forward declarations of functions used in shapes tables */

static void poly_init(node_t * n);
static void poly_free(node_t * n);
static port poly_port(node_t * n, char *portname, char *);
static bool poly_inside(inside_t * inside_context, pointf p);
static int poly_path(node_t * n, port * p, int side, boxf rv[], int *kptr);
static void poly_gencode(GVJ_t * job, node_t * n);

static void record_init(node_t * n);
static void record_free(node_t * n);
static port record_port(node_t * n, char *portname, char *);
static bool record_inside(inside_t * inside_context, pointf p);
static int record_path(node_t * n, port * p, int side, boxf rv[],
		       int *kptr);
static void record_gencode(GVJ_t * job, node_t * n);

static void point_init(node_t * n);
static void point_gencode(GVJ_t * job, node_t * n);
static bool point_inside(inside_t * inside_context, pointf p);

static bool epsf_inside(inside_t * inside_context, pointf p);
static void epsf_gencode(GVJ_t * job, node_t * n);

static pointf star_size (pointf);
static void star_vertices (pointf*, pointf*);
static bool star_inside(inside_t * inside_context, pointf p);
static poly_desc_t star_gen = {
    star_size,
    star_vertices,
};

static pointf cylinder_size (pointf);
static void cylinder_vertices (pointf*, pointf*);
static void cylinder_draw(GVJ_t *job, pointf *AF, size_t sides, int filled);
static poly_desc_t cylinder_gen = {
    cylinder_size,
    cylinder_vertices,
};

/* polygon descriptions.  "polygon" with 0 sides takes all user control */

/*			       regul perip sides orien disto skew */
static polygon_t p_polygon = {.peripheries = 1};

/* builtin polygon descriptions */
static polygon_t p_ellipse = {.peripheries = 1, .sides = 1};
static polygon_t p_circle = {.regular = true, .peripheries = 1, .sides = 1};
static polygon_t p_egg = {.peripheries = 1, .sides = 1, .distortion = -0.3};
static polygon_t p_triangle = {.peripheries = 1, .sides = 3};
static polygon_t p_box = {.peripheries = 1, .sides = 4};
static polygon_t p_square = {.regular = true, .peripheries = 1, .sides = 4};
static polygon_t p_plaintext = {.sides = 4};
static polygon_t p_plain = {.sides = 4};
static polygon_t p_diamond = {.peripheries = 1, .sides = 4, .orientation = 45.0};
static polygon_t p_trapezium = {.peripheries = 1, .sides = 4, .distortion = -0.4};
static polygon_t p_parallelogram = {.peripheries = 1, .sides = 4, .skew = 0.6};
static polygon_t p_house = {.peripheries = 1, .sides = 5, .distortion = -0.64};
static polygon_t p_pentagon = {.peripheries = 1, .sides = 5};
static polygon_t p_hexagon = {.peripheries = 1, .sides = 6};
static polygon_t p_septagon = {.peripheries = 1, .sides = 7};
static polygon_t p_octagon = {.peripheries = 1, .sides = 8};
static polygon_t p_note = {
    .peripheries = 1, .sides = 4, .option = {.shape = DOGEAR}};
static polygon_t p_tab = {
    .peripheries = 1, .sides = 4, .option = {.shape = TAB}};
static polygon_t p_folder = {
    .peripheries = 1, .sides = 4, .option = {.shape = FOLDER}};
static polygon_t p_box3d = {
    .peripheries = 1, .sides = 4, .option = {.shape = BOX3D}};
static polygon_t p_component = {
    .peripheries = 1, .sides = 4, .option = {.shape = COMPONENT}};
static polygon_t p_underline = {
    .peripheries = 1, .sides = 4, .option = {.underline = true}};
static polygon_t p_cylinder = {.peripheries = 1,
                               .sides = 19,
                               .option = {.shape = CYLINDER},
                               .vertices = (pointf *)&cylinder_gen};

/* redundant and undocumented builtin polygons */
static polygon_t p_doublecircle = {
    .regular = true, .peripheries = 2, .sides = 1};
static polygon_t p_invtriangle = {
    .peripheries = 1, .sides = 3, .orientation = 180.0};
static polygon_t p_invtrapezium = {
    .peripheries = 1, .sides = 4, .orientation = 180.0, .distortion = -0.4};
static polygon_t p_invhouse = {
    .peripheries = 1, .sides = 5, .orientation = 180.0, .distortion = -0.64};
static polygon_t p_doubleoctagon = {.peripheries = 2, .sides = 8};
static polygon_t p_tripleoctagon = {.peripheries = 3, .sides = 8};
static polygon_t p_Mdiamond = {
    .peripheries = 1,
    .sides = 4,
    .orientation = 45.0,
    .option = {.diagonals = true, .auxlabels = true}};
static polygon_t p_Msquare = {.regular = true,
                              .peripheries = 1,
                              .sides = 4,
                              .option = {.diagonals = true}};
static polygon_t p_Mcircle = {.regular = true,
                              .peripheries = 1,
                              .sides = 1,
                              .option = {.diagonals = true, .auxlabels = true}};

/* non-convex polygons */
static polygon_t p_star = {
    .peripheries = 1, .sides = 10, .vertices = (pointf *)&star_gen};

/* biological circuit shapes, as specified by SBOLv*/
/** gene expression symbols **/
static polygon_t p_promoter = {
    .peripheries = 1, .sides = 4, .option = {.shape = PROMOTER}};
static polygon_t p_cds = {
    .peripheries = 1, .sides = 4, .option = {.shape = CDS}};
static polygon_t p_terminator = {
    .peripheries = 1, .sides = 4, .option = {.shape = TERMINATOR}};
static polygon_t p_utr = {
    .peripheries = 1, .sides = 4, .option = {.shape = UTR}};
static polygon_t p_insulator = {
    .peripheries = 1, .sides = 4, .option = {.shape = INSULATOR}};
static polygon_t p_ribosite = {
    .peripheries = 1, .sides = 4, .option = {.shape = RIBOSITE}};
static polygon_t p_rnastab = {
    .peripheries = 1, .sides = 4, .option = {.shape = RNASTAB}};
static polygon_t p_proteasesite = {
    .peripheries = 1, .sides = 4, .option = {.shape = PROTEASESITE}};
static polygon_t p_proteinstab = {
    .peripheries = 1, .sides = 4, .option = {.shape = PROTEINSTAB}};
/** dna construction symbols **/
static polygon_t p_primersite = {
    .peripheries = 1, .sides = 4, .option = {.shape = PRIMERSITE}};
static polygon_t p_restrictionsite = {
    .peripheries = 1, .sides = 4, .option = {.shape = RESTRICTIONSITE}};
static polygon_t p_fivepoverhang = {
    .peripheries = 1, .sides = 4, .option = {.shape = FIVEPOVERHANG}};
static polygon_t p_threepoverhang = {
    .peripheries = 1, .sides = 4, .option = {.shape = THREEPOVERHANG}};
static polygon_t p_noverhang = {
    .peripheries = 1, .sides = 4, .option = {.shape = NOVERHANG}};
static polygon_t p_assembly = {
    .peripheries = 1, .sides = 4, .option = {.shape = ASSEMBLY}};
static polygon_t p_signature = {
    .peripheries = 1, .sides = 4, .option = {.shape = SIGNATURE}};
static polygon_t p_rpromoter = {
    .peripheries = 1, .sides = 4, .option = {.shape = RPROMOTER}};
static polygon_t p_rarrow = {
    .peripheries = 1, .sides = 4, .option = {.shape = RARROW}};
static polygon_t p_larrow = {
    .peripheries = 1, .sides = 4, .option = {.shape = LARROW}};
static polygon_t p_lpromoter = {
    .peripheries = 1, .sides = 4, .option = {.shape = LPROMOTER}};

static bool IS_BOX(node_t *n) {
  return ND_shape(n)->polygon == &p_box;
}

static bool IS_PLAIN(node_t *n) {
  return ND_shape(n)->polygon == &p_plain;
}

/// True if style requires processing through round_corners.
static bool SPECIAL_CORNERS(graphviz_polygon_style_t style) {
  return style.rounded || style.diagonals || style.shape != 0;
}

/*
 * every shape has these functions:
 *
 * void		SHAPE_init(node_t *n)
 *			initialize the shape (usually at least its size).
 * void		SHAPE_free(node_t *n)
 *			free all memory used by the shape
 * port		SHAPE_port(node_t *n, char *portname)
 *			return the aiming point and slope (if constrained)
 *			of a port.
 * int		SHAPE_inside(inside_t *inside_context, pointf p, edge_t *e);
 *			test if point is inside the node shape which is
 *			assumed convex.
 *			the point is relative to the node center.  the edge
 *			is passed in case the port affects spline clipping.
 * int		SHAPE_path(node *n, edge_t *e, int pt, boxf path[], int *nbox)
 *			create a path for the port of e that touches n,
 *			return side
 * void		SHAPE_gencode(GVJ_t *job, node_t *n)
 *			generate graphics code for a node.
 *
 * some shapes, polygons in particular, use additional shape control data *
 *
 */

static shape_functions poly_fns = {
    poly_init,
    poly_free,
    poly_port,
    poly_inside,
    poly_path,
    poly_gencode
};
static shape_functions point_fns = {
    point_init,
    poly_free,
    poly_port,
    point_inside,
    NULL,
    point_gencode
};
static shape_functions record_fns = {
    record_init,
    record_free,
    record_port,
    record_inside,
    record_path,
    record_gencode
};
static shape_functions epsf_fns = {
    epsf_init,
    epsf_free,
    poly_port,
    epsf_inside,
    NULL,
    epsf_gencode
};
static shape_functions star_fns = {
    poly_init,
    poly_free,
    poly_port,
    star_inside,
    poly_path,
    poly_gencode
};
static shape_functions cylinder_fns = {
    poly_init,
    poly_free,
    poly_port,
    poly_inside,
    poly_path,
    poly_gencode
};

static shape_desc Shapes[] = {	/* first entry is default for no such shape */
    {.name = "box", .fns = &poly_fns, .polygon = &p_box},
    {.name = "polygon", .fns = &poly_fns, .polygon = &p_polygon},
    {.name = "ellipse", .fns = &poly_fns, .polygon = &p_ellipse},
    {.name = "oval", .fns = &poly_fns, .polygon = &p_ellipse},
    {.name = "circle", .fns = &poly_fns, .polygon = &p_circle},
    {.name = "point", .fns = &point_fns, .polygon = &p_circle},
    {.name = "egg", .fns = &poly_fns, .polygon = &p_egg},
    {.name = "triangle", .fns = &poly_fns, .polygon = &p_triangle},
    {.name = "none", .fns = &poly_fns, .polygon = &p_plaintext},
    {.name = "plaintext", .fns = &poly_fns, .polygon = &p_plaintext},
    {.name = "plain", .fns = &poly_fns, .polygon = &p_plain},
    {.name = "diamond", .fns = &poly_fns, .polygon = &p_diamond},
    {.name = "trapezium", .fns = &poly_fns, .polygon = &p_trapezium},
    {.name = "parallelogram", .fns = &poly_fns, .polygon = &p_parallelogram},
    {.name = "house", .fns = &poly_fns, .polygon = &p_house},
    {.name = "pentagon", .fns = &poly_fns, .polygon = &p_pentagon},
    {.name = "hexagon", .fns = &poly_fns, .polygon = &p_hexagon},
    {.name = "septagon", .fns = &poly_fns, .polygon = &p_septagon},
    {.name = "octagon", .fns = &poly_fns, .polygon = &p_octagon},
    {.name = "note", .fns = &poly_fns, .polygon = &p_note},
    {.name = "tab", .fns = &poly_fns, .polygon = &p_tab},
    {.name = "folder", .fns = &poly_fns, .polygon = &p_folder},
    {.name = "box3d", .fns = &poly_fns, .polygon = &p_box3d},
    {.name = "component", .fns = &poly_fns, .polygon = &p_component},
    {.name = "cylinder", .fns = &cylinder_fns, .polygon = &p_cylinder},
    {.name = "rect", .fns = &poly_fns, .polygon = &p_box},
    {.name = "rectangle", .fns = &poly_fns, .polygon = &p_box},
    {.name = "square", .fns = &poly_fns, .polygon = &p_square},
    {.name = "doublecircle", .fns = &poly_fns, .polygon = &p_doublecircle},
    {.name = "doubleoctagon", .fns = &poly_fns, .polygon = &p_doubleoctagon},
    {.name = "tripleoctagon", .fns = &poly_fns, .polygon = &p_tripleoctagon},
    {.name = "invtriangle", .fns = &poly_fns, .polygon = &p_invtriangle},
    {.name = "invtrapezium", .fns = &poly_fns, .polygon = &p_invtrapezium},
    {.name = "invhouse", .fns = &poly_fns, .polygon = &p_invhouse},
    {.name = "underline", .fns = &poly_fns, .polygon = &p_underline},
    {.name = "Mdiamond", .fns = &poly_fns, .polygon = &p_Mdiamond},
    {.name = "Msquare", .fns = &poly_fns, .polygon = &p_Msquare},
    {.name = "Mcircle", .fns = &poly_fns, .polygon = &p_Mcircle},
	/* biological circuit shapes, as specified by SBOLv*/
	/** gene expression symbols **/
    {.name = "promoter", .fns = &poly_fns, .polygon = &p_promoter},
    {.name = "cds", .fns =  &poly_fns, .polygon = &p_cds},
    {.name = "terminator", .fns =  &poly_fns, .polygon = &p_terminator},
    {.name = "utr", .fns =  &poly_fns, .polygon = &p_utr},
    {.name = "insulator", .fns = &poly_fns, .polygon = &p_insulator},
    {.name = "ribosite", .fns = &poly_fns, .polygon = &p_ribosite},
    {.name = "rnastab", .fns = &poly_fns, .polygon = &p_rnastab},
    {.name = "proteasesite", .fns = &poly_fns, .polygon = &p_proteasesite},
    {.name = "proteinstab", .fns = &poly_fns, .polygon = &p_proteinstab},
	/** dna construction symbols **/
    {.name = "primersite", .fns =  &poly_fns, .polygon = &p_primersite},
    {.name = "restrictionsite", .fns = &poly_fns, .polygon = &p_restrictionsite},
    {.name = "fivepoverhang", .fns = &poly_fns, .polygon = &p_fivepoverhang},
    {.name = "threepoverhang", .fns = &poly_fns, .polygon = &p_threepoverhang},
    {.name = "noverhang", .fns = &poly_fns, .polygon = &p_noverhang},
    {.name = "assembly", .fns = &poly_fns, .polygon = &p_assembly},
    {.name = "signature", .fns = &poly_fns, .polygon = &p_signature},
    {.name = "rpromoter", .fns = &poly_fns, .polygon = &p_rpromoter},
    {.name = "larrow", .fns =  &poly_fns, .polygon = &p_larrow},
    {.name = "rarrow", .fns =  &poly_fns, .polygon = &p_rarrow},
    {.name = "lpromoter", .fns =  &poly_fns, .polygon = &p_lpromoter},
	/*  *** shapes other than polygons  *** */
    {.name = "record", .fns = &record_fns, .polygon = NULL},
    {.name = "Mrecord", .fns = &record_fns, .polygon = NULL},
    {.name = "epsf", .fns = &epsf_fns, .polygon = NULL},
    {.name = "star", .fns = &star_fns, .polygon = &p_star},
    {0}
};

static void unrecognized(node_t * n, char *p)
{
    agwarningf("node %s, port %s unrecognized\n", agnameof(n), p);
}

static double quant(double val, double q)
{
  return ceil(val / q) * q;
}

/* test if both p0 and p1 are on the same side of the line L0,L1 */
static int same_side(pointf p0, pointf p1, pointf L0, pointf L1)
{
    int s0, s1;
    double a, b, c;

    /* a x + b y = c */
    a = -(L1.y - L0.y);
    b = L1.x - L0.x;
    c = a * L0.x + b * L0.y;

    s0 = a * p0.x + b * p0.y - c >= 0;
    s1 = a * p1.x + b * p1.y - c >= 0;
    return s0 == s1;
}

static
char* penColor(GVJ_t * job, node_t * n)
{
    char *color;

    color = late_nnstring(n, N_color, "");
    if (!color[0])
	color = DEFAULT_COLOR;
    gvrender_set_pencolor(job, color);
    return color;
}

static
char *findFillDflt(node_t * n, char *dflt)
{
    char *color;

    color = late_nnstring(n, N_fillcolor, "");
    if (!color[0]) {
	/* for backward compatibility, default fill is same as pen */
	color = late_nnstring(n, N_color, "");
	if (!color[0]) {
	    color = dflt;
	}
    }
    return color;
}

static
char *findFill(node_t * n)
{
    return findFillDflt(n, DEFAULT_FILL);
}

static bool isBox(node_t *n) {
    polygon_t *p;

    if ((p = ND_shape(n)->polygon)) {
      return p->sides == 4 && fabs(fmod(p->orientation, 90)) < 0.5 &&
             is_exactly_zero(p->distortion) && is_exactly_zero(p->skew);
    }
    return false;
}

static bool isEllipse(node_t *n) {
    polygon_t *p;

    if ((p = ND_shape(n)->polygon)) {
	return p->sides <= 2;
    }
    return false;
}

/// bitwise-OR styles
static graphviz_polygon_style_t style_or(graphviz_polygon_style_t a,
                                         graphviz_polygon_style_t b) {

  // bitwise-or-ing the shape does not make sense, so there better only be one
  assert(a.shape == 0 || b.shape == 0);

  return (graphviz_polygon_style_t){
    .filled = a.filled || b.filled,
    .radial = a.radial || b.radial,
    .rounded = a.rounded || b.rounded,
    .diagonals = a.diagonals || b.diagonals,
    .auxlabels = a.auxlabels || b.auxlabels,
    .invisible = a.invisible || b.invisible,
    .striped = a.striped || b.striped,
    .dotted = a.dotted || b.dotted,
    .dashed = a.dashed || b.dashed,
    .wedged = a.wedged || b.wedged,
    .underline = a.underline || b.underline,
    .fixedshape = a.fixedshape || b.fixedshape,
    .shape = a.shape | b.shape,
  };
}

static char **checkStyle(node_t *n, graphviz_polygon_style_t *flagp) {
    char *style;
    char **pstyle = 0;
    graphviz_polygon_style_t istyle = {0};
    polygon_t *poly;

    style = late_nnstring(n, N_style, "");
    if (style[0]) {
	char **pp;
	char **qp;
	char *p;
	pp = pstyle = parse_style(style);
	while ((p = *pp)) {
	    if (streq(p, "filled")) {
		istyle.filled = true;
		pp++;
	    } else if (streq(p, "rounded")) {
		istyle.rounded = true;
		qp = pp;	/* remove rounded from list passed to renderer */
		do {
		    qp++;
		    *(qp - 1) = *qp;
		} while (*qp);
	    } else if (streq(p, "diagonals")) {
		istyle.diagonals = true;
		qp = pp;	/* remove diagonals from list passed to renderer */
		do {
		    qp++;
		    *(qp - 1) = *qp;
		} while (*qp);
	    } else if (streq(p, "invis")) {
		istyle.invisible = true;
		pp++;
	    } else if (streq(p, "radial")) {
		istyle.radial = true;
		istyle.filled = true;
		qp = pp;	/* remove radial from list passed to renderer */
		do {
		    qp++;
		    *(qp - 1) = *qp;
		} while (*qp);
	    } else if (streq(p, "striped") && isBox(n)) {
		istyle.striped = true;
		qp = pp;	/* remove striped from list passed to renderer */
		do {
		    qp++;
		    *(qp - 1) = *qp;
		} while (*qp);
	    } else if (streq(p, "wedged") && isEllipse(n)) {
		istyle.wedged = true;
		qp = pp;	/* remove wedged from list passed to renderer */
		do {
		    qp++;
		    *(qp - 1) = *qp;
		} while (*qp);
	    } else
		pp++;
	}
    }
    if ((poly = ND_shape(n)->polygon))
	istyle = style_or(istyle, poly->option);

    *flagp = istyle;
    return pstyle;
}

static graphviz_polygon_style_t stylenode(GVJ_t *job, node_t *n) {
    char **pstyle, *s;
    graphviz_polygon_style_t istyle = {0};
    double penwidth;

    if ((pstyle = checkStyle(n, &istyle)))
	gvrender_set_style(job, pstyle);

    if (N_penwidth && (s = agxget(n, N_penwidth)) && s[0]) {
	penwidth = late_double(n, N_penwidth, 1.0, 0.0);
	gvrender_set_penwidth(job, penwidth);
    }

    return istyle;
}

static void Mcircle_hack(GVJ_t * job, node_t * n)
{
    double x, y;
    pointf AF[2], p;

    y = .7500;
    x = .6614;			/* x^2 + y^2 = 1.0 */
    p.y = y * ND_ht(n) / 2.0;
    p.x = ND_rw(n) * x;		/* assume node is symmetric */

    AF[0] = add_pointf(p, ND_coord(n));
    AF[1].y = AF[0].y;
    AF[1].x = AF[0].x - 2 * p.x;
    gvrender_polyline(job, AF, 2);
    AF[0].y -= 2 * p.y;
    AF[1].y = AF[0].y;
    gvrender_polyline(job, AF, 2);
}

static pointf * alloc_interpolation_points(pointf *AF, size_t sides,
		graphviz_polygon_style_t style, bool rounded)
{
    pointf *B = gv_calloc(4 * sides + 4, sizeof(pointf));
    size_t i = 0;
    pointf p0, p1;
    double dx, dy, t;
    /* rbconst is distance offset from a corner of the polygon.
     * It should be the same for every corner, and also never
     * bigger than one-third the length of a side.
     */
    double rbconst = RBCONST;
    for (size_t seg = 0; seg < sides; seg++) {
	p0 = AF[seg];
	if (seg + 1 < sides)
	    p1 = AF[seg + 1];
	else
	    p1 = AF[0];
	dx = p1.x - p0.x;
	dy = p1.y - p0.y;
	const double d = hypot(dx, dy);
	rbconst = fmin(rbconst, d / 3.0);
    }
    for (size_t seg = 0; seg < sides; seg++) {
	p0 = AF[seg];
	if (seg + 1 < sides)
	    p1 = AF[seg + 1];
	else
	    p1 = AF[0];
	dx = p1.x - p0.x;
	dy = p1.y - p0.y;
	const double d = hypot(dx, dy);
	t = rbconst / d;
	if (style.shape == BOX3D || style.shape == COMPONENT)
	    t /= 3;
	else if (style.shape == DOGEAR)
	    t /= 2;
	if (!rounded)
	    B[i++] = p0;
	else
	    B[i++] = interpolate_pointf(RBCURVE * t, p0, p1);
	B[i++] = interpolate_pointf(t, p0, p1);
	B[i++] = interpolate_pointf(1.0 - t, p0, p1);
	if (rounded)
	    B[i++] = interpolate_pointf(1.0 - RBCURVE * t, p0, p1);
    }
    B[i++] = B[0];
    B[i++] = B[1];
    B[i++] = B[2];

    return B;
}

/**
 * @brief draws polygons with diagonals on corners
 *
 * Diagonals are weird. Rewrite someday.
 */
static void diagonals_draw(GVJ_t *job, pointf *AF, size_t sides,
                           graphviz_polygon_style_t style, int filled)
{
  pointf *B = alloc_interpolation_points(AF, sides, style, false);
  gvrender_polygon(job, AF, sides, filled);

  for (size_t seg = 0; seg < sides; seg++) {
    pointf C[] = {B[3 * seg + 2], B[3 * seg + 4]};
    gvrender_polyline(job, C, 2);
  }
  free(B);
}

/**
 * @brief draws rounded polygons with
 * [Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve)
 *
 * For example, a rounded star looks like a cartoon starfish.
 */
static void rounded_draw(GVJ_t *job, pointf *AF, size_t sides,
                         graphviz_polygon_style_t style, int filled)
{
  size_t i = 0;

  pointf *B = alloc_interpolation_points(AF, sides, style, true);
  pointf *pts = gv_calloc(6 * sides + 2, sizeof(pointf));
  for (size_t seg = 0; seg < sides; seg++) {
    pts[i++] = B[4 * seg];
    pts[i++] = B[4 * seg + 1];
    pts[i++] = B[4 * seg + 1];
    pts[i++] = B[4 * seg + 2];
    pts[i++] = B[4 * seg + 2];
    pts[i++] = B[4 * seg + 3];
  }
  pts[i++] = pts[0];
  pts[i++] = pts[1];
  gvrender_beziercurve(job, pts + 1, i - 1, filled);
  free(pts);
  free(B);
}

/**
 * @file
 * ~~~~
 *                y
 *                 🡑
 *                 │
 *                 │
 *                 │         line[1]
 *                 │       ⟋
 *           mid_y ┤   middle
 *                 │   ⟋
 *                 │line[0]         x
 *                ─┼─────┬───────────🡒
 *                      mid_x
 * ~~~~
 */

/**
 * @brief X coordinate of line midpoint
 * @param line two points
 * @returns X coordinate of midpoint
 */
static double mid_x(const pointf line[2]) {
  return (line[0].x + line[1].x) / 2;
}

/**
 * @brief Y coordinate of line midpoint
 * @param line two points
 * @returns Y coordinate of midpoint
 */
static double mid_y(const pointf line[2]) {
  return (line[0].y + line[1].y) / 2;
}

/**
 * @brief Handle some special graphical cases, such as rounding the shape,
 * adding diagonals at corners, or drawing certain non-simple figures.
 *
 * Any drawing done here should assume fillcolors, pencolors, etc.
 * have been set by the calling routine. Normally, the drawing should
 * consist of a region, filled or unfilled, followed by additional line
 * segments. A single fill is necessary for gradient colors to work.
 */
void round_corners(GVJ_t *job, pointf *AF, size_t sides,
                   graphviz_polygon_style_t style, int filled) {
    assert(job != NULL);
    assert(AF != NULL);
    assert(sides > 0);
    assert(memcmp(&style, &(graphviz_polygon_style_t){0}, sizeof(style)) != 0);

    pointf *B, C[5], *D;

    struct {
	unsigned shape: 7;
    } mode = {0};

    if (style.diagonals) {
	diagonals_draw(job, AF, sides, style, filled);
	return;
    } else if (style.shape != 0) {
	mode.shape = style.shape;
    } else if (style.rounded) {
	rounded_draw(job, AF, sides, style, filled);
	return;
    } else {
	UNREACHABLE();
    }

    if (mode.shape == CYLINDER) {
	cylinder_draw(job, AF, sides, filled);
	return;
    }
    B = alloc_interpolation_points(AF, sides, style, false);
    switch (mode.shape) {
    case DOGEAR:
	/* Add the cutoff edge. */
	D = gv_calloc(sides + 1, sizeof(pointf));
	for (size_t seg = 1; seg < sides; seg++)
	    D[seg] = AF[seg];
	D[0] = B[3 * (sides - 1) + 4];
	D[sides] = B[3 * (sides - 1) + 2];
	gvrender_polygon(job, D, sides + 1, filled);
	free(D);

	/* Draw the inner edge. */
	const size_t sseg = sides - 1;
	C[0] = B[3 * sseg + 2];
	C[1] = B[3 * sseg + 4];
	C[2].x = C[1].x + (C[0].x - B[3 * sseg + 3].x);
	C[2].y = C[1].y + (C[0].y - B[3 * sseg + 3].y);
	gvrender_polyline(job, C + 1, 2);
	C[1] = C[2];
	gvrender_polyline(job, C, 2);
	break;
    case TAB:
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *  D[3] ×──× D[2]
	 *       │  │          B[1]
	 *  B[3] ×──×──────────×──× AF[0]=B[0]=D[0]
	 *       │  B[2]=D[1]     │
	 *  B[4] ×                │
	 *       │                │
	 *  B[5] ×                │
	 *       └────────────────┘
	 *
	 */
	/* Add the tab edges. */
	D = gv_calloc(sides + 2, sizeof(pointf));
	D[0] = AF[0];
	D[1] = B[2];
	D[2].x = B[2].x + (B[3].x - B[4].x) / 3;
	D[2].y = B[2].y + (B[3].y - B[4].y) / 3;
	D[3].x = B[3].x + (B[3].x - B[4].x) / 3;
	D[3].y = B[3].y + (B[3].y - B[4].y) / 3;
	for (size_t seg = 4; seg < sides + 2; seg++)
	    D[seg] = AF[seg - 2];
	gvrender_polygon(job, D, sides + 2, filled);
	free(D);


	/* Draw the inner edge. */
	C[0] = B[3];
	C[1] = B[2];
	gvrender_polyline(job, C, 2);
	break;
    case FOLDER:
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *            D[2] ×────× D[1]
	 *  B[3]=         ╱      ╲
	 *  D[4] ×──×────×     ×  × AF[0]=B[0]=D[0]
	 *       │  B[2] D[3] B[1]│
	 *  B[4] ×                │
	 *       │                │
	 *  B[5] ×                │
	 *       └────────────────┘
	 *
	 */
	/* Add the folder edges. */
	D = gv_calloc(sides + 3, sizeof(pointf));
	D[0] = AF[0];
	D[1].x = AF[0].x - (AF[0].x - B[1].x) / 4;
	D[1].y = AF[0].y + (B[3].y - B[4].y) / 3;
	D[2].x = AF[0].x - 2 * (AF[0].x - B[1].x);
	D[2].y = D[1].y;
	D[3].x = AF[0].x - 2.25 * (AF[0].x - B[1].x);
	D[3].y = B[3].y;
	D[4].x = B[3].x;
	D[4].y = B[3].y;
	for (size_t seg = 4; seg < sides + 3; seg++)
	    D[seg] = AF[seg - 3];
	gvrender_polygon(job, D, sides + 3, filled);
	free(D);
	break;
    case BOX3D:
	assert(sides == 4);
	/* Adjust for the cutoff edges. */
	D = gv_calloc(sides + 2, sizeof(pointf));
	D[0] = AF[0];
	D[1] = B[2];
	D[2] = B[4];
	D[3] = AF[2];
	D[4] = B[8];
	D[5] = B[10];
	gvrender_polygon(job, D, sides + 2, filled);
	free(D);

	/* Draw the inner vertices. */
	C[0].x = B[1].x + (B[11].x - B[0].x);
	C[0].y = B[1].y + (B[11].y - B[0].y);
	C[1] = B[4];
	gvrender_polyline(job, C, 2);
	C[1] = B[8];
	gvrender_polyline(job, C, 2);
	C[1] = B[0];
	gvrender_polyline(job, C, 2);
	break;
    case COMPONENT:
	assert(sides == 4);
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *  D[1] ×────────────────× D[0]
	 *       │                │
	 *  3×───×2──┐            │
	 *   │       │            │
	 *  4×───×5──┘            │
	 *       │                │
	 *  7×───×6──┐            │
	 *   │       │            │
	 *  8×───×9──┘            │
	 *       │                │
	 *     10×────────────────× D[11]
	 *
	 */
	D = gv_calloc(sides + 8, sizeof(pointf));
	D[0] = AF[0];
	D[1] = AF[1];
	D[2].x = B[3].x + (B[4].x - B[3].x);
	D[2].y = B[3].y + (B[4].y - B[3].y);
	D[3].x = D[2].x + (B[3].x - B[2].x);
	D[3].y = D[2].y + (B[3].y - B[2].y);
	D[4].x = D[3].x + (B[4].x - B[3].x);
	D[4].y = D[3].y + (B[4].y - B[3].y);
	D[5].x = D[4].x + (D[2].x - D[3].x);
	D[5].y = D[4].y + (D[2].y - D[3].y);

	D[9].x = B[6].x + (B[5].x - B[6].x);
	D[9].y = B[6].y + (B[5].y - B[6].y);
	D[8].x = D[9].x + (B[6].x - B[7].x);
	D[8].y = D[9].y + (B[6].y - B[7].y);
	D[7].x = D[8].x + (B[5].x - B[6].x);
	D[7].y = D[8].y + (B[5].y - B[6].y);
	D[6].x = D[7].x + (D[9].x - D[8].x);
	D[6].y = D[7].y + (D[9].y - D[8].y);

	D[10] = AF[2];
	D[11] = AF[3];
	gvrender_polygon(job, D, sides + 8, filled);

	/* Draw the internal vertices. */
	C[0] = D[2];
	C[1].x = D[2].x - (D[3].x - D[2].x);
	C[1].y = D[2].y - (D[3].y - D[2].y);
	C[2].x = C[1].x + (D[4].x - D[3].x);
	C[2].y = C[1].y + (D[4].y - D[3].y);
	C[3] = D[5];
	gvrender_polyline(job, C, 4);
	C[0] = D[6];
	C[1].x = D[6].x - (D[7].x - D[6].x);
	C[1].y = D[6].y - (D[7].y - D[6].y);
	C[2].x = C[1].x + (D[8].x - D[7].x);
	C[2].y = C[1].y + (D[8].y - D[7].y);
	C[3] = D[9];
	gvrender_polyline(job, C, 4);

	free(D);
	break;

    case PROMOTER:
	/*
	 * L-shaped arrow on a center line, scales in the x direction
	 *
	 *
	 *      D[1]              │╲
	 *       ×────────────────× ╲
	 *       │              D[0] ╲
	 *       │                    ╲
	 *       │                    ╱
	 *       │             D[5]  ╱
	 *       │        ┌───────× ╱
	 *       │        │       │╱
	 *  ─────┴────────┴─────────────
	 */
	/* Add the tab edges. */

	//the arrow's thickness is (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides + 5, sizeof(pointf));
	D[0].x = mid_x(AF) + (AF[0].x - AF[1].x)/8; //x_center + width
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)*3/2; //D[4].y + width
	D[1].x = mid_x(AF) - (AF[0].x - AF[1].x)/4; //x_center - 2*width
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = mid_y(&AF[1]);
	D[3].x = D[2].x + (B[2].x - B[3].x)/2; //D[2].x + width
	D[3].y = mid_y(&AF[1]);
	D[4].x = D[3].x;
	D[4].y = mid_y(&AF[1]) + (B[3].y-B[4].y); //highest cds point
	D[5].x = D[0].x;
	D[5].y = D[4].y; //highest cds point
	D[6].x = D[0].x;
	D[6].y = D[4].y - (B[3].y-B[4].y)/4; //D[4].y - width/2
	D[7].x = D[6].x + (B[2].x - B[3].x); //D[6].x + 2*width
	D[7].y = D[6].y + (B[3].y - B[4].y)/2; //D[6].y + width
	D[8].x = D[0].x;
	D[8].y = D[0].y + (B[3].y - B[4].y)/4;//D[0].y + width/2
	gvrender_polygon(job, D, sides + 5, filled);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;

    case CDS:
	/*
	 * arrow without the protrusions, scales normally
	 *
	 *
	 *      D[1] = AF[1]
	 *       ×────────────────×╲
	 *       │              D[0]╲
	 *       │                   ╲
	 *       │                   ╱
	 *       │                  ╱
	 *       ×────────────────×╱
	 *                        D[3]
	 *
	 */
	D = gv_calloc(sides + 1, sizeof(pointf));
	D[0].x = B[1].x;
	D[0].y = B[1].y - (B[3].y - B[4].y)/2;
	D[1].x = B[3].x;
	D[1].y = B[3].y - (B[3].y - B[4].y)/2;
	D[2].x = AF[2].x;
	D[2].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[3].x = B[1].x;
	D[3].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[4].y = AF[0].y - (AF[0].y - AF[3].y)/2;
	D[4].x = AF[0].x;

	gvrender_polygon(job, D, sides + 1, filled);
	free(D);

	break;

    case TERMINATOR:
	/*
	* T-shape, does not scale, always in the center
	*
	*
	*      D[4]
	*       ×────────────────×
	*       │               D[3]
	*       │                │
	*       │                │
	*       │  D[6]    D[1]  │
	*   D[5]×───×       ×────× D[2]
	*           │       │
	*  ─────────┴───────×─D[0]──────
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	D = gv_calloc(sides + 4, sizeof(pointf));
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/4; //x_center + width/2
	D[0].y = mid_y(&AF[1]);
	D[1].x = D[0].x;
	D[1].y = D[0].y + (B[3].y-B[4].y)/2;
	D[2].x = D[1].x + (B[2].x-B[3].x)/2;
	D[2].y = D[1].y;
	D[3].x = D[2].x;
	D[3].y = D[2].y + (B[3].y-B[4].y)/2;
	D[4].x = mid_x(AF) - (B[2].x-B[3].x)*3/4; //D[3].y mirrored across the center
	D[4].y = D[3].y;
	D[5].x = D[4].x;
	D[5].y = D[2].y;
	D[6].x = mid_x(AF) - (B[2].x-B[3].x)/4; //D[1].x mirrored across the center
	D[6].y = D[1].y;
	D[7].x = D[6].x;
	D[7].y = D[0].y;
	gvrender_polygon(job, D, sides + 4, filled);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;

    case UTR:
	/*
	 * half-octagon with line, does not scale, always in center
	 *
	 *      D[3]
	 *         ─────  D[2]
	 *        ╱     ╲
	 *       ╱       ╲ D[1]
	 *       │       │
	 *  ─────┴───────┴───────
	 *                  D[0]
	 *
	 *
	 *
	 */
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	D = gv_calloc(sides + 2, sizeof(pointf));
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)*3/4; //x_center+width
	D[0].y = mid_y(&AF[1]);
	D[1].x = D[0].x;
	D[1].y = D[0].y + (B[3].y-B[4].y)/4; //D[0].y+width/2
	D[2].x = mid_x(AF) + (B[2].x-B[3].x)/4; //x_center+width/2
	D[2].y = D[1].y + (B[3].y-B[4].y)/2; //D[1].y+width
	D[3].x = mid_x(AF) - (B[2].x-B[3].x)/4; //D[2].x mirrored across the center
	D[3].y = D[2].y;
	D[4].x = mid_x(AF) - (B[2].x-B[3].x)*3/4;
	D[4].y = D[1].y;
	D[5].x = D[4].x;
	D[5].y = D[0].y;
	gvrender_polygon(job, D, sides + 2, filled);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case PRIMERSITE:
	/*
	* half arrow shape, scales in the x-direction
	*                 D[1]
	*                   │╲
	*                   │ ╲
	*                   │  ╲
	*       ┌───────────┘   ╲
	*       │                ╲
	*       └─────────────────╲ D[0]
	*
	*   ────────────────────────────────
	*
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides + 1, sizeof(pointf));
	D[0].x = mid_x(AF) + (B[2].x-B[3].x);//x_center + width*2
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/4;//y_center + 1/2 width
	D[1].x = D[0].x - (B[2].x-B[3].x); //x_center
	D[1].y = D[0].y + (B[3].y-B[4].y);
	D[2].x = D[1].x;
	D[2].y = D[0].y + (B[3].y-B[4].y)/2;
	D[3].x = mid_x(AF) - (AF[0].x - AF[1].x)/4;//x_center - 2*(scalable width)
	D[3].y = D[2].y;
	D[4].x = D[3].x;
	D[4].y = D[0].y;
	gvrender_polygon(job, D, sides + 1, filled);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case RESTRICTIONSITE:
	/*
	* zigzag shape, scales in the x-direction (only the middle section)
	*
	*
	*       ┌───D[2]
	*       │   └──────── D[0]
	*   ────┤            ├────
	*       └────────┐   │
	*       D[4]     └─── D[7]
	*
	*
	*
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides + 4, sizeof(pointf));
	D[0].x = mid_x(AF) + (AF[0].x - AF[1].x)/8 + (B[2].x-B[3].x)/2;//x_center + scalable_width + width
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/4;//y_center + 1/2 width
	D[1].x = mid_x(AF) - (AF[0].x - AF[1].x)/8; //x_center - width
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[2].x - (B[2].x-B[3].x)/2; //D[2].x - width
	D[3].y = D[2].y;
	D[4].x = D[3].x;
	D[4].y = mid_y(&AF[1]) - (B[3].y-B[4].y)/4; //y_center - 1/2(width)
	D[5].x = D[0].x - (B[2].x-B[3].x)/2;
	D[5].y = D[4].y;
	D[6].x = D[5].x;
	D[6].y = D[5].y - (B[3].y-B[4].y)/2;
	D[7].x = D[0].x;
	D[7].y = D[6].y;
	gvrender_polygon(job, D, sides + 4, filled);

	/*dsDNA line left half*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = D[4].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);

	/*dsDNA line right half*/
	C[0].x = D[7].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case FIVEPOVERHANG:
	/*
	*  does not scale, on the left side
	*
	*  D[3]──────D[2]
	*  │          │
	*  D[0]──────D[1]
	*        ┌────┐ ────────────
	*        │    │
	*       D[0]──D[1]
	*
	*
	*
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = AF[1].x;//the very left edge
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/8;//y_center + 1/4 width
	D[1].x = D[0].x + 2*(B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*second, lower shape*/
	free(D);
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = AF[1].x + (B[2].x-B[3].x);
	D[0].y = mid_y(&AF[1]) - (B[3].y-B[4].y)*5/8; //y_center - 5/4 width
	D[1].x = D[0].x + (B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*dsDNA line right half*/
	C[0].x = D[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case THREEPOVERHANG:
	/*
	*  does not scale, on the right side
	*
	*          D[2]──────D[1]
	*          │          │
	*  ─────── D[3]──────D[0]
	*          ┌────┐ D[1]
	*          │    │
	*          D[3]──D[0]
	*
	*
	*
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = AF[0].x;//the very right edge
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/8;//y_center + 1/4 width
	D[1].x = D[0].x;
	D[1].y = D[0].y + (B[3].y-B[4].y)/2;
	D[2].x = D[1].x - 2*(B[3].y-B[4].y);
	D[2].y = D[1].y;
	D[3].x = D[2].x;
	D[3].y = D[0].y;
	gvrender_polygon(job, D, sides, filled);

	/*second, lower shape*/
	free(D);
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = AF[0].x - (B[2].x-B[3].x);
	D[0].y = mid_y(&AF[1]) - (B[3].y-B[4].y)*5/8; //y_center - 5/4 width
	D[1].x = D[0].x;
	D[1].y = D[0].y + (B[3].y-B[4].y)/2;
	D[2].x = D[1].x - (B[3].y-B[4].y);
	D[2].y = D[1].y;
	D[3].x = D[2].x;
	D[3].y = D[0].y;
	gvrender_polygon(job, D, sides, filled);

	/*dsDNA line left half*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = D[3].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case NOVERHANG:
	/*
	*  does not scale
	*
	*     D[3]──────D[2]   D[3]──────D[2]
	*     │          │      │          │
	*  ───D[0]──────D[1]   D[0]──────D[1]────
	*     D[3]──────D[2]   D[3]──────D[2]
	*     │          │      │          │
	*     D[0]──────D[1]   D[0]──────D[1]
	*
	*
	*
	*
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	/*upper left rectangle*/
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) - (B[2].x-B[3].x)*9/8; //x_center - 2*width - 1/4*width
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/8;//y_center + 1/4 width
	D[1].x = D[0].x + (B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*lower, left rectangle*/
	free(D);
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) - (B[2].x-B[3].x)*9/8; //x_center - 2*width - 1/4*width
	D[0].y = mid_y(&AF[1]) - (B[3].y-B[4].y)*5/8;//y_center - width - 1/4 width
	D[1].x = D[0].x + (B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*lower, right rectangle*/
	free(D);
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/8; //x_center + 1/4*width
	D[0].y = mid_y(&AF[1]) - (B[3].y-B[4].y)*5/8;//y_center - width - 1/4 width
	D[1].x = D[0].x + (B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*upper, right rectangle*/
	free(D);
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/8; //x_center + 1/4*width
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/8;//y_center - width - 1/4 width
	D[1].x = D[0].x + (B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*dsDNA line right half*/
	C[0].x = D[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);

	/*dsDNA line left half*/
	C[0].x = mid_x(AF) - (B[2].x-B[3].x)*9/8; //D[0].x of of the left rectangles
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[1].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case ASSEMBLY:
	/*
	*  does not scale
	*
	*      D[3]──────────D[2]
	*      │               │
	*     D[0]──────────D[1]
	* ────                  ─────────
	*      D[3]──────────D[2]
	*      │               │
	*     D[0]──────────D[1]
	*
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) - (B[2].x-B[3].x); //x_center - 2*width
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/8;//y_center + 1/4 width
	D[1].x = D[0].x + 2*(B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*second, lower shape*/
	free(D);
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) - (B[2].x-B[3].x); //x_center - 2*width
	D[0].y = mid_y(&AF[1]) - (B[3].y-B[4].y)*5/8;//y_center - width - 1/4 width
	D[1].x = D[0].x + 2*(B[2].x-B[3].x);
	D[1].y = D[0].y;
	D[2].x = D[1].x;
	D[2].y = D[1].y + (B[3].y-B[4].y)/2;
	D[3].x = D[0].x;
	D[3].y = D[2].y;
	gvrender_polygon(job, D, sides, filled);

	/*dsDNA line right half*/
	C[0].x = D[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);

	/*dsDNA line left half*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = D[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case SIGNATURE:
	/*
	*
	*
	*   ┌──────────────┐
	*   │              │
	*   │x             │
	*   │_____________ │
	*   └──────────────┘
	*/
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	// the thickness is substituted with (AF[0].x - AF[1].x)/8 to make it scalable
	// in the y with label length
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = AF[0].x;
	D[0].y = B[1].y - (B[3].y - B[4].y)/2;
	D[1].x = B[3].x;
	D[1].y = B[3].y - (B[3].y - B[4].y)/2;
	D[2].x = AF[2].x;
	D[2].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[3].x = AF[0].x;
	D[3].y = AF[2].y + (B[3].y - B[4].y)/2;
	gvrender_polygon(job, D, sides, filled);

	/* "\" of the X*/
	C[0].x = AF[1].x + (B[2].x-B[3].x)/4;
	C[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/8; //y_center + 1/4 width
	C[1].x = C[0].x + (B[2].x-B[3].x)/4;//C[0].x + width/2
	C[1].y = C[0].y - (B[3].y-B[4].y)/4;//C[0].y - width/2
	gvrender_polyline(job, C, 2);

	/*"/" of the X*/
	C[0].x = AF[1].x + (B[2].x-B[3].x)/4;
	C[0].y = mid_y(&AF[1]) - (B[3].y-B[4].y)/8; //y_center - 1/4 width
	C[1].x = C[0].x + (B[2].x-B[3].x)/4;//C[0].x + width/2
	C[1].y = C[0].y + (B[3].y-B[4].y)/4;//C[0].y + width/2
	gvrender_polyline(job, C, 2);

	/*bottom line*/
	C[0].x = AF[1].x + (B[2].x-B[3].x)/4;
	C[0].y = AF[2].y + (B[3].y-B[4].y)*3/4;
	C[1].x = AF[0].x - (B[2].x-B[3].x)/4;
	C[1].y = C[0].y;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case INSULATOR:
	/*
	 * double square
	 *
	 *  ┌─────┐
	 *──┤ ┌─┐ ├───
	 *  │ └─┘ │
	 *  └─────┘
	 *
	 */
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	D = gv_calloc(sides, sizeof(pointf));
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/2; //x_center+width
	D[0].y = mid_y(&AF[1]) + (B[2].x-B[3].x)/2;
	D[1].x = D[0].x;
	D[1].y = mid_y(&AF[1]) - (B[2].x-B[3].x)/2; //D[0].y- width
	D[2].x = mid_x(AF) - (B[2].x-B[3].x)/2; //x_center-width
	D[2].y = D[1].y;
	D[3].x = D[2].x;
	D[3].y = D[0].y;
	gvrender_polygon(job, D, sides, filled);
	free(D);

	/*outer square line*/
	C[0].x = mid_x(AF) + (B[2].x-B[3].x)*3/4; //x_center+1.5*width
	C[0].y = mid_y(&AF[1]) + (B[2].x-B[3].x)*3/4; //y_center
	C[1].x = C[0].x;
	C[1].y = mid_y(&AF[1]) - (B[2].x-B[3].x)*3/4; //y_center- 1.5*width
	C[2].x = mid_x(AF) - (B[2].x-B[3].x)*3/4; //x_center-1.5*width
	C[2].y = C[1].y;
	C[3].x = C[2].x;
	C[3].y = C[0].y;
	C[4] = C[0];
	gvrender_polyline(job, C, 5);

	/*dsDNA line right half*/
	C[0].x = mid_x(AF) + (B[2].x-B[3].x)*3/4;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);

	/*dsDNA line left half*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = mid_x(AF) - (B[2].x-B[3].x)*3/4;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);

	break;
    case RIBOSITE:
	/*
	 * X with a dashed line on the bottom
	 *
	 *
	 *           X
	 *           ╎
	 *      ─────┴──────
	 */
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;

	D = gv_calloc(sides + 12, sizeof(pointf)); // 12-sided x
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/4; //x_center+width/2 , lower right corner of the x
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/2; //y_center + width
	D[1].x = D[0].x;
	D[1].y = D[0].y + (B[3].y-B[4].y)/8; //D[0].y +width/4
	D[2].x = D[0].x - (B[2].x-B[3].x)/8; //D[0].x- width/4 //right nook of the x
	D[2].y = D[1].y + (B[3].y-B[4].y)/8; //D[0].y+width/2 or D[1].y+width/4
	D[3].x = D[0].x;
	D[3].y = D[2].y + (B[3].y-B[4].y)/8; //D[2].y + width/4
	D[4].x = D[0].x;
	D[4].y = D[3].y + (B[3].y-B[4].y)/8; //top right corner of the x
	D[5].x = D[2].x;
	D[5].y = D[4].y;
	D[6].x = mid_x(AF);
	D[6].y = D[3].y; //top nook
	D[7].x = D[6].x - (B[2].x-B[3].x)/8; //D[5] mirrored across y
	D[7].y = D[5].y;
	D[8].x = D[7].x - (B[2].x-B[3].x)/8;//top left corner
	D[8].y = D[7].y;
	D[9].x = D[8].x;
	D[9].y = D[3].y;
	D[10].x = D[8].x + (B[2].x-B[3].x)/8;
	D[10].y = D[2].y;
	D[11].x = D[8].x;
	D[11].y = D[1].y;
	D[12].x = D[8].x;
	D[12].y = D[0].y;
	D[13].x = D[10].x;
	D[13].y = D[12].y;
	D[14].x = D[6].x; //bottom nook
	D[14].y = D[1].y;
	D[15].x = D[2].x;
	D[15].y = D[0].y;
	gvrender_polygon(job, D, sides + 12, filled);

	//2-part dash line

	/*line below the x, bottom dash*/
	C[0].x = D[14].x; //x_center
	C[0].y = mid_y(&AF[1]);
	C[1].x = C[0].x;
	C[1].y = C[0].y + (B[3].y-B[4].y)/8; //y_center + 1/4*width
	gvrender_polyline(job, C, 2);

	/*line below the x, top dash*/
	C[0].x = D[14].x; //x_center
	C[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/4;
	C[1].x = C[0].x;
	C[1].y = C[0].y + (B[3].y-B[4].y)/8;
	gvrender_polyline(job, C, 2);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case RNASTAB:
	/*
	 * octagon with a dashed line on the bottom
	 *
	 *
	 *           O
	 *           ╎
	 *      ─────┴──────
	 */
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;

	D = gv_calloc(sides + 4, sizeof(pointf)); // 12-sided x
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/8; //x_center+width/8 , lower right corner of the hexagon
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/2; //y_center + width
	D[1].x = D[0].x + (B[2].x-B[3].x)/8;
	D[1].y = D[0].y + (B[3].y-B[4].y)/8; //D[0].y +width/4
	D[2].x = D[1].x; //D[0].x- width/4
	D[2].y = D[1].y + (B[3].y-B[4].y)/4; //D[1].y+width/2
	D[3].x = D[0].x;
	D[3].y = D[2].y + (B[3].y-B[4].y)/8; //D[2].y + width/4
	D[4].x = D[3].x - (B[2].x-B[3].x)/4;
	D[4].y = D[3].y; //top of the hexagon
	D[5].x = D[4].x - (B[2].x-B[3].x)/8;
	D[5].y = D[2].y;
	D[6].x = D[5].x;
	D[6].y = D[1].y; //left side
	D[7].x = D[4].x;
	D[7].y = D[0].y; //bottom
	gvrender_polygon(job, D, sides + 4, filled);

	//2-part dash line

	/*line below the x, bottom dash*/
	C[0].x = mid_x(AF);
	C[0].y = mid_y(&AF[1]);
	C[1].x = C[0].x;
	C[1].y = C[0].y + (B[3].y-B[4].y)/8; //y_center + 1/4*width
	gvrender_polyline(job, C, 2);

	/*line below the x, top dash*/
	C[0].x = mid_x(AF);
	C[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/4;
	C[1].x = C[0].x;
	C[1].y = C[0].y + (B[3].y-B[4].y)/8;
	gvrender_polyline(job, C, 2);



	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case PROTEASESITE:
	/*
	 * X with a solid line on the bottom
	 *
	 *
	 *           X
	 *           │
	 *      ─────┴──────
	 */
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;
	D = gv_calloc(sides + 12, sizeof(pointf)); // 12-sided x
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/4; //x_center+width/2 , lower right corner of the x
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/2; //y_center + width
	D[1].x = D[0].x;
	D[1].y = D[0].y + (B[3].y-B[4].y)/8; //D[0].y +width/4
	D[2].x = D[0].x - (B[2].x-B[3].x)/8; //D[0].x- width/4 //right nook of the x
	D[2].y = D[1].y + (B[3].y-B[4].y)/8; //D[0].y+width/2 or D[1].y+width/4
	D[3].x = D[0].x;
	D[3].y = D[2].y + (B[3].y-B[4].y)/8; //D[2].y + width/4
	D[4].x = D[0].x;
	D[4].y = D[3].y + (B[3].y-B[4].y)/8; //top right corner of the x
	D[5].x = D[2].x;
	D[5].y = D[4].y;
	D[6].x = mid_x(AF);
	D[6].y = D[3].y; //top nook
	D[7].x = D[6].x - (B[2].x-B[3].x)/8; //D[5] mirrored across y
	D[7].y = D[5].y;
	D[8].x = D[7].x - (B[2].x-B[3].x)/8;//top left corner
	D[8].y = D[7].y;
	D[9].x = D[8].x;
	D[9].y = D[3].y;
	D[10].x = D[8].x + (B[2].x-B[3].x)/8;
	D[10].y = D[2].y;
	D[11].x = D[8].x;
	D[11].y = D[1].y;
	D[12].x = D[8].x;
	D[12].y = D[0].y;
	D[13].x = D[10].x;
	D[13].y = D[12].y;
	D[14].x = D[6].x; //bottom nook
	D[14].y = D[1].y;
	D[15].x = D[2].x;
	D[15].y = D[0].y;
	gvrender_polygon(job, D, sides + 12, filled);


	/*line below the x*/
	C[0] = D[14];
	C[1].x = C[0].x;
	C[1].y = mid_y(&AF[1]);
	gvrender_polyline(job, C, 2);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;
    case PROTEINSTAB:
	/*
	 * octagon with a solid line on the bottom
	 *
	 *
	 *           O
	 *           │
	 *      ─────┴──────
	 */
	//width units are (B[2].x-B[3].x)/2 or (B[3].y-B[4].y)/2;

	D = gv_calloc(sides + 4, sizeof(pointf)); // 12-sided x
	D[0].x = mid_x(AF) + (B[2].x-B[3].x)/8; //x_center+width/8 , lower right corner of the hexagon
	D[0].y = mid_y(&AF[1]) + (B[3].y-B[4].y)/2; //y_center + width
	D[1].x = D[0].x + (B[2].x-B[3].x)/8;
	D[1].y = D[0].y + (B[3].y-B[4].y)/8; //D[0].y +width/4
	D[2].x = D[1].x; //D[0].x- width/4
	D[2].y = D[1].y + (B[3].y-B[4].y)/4; //D[1].y+width/2
	D[3].x = D[0].x;
	D[3].y = D[2].y + (B[3].y-B[4].y)/8; //D[2].y + width/4
	D[4].x = D[3].x - (B[2].x-B[3].x)/4;
	D[4].y = D[3].y; //top of the hexagon
	D[5].x = D[4].x - (B[2].x-B[3].x)/8;
	D[5].y = D[2].y;
	D[6].x = D[5].x;
	D[6].y = D[1].y; //left side
	D[7].x = D[4].x;
	D[7].y = D[0].y; //bottom
	gvrender_polygon(job, D, sides + 4, filled);

	/*line below the x*/
	C[0].x = mid_x(AF);
	C[0].y = D[0].y;
	C[1].x = C[0].x;
	C[1].y = mid_y(&AF[1]);
	gvrender_polyline(job, C, 2);

	/*dsDNA line*/
	C[0].x = AF[1].x;
	C[0].y = mid_y(&AF[1]);
	C[1].x = AF[0].x;
	C[1].y = AF[2].y + (AF[0].y - AF[3].y)/2;
	gvrender_polyline(job, C, 2);
	free(D);

	break;

    case RPROMOTER:
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *
	 *      D[1] = AF[1]      │╲
	 *       ×────────────────× ╲
	 *       │              D[0] ╲
	 *       │                    ╲
	 *       │                    ╱
	 *       │                   ╱
	 *       │        ┌───────┐ ╱
	 *       │        │       │╱
	 *       └────────┘
	 */
	/* Add the tab edges. */
	D = gv_calloc(sides + 5, sizeof(pointf)); // 5 new points
	D[0].x = B[1].x - (B[2].x - B[3].x)/2;
	D[0].y = B[1].y - (B[3].y - B[4].y)/2;
	D[1].x = B[3].x;
	D[1].y = B[3].y - (B[3].y - B[4].y)/2;
	D[2].x = AF[2].x;
	D[2].y = AF[2].y;
	D[3].x = B[2].x + (B[2].x - B[3].x)/2;
	D[3].y = AF[2].y;
	D[4].x = B[2].x + (B[2].x - B[3].x)/2;
	D[4].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[5].x = B[1].x - (B[2].x - B[3].x)/2;
	D[5].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[6].x = B[1].x - (B[2].x - B[3].x)/2;
	D[6].y = AF[3].y;
	D[7].y = AF[0].y - (AF[0].y - AF[3].y)/2; /*triangle point */
	D[7].x = AF[0].x; /*triangle point */
	D[8].y = AF[0].y;
	D[8].x = B[1].x - (B[2].x - B[3].x)/2;

	gvrender_polygon(job, D, sides + 5, filled);
	free(D);
	break;

    case RARROW:
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *
	 *      D[1] = AF[1]      │╲
	 *       ×────────────────× ╲
	 *       │              D[0] ╲
	 *       │                    ╲
	 *       │                    ╱
	 *       │                   ╱
	 *       └────────────────┐ ╱
	 *                        │╱
	 *
	 */
	/* Add the tab edges. */
	D = gv_calloc(sides + 3, sizeof(pointf)); // 3 new points
	D[0].x = B[1].x - (B[2].x - B[3].x)/2;
	D[0].y = B[1].y - (B[3].y - B[4].y)/2;
	D[1].x = B[3].x;
	D[1].y = B[3].y - (B[3].y - B[4].y)/2;
	D[2].x = AF[2].x;
	D[2].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[3].x = B[1].x - (B[2].x - B[3].x)/2;
	D[3].y = AF[2].y + (B[3].y - B[4].y)/2;
	D[4].x = B[1].x - (B[2].x - B[3].x)/2;
	D[4].y = AF[3].y;
	D[5].y = AF[0].y - (AF[0].y - AF[3].y)/2;/*triangle point*/
	D[5].x = AF[0].x; /*triangle point */
	D[6].y = AF[0].y;
	D[6].x = B[1].x - (B[2].x - B[3].x)/2;

	gvrender_polygon(job, D, sides + 3, filled);
	free(D);
	break;

    case LARROW:
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *
	 *      ╱│
	 *     ╱ └────────────────┐
	 *    ╱                   │
	 *    ╲                   │
	 *     ╲ ┌────────────────┘
	 *      ╲│
	 *
	 */
	/* Add the tab edges. */
	D = gv_calloc(sides + 3, sizeof(pointf)); // 3 new points
	D[0].x = AF[0].x;
	D[0].y = AF[0].y - (B[3].y-B[4].y)/2;
	D[1].x = B[2].x + (B[2].x - B[3].x)/2;
	D[1].y = AF[0].y - (B[3].y-B[4].y)/2;/*D[0].y*/
	D[2].x = B[2].x + (B[2].x - B[3].x)/2;/*D[1].x*/
	D[2].y = B[2].y;
	D[3].x = AF[1].x; /*triangle point*/
	D[3].y = AF[1].y - (AF[1].y - AF[2].y)/2; /*triangle point*/
	D[4].x = B[2].x + (B[2].x - B[3].x)/2;/*D[1].x*/
	D[4].y = AF[2].y;
	D[5].y = AF[2].y + (B[3].y-B[4].y)/2;
	D[5].x = B[2].x + (B[2].x - B[3].x)/2;/*D[1].x*/
	D[6].y = AF[3].y + (B[3].y - B[4].y)/2;
	D[6].x = AF[0].x;/*D[0]*/

	gvrender_polygon(job, D, sides + 3, filled);
	free(D);
	break;

    case LPROMOTER:
	/*
	 * Adjust the perimeter for the protrusions.
	 *
	 *
	 *      ╱│
	 *     ╱ └────────────────×
	 *    ╱                 D[0]
	 *   ╱                    │
	 *   ╲                    │
	 *    ╲                   │
	 *     ╲ ┌────────┐       │
	 *      ╲│        │       │
	 *                └───────┘
	 */
	/* Add the tab edges. */
	D = gv_calloc(sides + 5, sizeof(pointf)); // 3 new points
	D[0].x = AF[0].x;
	D[0].y = AF[0].y - (B[3].y-B[4].y)/2;
	D[1].x = B[2].x + (B[2].x - B[3].x)/2;
	D[1].y = AF[0].y - (B[3].y-B[4].y)/2;/*D[0].y*/
	D[2].x = B[2].x + (B[2].x - B[3].x)/2;/*D[1].x*/
	D[2].y = B[2].y;
	D[3].x = AF[1].x; /*triangle point*/
	D[3].y = AF[1].y - (AF[1].y - AF[2].y)/2; /*triangle point*/
	D[4].x = B[2].x + (B[2].x - B[3].x)/2;/*D[1].x*/
	D[4].y = AF[2].y;
	D[5].y = AF[2].y + (B[3].y-B[4].y)/2;
	D[5].x = B[2].x + (B[2].x - B[3].x)/2;/*D[1].x*/
	D[6].y = AF[3].y + (B[3].y - B[4].y)/2;
	D[6].x = B[1].x - (B[2].x - B[3].x)/2;
	D[7].x = B[1].x - (B[2].x - B[3].x)/2;/*D[6].x*/
	D[7].y = AF[3].y;
	D[8].x = AF[3].x;
	D[8].y = AF[3].y;

	gvrender_polygon(job, D, sides + 5, filled);
	free(D);
	break;
    }
    free(B);
}

/*=============================poly start=========================*/

/* userSize;
 * Return maximum size, in points, of width and height supplied
 * by user, if any. Return 0 otherwise.
 */
static double userSize(node_t * n)
{
    double w, h;
    w = late_double(n, N_width, 0.0, MIN_NODEWIDTH);
    h = late_double(n, N_height, 0.0, MIN_NODEHEIGHT);
    return INCH2PS(fmax(w, h));
}

shape_kind shapeOf(node_t * n)
{
    shape_desc *sh = ND_shape(n);
    void (*ifn) (node_t *);

    if (!sh)
	return SH_UNSET;
    ifn = ND_shape(n)->fns->initfn;
    if (ifn == poly_init)
	return SH_POLY;
    else if (ifn == record_init)
	return SH_RECORD;
    else if (ifn == point_init)
	return SH_POINT;
    else if (ifn == epsf_init)
	return SH_EPSF;
    else
	return SH_UNSET;
}

bool isPolygon(node_t * n)
{
    return ND_shape(n) && ND_shape(n)->fns->initfn == poly_init;
}

static void poly_init(node_t * n)
{
    pointf dimen, min_bb;
    pointf outline_bb;
    point imagesize;
    pointf *vertices;
    char *p, *sfile, *fxd;
    double temp, alpha, beta, gamma;
    double orientation, distortion, skew;
    double scalex, scaley;
    double width, height, marginx, marginy, spacex;
    polygon_t *poly = gv_alloc(sizeof(polygon_t));
    bool isPlain = IS_PLAIN(n);

    bool regular = !!ND_shape(n)->polygon->regular;
    size_t peripheries = ND_shape(n)->polygon->peripheries;
    size_t sides = ND_shape(n)->polygon->sides;
    orientation = ND_shape(n)->polygon->orientation;
    skew = ND_shape(n)->polygon->skew;
    distortion = ND_shape(n)->polygon->distortion;
    regular |= mapbool(agget(n, "regular"));

    /* all calculations in floating point POINTS */

    /* make x and y dimensions equal if node is regular
     *   If the user has specified either width or height, use the max.
     *   Else use minimum default value.
     * If node is not regular, use the current width and height.
     */
    if (isPlain) {
	width = height = 0;
    }
    else if (regular) {
	double sz = userSize(n);
	if (sz > 0.0)
	    width = height = sz;
	else {
	    width = ND_width(n);
	    height = ND_height(n);
	    width = height = INCH2PS(fmin(width, height));
	}
    } else {
	width = INCH2PS(ND_width(n));
	height = INCH2PS(ND_height(n));
    }

    peripheries = (size_t)late_int(n, N_peripheries, (int)peripheries, 0);
    orientation += late_double(n, N_orientation, 0.0, -360.0);
    if (sides == 0) {		/* not for builtins */
	skew = late_double(n, N_skew, 0.0, -100.0);
	sides = (size_t)late_int(n, N_sides, 4, 0);
	distortion = late_double(n, N_distortion, 0.0, -100.0);
    }

    /* get label dimensions */
    dimen = ND_label(n)->dimen;

    /* minimal whitespace around label */
    if (dimen.x > 0 || dimen.y > 0) {
	/* padding */
	if (!isPlain) {
	    if ((p = agget(n, "margin"))) {
		marginx = marginy = 0;
		const int i = sscanf(p, "%lf,%lf", &marginx, &marginy);
		marginx = fmax(marginx, 0);
		marginy = fmax(marginy, 0);
		if (i > 0) {
		    dimen.x += 2 * INCH2PS(marginx);
		    if (i > 1)
			dimen.y += 2 * INCH2PS(marginy);
		    else
			dimen.y += 2 * INCH2PS(marginx);
		} else
		    PAD(dimen);
	    } else
		PAD(dimen);
	}
    }
    spacex = dimen.x - ND_label(n)->dimen.x;

    /* quantization */
    if ((temp = GD_drawing(agraphof(n))->quantum) > 0.0) {
	temp = INCH2PS(temp);
	dimen.x = quant(dimen.x, temp);
	dimen.y = quant(dimen.y, temp);
    }

    imagesize.x = imagesize.y = 0;
    if (ND_shape(n)->usershape) {
	/* custom requires a shapefile
	 * not custom is an adaptable user shape such as a postscript
	 * function.
	 */
	if (streq(ND_shape(n)->name, "custom")) {
	    sfile = agget(n, "shapefile");
	    imagesize = gvusershape_size(agraphof(n), sfile);
	    if (imagesize.x == -1 && imagesize.y == -1) {
		agwarningf(
		      "No or improper shapefile=\"%s\" for node \"%s\"\n",
		      sfile ? sfile : "<nil>", agnameof(n));
		imagesize.x = imagesize.y = 0;
	    } else {
		GD_has_images(agraphof(n)) = true;
		imagesize.x += 2;	/* some fixed padding */
		imagesize.y += 2;
	    }
	}
    } else if ((sfile = agget(n, "image")) && *sfile != '\0') {
	imagesize = gvusershape_size(agraphof(n), sfile);
	if (imagesize.x == -1 && imagesize.y == -1) {
	    agwarningf(
		  "No or improper image=\"%s\" for node \"%s\"\n",
		  sfile ? sfile : "<nil>", agnameof(n));
	    imagesize.x = imagesize.y = 0;
	} else {
	    GD_has_images(agraphof(n)) = true;
	    imagesize.x += 2;	/* some fixed padding */
	    imagesize.y += 2;
	}
    }

    /* initialize node bb to labelsize */
    pointf bb = {.x = fmax(dimen.x, imagesize.x),
                 .y = fmax(dimen.y, imagesize.y)};

    /* I don't know how to distort or skew ellipses in postscript */
    /* Convert request to a polygon with a large number of sides */
    if (sides <= 2 &&
        (!is_exactly_zero(distortion) || !is_exactly_zero(skew))) {
	sides = 120;
    }

    /* extra sizing depends on if label is centered vertically */
    p = agget(n, "labelloc");
    if (p && (p[0] == 't' || p[0] == 'b'))
	ND_label(n)->valign = p[0];
    else
	ND_label(n)->valign = 'c';

    const bool isBox = sides == 4 && fabs(fmod(orientation, 90)) < 0.5
                    && is_exactly_zero(distortion) && is_exactly_zero(skew);
    if (isBox) {
	/* for regular boxes the fit should be exact */
    } else if (ND_shape(n)->polygon->vertices) {
	poly_desc_t* pd = (poly_desc_t*)ND_shape(n)->polygon->vertices;
	bb = pd->size_gen(bb);
    } else {
	/* for all other shapes, compute a smallest ellipse
	 * containing bb centered on the origin, and then pad for that.
	 * We assume the ellipse is defined by a scaling up of bb.
	 */
	temp = bb.y * SQRT2;
	if (height > temp && ND_label(n)->valign == 'c') {
	    /* if there is height to spare
	     * and the label is centered vertically
	     * then just pad x in proportion to the spare height */
	    bb.x *= sqrt(1. / (1. - SQR(bb.y / height)));
	} else {
	    bb.x *= SQRT2;
	    bb.y = temp;
	}
#if 1
	if (sides > 2) {
	    temp = cos(M_PI / (double)sides);
	    bb.x /= temp;
	    bb.y /= temp;
	    /* FIXME - for odd-sided polygons, e.g. triangles, there
	       would be a better fit with some vertical adjustment of the shape */
	}
#endif
    }

    /* at this point, bb is the minimum size of node that can hold the label */
    min_bb = bb;

    /* increase node size to width/height if needed */
    fxd = late_string(n, N_fixed, "false");
    if (*fxd == 's' && streq(fxd,"shape")) {
	bb = (pointf){.x = width, .y = height};
	poly->option.fixedshape = true;
    } else if (mapbool(fxd)) {
	/* check only label, as images we can scale to fit */
	if (width < ND_label(n)->dimen.x || height < ND_label(n)->dimen.y)
	    agwarningf(
		  "node '%s', graph '%s' size too small for label\n",
		  agnameof(n), agnameof(agraphof(n)));
	bb = (pointf){.x = width, .y = height};
    } else {
	bb.x = width = fmax(width, bb.x);
	bb.y = height = fmax(height, bb.y);
    }

    /* If regular, make dimensions the same.
     * Need this to guarantee final node size is regular.
     */
    if (regular) {
	width = height = bb.x = bb.y = fmax(bb.x, bb.y);
    }

    /* Compute space available for label.  Provides the justification borders */
    if (!mapbool(late_string(n, N_nojustify, "false"))) {
	if (isBox) {
	    ND_label(n)->space.x = fmax(dimen.x, bb.x) - spacex;
	}
	else if (dimen.y < bb.y) {
	    temp = bb.x * sqrt(1.0 - SQR(dimen.y) / SQR(bb.y));
	    ND_label(n)->space.x = fmax(dimen.x, temp) - spacex;
        }
	else
	    ND_label(n)->space.x = dimen.x - spacex;
    } else {
	ND_label(n)->space.x = dimen.x - spacex;
    }

    if (!poly->option.fixedshape) {
	temp = bb.y - min_bb.y;
	if (dimen.y < imagesize.y)
	    temp += imagesize.y - dimen.y;
	ND_label(n)->space.y = dimen.y + temp;
    }

    const double penwidth = late_double(n, N_penwidth, DEFAULT_NODEPENWIDTH,
                                        MIN_NODEPENWIDTH);

    size_t outp = peripheries;
    if (peripheries < 1)
	outp = 1;

    if (peripheries >= 1 && penwidth > 0) {
        // allocate extra vertices representing the outline, i.e., the outermost
        // periphery with penwidth taken into account
        ++outp;
    }

    if (sides < 3) {		/* ellipses */
	sides = 2;
	vertices = gv_calloc(outp * sides, sizeof(pointf));
	pointf P = {.x = bb.x / 2., .y = bb.y / 2.};
	vertices[0] = (pointf){.x = -P.x, .y = -P.y};
	vertices[1] = P;
	if (peripheries > 1) {
	    for (size_t j = 1, i = 2; j < peripheries; j++) {
		P.x += GAP;
		P.y += GAP;
		vertices[i] = (pointf){.x = -P.x, .y = -P.y};
		i++;
		vertices[i] = P;
		i++;
	    }
	    bb.x = 2. * P.x;
	    bb.y = 2. * P.y;
	}
	outline_bb = bb;
	if (outp > peripheries) {
	  // add an outline at half the penwidth outside the outermost periphery
	  P.x += penwidth / 2;
	  P.y += penwidth / 2;
	  size_t i = sides * peripheries;
	  vertices[i] = (pointf){.x = -P.x, .y = -P.y};
	  i++;
	  vertices[i] = P;
	  i++;
	  outline_bb.x = 2. * P.x;
	  outline_bb.y = 2. * P.y;
	}
    } else {

/*
 * FIXME - this code is wrong - it doesn't work for concave boundaries.
 *          (e.g. "folder"  or "promoter")
 *   I don't think it even needs sectorangle, or knowledge of skewed shapes.
 *   (Concepts that only work for convex regular (modulo skew/distort) polygons.)
 *
 *   I think it only needs to know inside v. outside (by always drawing
 *   boundaries clockwise, say),  and the two adjacent segments.
 *
 *   It needs to find the point where the two lines, parallel to
 *   the current segments, and outside by GAP distance, intersect.
 */

	double sinx = 0, cosx = 0, xmax, ymax;
	vertices = gv_calloc(outp * sides, sizeof(pointf));
	if (ND_shape(n)->polygon->vertices) {
	    poly_desc_t* pd = (poly_desc_t*)ND_shape(n)->polygon->vertices;
	    pd->vertex_gen (vertices, &bb);
	    xmax = bb.x/2;
	    ymax = bb.y/2;
	} else {
	    double angle, sectorangle, sidelength, skewdist, gdistortion, gskew;
	    sectorangle = 2. * M_PI / (double)sides;
	    sidelength = sin(sectorangle / 2.);
	    skewdist = hypot(fabs(distortion) + fabs(skew), 1.);
	    gdistortion = distortion * SQRT2 / cos(sectorangle / 2.);
	    gskew = skew / 2.;
	    angle = (sectorangle - M_PI) / 2.;
	    sinx = sin(angle);
	    cosx = cos(angle);
	    pointf R = {.x = .5 * cosx, .y = .5 * sinx};
	    xmax = ymax = 0.;
	    angle += (M_PI - sectorangle) / 2.;
	    for (size_t i = 0; i < sides; i++) {

	    /*next regular vertex */
		angle += sectorangle;
		sinx = sin(angle);
		cosx = cos(angle);
		R.x += sidelength * cosx;
		R.y += sidelength * sinx;

	    /*distort and skew */
		pointf P = {
		  .x = R.x * (skewdist + R.y * gdistortion) + R.y * gskew,
		  .y = R.y};

	    /*orient P.x,P.y */
		alpha = RADIANS(orientation) + atan2(P.y, P.x);
		sinx = sin(alpha);
		cosx = cos(alpha);
		P.x = P.y = hypot(P.x, P.y);
		P.x *= cosx;
		P.y *= sinx;

	    /*scale for label */
		P.x *= bb.x;
		P.y *= bb.y;

	    /*find max for bounding box */
		xmax = fmax(fabs(P.x), xmax);
		ymax = fmax(fabs(P.y), ymax);

	    /* store result in array of points */
		vertices[i] = P;
		if (isBox) { /* enforce exact symmetry of box */
		    vertices[1] = (pointf){.x = -P.x, .y = P.y};
		    vertices[2] = (pointf){.x = -P.x, .y = -P.y};
		    vertices[3] = (pointf){.x = P.x, .y = -P.y};
		    break;
		}
	    }
	}

	/* apply minimum dimensions */
	xmax *= 2.;
	ymax *= 2.;
	bb = (pointf){.x = fmax(width, xmax), .y = fmax(height, ymax)};
	outline_bb = bb;

	scalex = bb.x / xmax;
	scaley = bb.y / ymax;

	size_t i;
	for (i = 0; i < sides; i++) {
	    pointf P = vertices[i];
	    P.x *= scalex;
	    P.y *= scaley;
	    vertices[i] = P;
	}

	if (outp > 1) {
	    pointf R = vertices[0];
	    pointf Q;
	    for (size_t j = 1; j < sides; j++) {
		Q = vertices[(i - j) % sides];
		if (!is_exactly_equal(Q.x, R.x) || !is_exactly_equal(Q.y, R.y)) {
		    break;
		}
	    }
	    assert(!is_exactly_equal(R.x, Q.x) || !is_exactly_equal(R.y, Q.y));
	    beta = atan2(R.y - Q.y, R.x - Q.x);
	    pointf Qprev = Q;
	    for (i = 0; i < sides; i++) {

		/*for each vertex find the bisector */
		Q = vertices[i];
		if (is_exactly_equal(Q.x, Qprev.x) && is_exactly_equal(Q.y, Qprev.y)) {
		    // The vertex points for the side ending at Q are equal,
		    // i.e. this side is actually a point and its angle is
		    // undefined. Therefore we keep the same offset for the end
		    // point as already calculated for the start point. This may
		    // occur for shapes which are represented as polygons during
		    // layout, but are drawn using Bézier curves during
		    // rendering, e.g. for the `cylinder` shape.
		} else {
		    for (size_t j = 1; j < sides; j++) {
			R = vertices[(i + j) % sides];
			if (!is_exactly_equal(R.x, Q.x) || !is_exactly_equal(R.y, Q.y)) {
			    break;
			}
		    }
		    assert(!is_exactly_equal(R.x, Q.x) || !is_exactly_equal(R.y, Q.y));
		    alpha = beta;
		    beta = atan2(R.y - Q.y, R.x - Q.x);
		    gamma = (alpha + M_PI - beta) / 2.;

		    /*find distance along bisector to */
		    /*intersection of next periphery */
		    temp = GAP / sin(gamma);

		    /*convert this distance to x and y */
		    sinx = sin(alpha - gamma) * temp;
		    cosx = cos(alpha - gamma) * temp;
		}
		assert(cosx != 0 || sinx != 0);
		Qprev = Q;

		/*save the vertices of all the */
		/*peripheries at this base vertex */
		for (size_t j = 1; j < peripheries; j++) {
		    Q.x += cosx;
		    Q.y += sinx;
		    vertices[i + j * sides] = Q;
		}
		if (outp > peripheries) {
		    // add an outline at half the penwidth outside the outermost periphery
		    Q.x += cosx * penwidth / 2 / GAP;
		    Q.y += sinx * penwidth / 2 / GAP;
		    vertices[i + peripheries * sides] = Q;
		}
	    }
	    for (i = 0; i < sides; i++) {
		pointf P = vertices[i + (peripheries - 1) * sides];
		bb = (pointf){.x = fmax(2.0 * fabs(P.x), bb.x),
		              .y = fmax(2.0 * fabs(P.y), bb.y)};
		Q = vertices[i + (outp - 1) * sides];
		outline_bb = (pointf){.x = fmax(2.0 * fabs(Q.x), outline_bb.x),
		                      .y = fmax(2.0 * fabs(Q.y), outline_bb.y)};
	    }
	}
    }
    poly->regular = regular;
    poly->peripheries = peripheries;
    poly->sides = sides;
    poly->orientation = orientation;
    poly->skew = skew;
    poly->distortion = distortion;
    poly->vertices = vertices;

    if (poly->option.fixedshape) {
	/* set width and height to reflect label and shape */
	ND_width(n) = PS2INCH(fmax(dimen.x, bb.x));
	ND_height(n) = PS2INCH(fmax(dimen.y, bb.y));
	ND_outline_width(n) = PS2INCH(fmax(dimen.x, outline_bb.x));
	ND_outline_height(n) = PS2INCH(fmax(dimen.y, outline_bb.y));
    } else {
	ND_width(n) = PS2INCH(bb.x);
	ND_height(n) = PS2INCH(bb.y);
	ND_outline_width(n) = PS2INCH(outline_bb.x);
	ND_outline_height(n) = PS2INCH(outline_bb.y);
    }
    ND_shape_info(n) = poly;
}

static void poly_free(node_t * n)
{
    polygon_t *p = ND_shape_info(n);

    if (p) {
	free(p->vertices);
	free(p);
    }
}

/* poly_inside:
 * Return true if point p is inside polygonal shape of node inside_context->s.n.
 * Calculations are done using unrotated node shape. Thus, if p is in a rotated
 * coordinate system, it is reset as P in the unrotated coordinate system. Similarly,
 * the ND_rw, ND_lw and ND_ht values are rotated if the graph is flipped.
 */
static bool poly_inside(inside_t * inside_context, pointf p)
{
    size_t sides;
    const pointf O = {0};
    pointf *vertex = NULL;

    int s;
    pointf P, Q, R;
    boxf *bp;
    node_t *n;

    if (!inside_context) {
	return false;
    }

    bp = inside_context->s.bp;
    n = inside_context->s.n;
    P = ccwrotatepf(p, 90 * GD_rankdir(agraphof(n)));

    /* Quick test if port rectangle is target */
    if (bp) {
	boxf bbox = *bp;
	return INSIDE(P, bbox);
    }

    if (n != inside_context->s.lastn) {
	double n_width, n_height;
	double n_outline_width;
	double n_outline_height;
	inside_context->s.last_poly = ND_shape_info(n);
	vertex = inside_context->s.last_poly->vertices;
	sides = inside_context->s.last_poly->sides;

	double xsize, ysize;
	if (inside_context->s.last_poly->option.fixedshape) {
	    boxf bb = polyBB(inside_context->s.last_poly);
	    n_width = bb.UR.x - bb.LL.x;
	    n_height = bb.UR.y - bb.LL.y;
	    n_outline_width = n_width;
	    n_outline_height = n_height;
	    /* get point and node size adjusted for rankdir=LR */
	    if (GD_flip(agraphof(n))) {
		ysize = n_width;
		xsize = n_height;
	    } else {
		xsize = n_width;
		ysize = n_height;
	    }
	} else {
	    /* get point and node size adjusted for rankdir=LR */
	    if (GD_flip(agraphof(n))) {
		ysize = ND_lw(n) + ND_rw(n);
		xsize = ND_ht(n);
	    } else {
		xsize = ND_lw(n) + ND_rw(n);
		ysize = ND_ht(n);
	    }
	    n_width = INCH2PS(ND_width(n));
	    n_height = INCH2PS(ND_height(n));
	    n_outline_width = INCH2PS(ND_outline_width(n));
	    n_outline_height = INCH2PS(ND_outline_height(n));
	}

	/* scale */
	inside_context->s.scalex = n_width;
	if (!is_exactly_zero(xsize))
	    inside_context->s.scalex /= xsize;
	inside_context->s.scaley = n_height;
	if (!is_exactly_zero(ysize))
	    inside_context->s.scaley /= ysize;
	inside_context->s.box_URx = n_outline_width / 2;
	inside_context->s.box_URy = n_outline_height / 2;

	const double penwidth = late_double(n, N_penwidth, DEFAULT_NODEPENWIDTH,
	                                    MIN_NODEPENWIDTH);
	if (inside_context->s.last_poly->peripheries >= 1 && penwidth > 0) {
	    /* index to outline, i.e., the outer-periphery with penwidth taken into account */
	    inside_context->s.outp =
	      (inside_context->s.last_poly->peripheries + 1 - 1) * sides;
	} else if (inside_context->s.last_poly->peripheries < 1) {
	    inside_context->s.outp = 0;
	} else {
	    /* index to outer-periphery */
	    inside_context->s.outp =
	      (inside_context->s.last_poly->peripheries - 1) * sides;
	}
	inside_context->s.lastn = n;
    } else {
	vertex = inside_context->s.last_poly->vertices;
	sides = inside_context->s.last_poly->sides;
    }

    /* scale */
    P.x *= inside_context->s.scalex;
    P.y *= inside_context->s.scaley;

    /* inside bounding box? */
    if (fabs(P.x) > inside_context->s.box_URx ||
        fabs(P.y) > inside_context->s.box_URy)
	return false;

    /* ellipses */
    if (sides <= 2)
	return hypot(P.x / inside_context->s.box_URx,
	             P.y / inside_context->s.box_URy) < 1;

    /* use fast test in case we are converging on a segment */
    size_t i = inside_context->s.last % sides; // in case last left over from larger polygon
    size_t i1 = (i + 1) % sides;
    Q = vertex[i + inside_context->s.outp];
    R = vertex[i1 + inside_context->s.outp];
    if (!same_side(P, O, Q, R))   /* false if outside the segment's face */
	return false;
    /* else inside the segment face... */
    if ((s = same_side(P, Q, R, O)) && same_side(P, R, O, Q)) /* true if between the segment's sides */
	return true;
    /* else maybe in another segment */
    for (size_t j = 1; j < sides; j++) { // iterate over remaining segments
	if (s) { /* clockwise */
	    i = i1;
	    i1 = (i + 1) % sides;
	} else { /* counter clockwise */
	    i1 = i;
	    i = (i + sides - 1) % sides;
	}
	if (!same_side(P, O, vertex[i + inside_context->s.outp],
	               vertex[i1 + inside_context->s.outp])) { // false if outside any other segment’s face
	    inside_context->s.last = i;
	    return false;
	}
    }
    /* inside all segments' faces */
    inside_context->s.last = i; // in case next edge is to same side
    return true;
}

static int poly_path(node_t * n, port * p, int side, boxf rv[], int *kptr)
{
  (void)n;
  (void)p;
  (void)side;
  (void)rv;
  (void)kptr;

  return 0;
}

static unsigned char invflip_side(unsigned char side, int rankdir) {
    switch (rankdir) {
    case RANKDIR_TB:
	break;
    case RANKDIR_BT:
	switch (side) {
	case TOP:
	    side = BOTTOM;
	    break;
	case BOTTOM:
	    side = TOP;
	    break;
	default:
	    break;
	}
	break;
    case RANKDIR_LR:
	switch (side) {
	case TOP:
	    side = RIGHT;
	    break;
	case BOTTOM:
	    side = LEFT;
	    break;
	case LEFT:
	    side = TOP;
	    break;
	case RIGHT:
	    side = BOTTOM;
	    break;
	default:
	    break;
	}
	break;
    case RANKDIR_RL:
	switch (side) {
	case TOP:
	    side = RIGHT;
	    break;
	case BOTTOM:
	    side = LEFT;
	    break;
	case LEFT:
	    side = BOTTOM;
	    break;
	case RIGHT:
	    side = TOP;
	    break;
	default:
	    break;
	}
	break;
    default:
	UNREACHABLE();
    }
    return side;
}

static double invflip_angle(double angle, int rankdir)
{
    switch (rankdir) {
    case RANKDIR_TB:
	break;
    case RANKDIR_BT:
	angle *= -1;
	break;
    case RANKDIR_LR:
	angle -= M_PI * 0.5;
	break;
    case RANKDIR_RL:
	if (angle == M_PI)
	    angle = -0.5 * M_PI;
	else if (angle == M_PI * 0.75)
	    angle = -0.25 * M_PI;
	else if (angle == M_PI * 0.5)
	    angle = 0;
	else if (angle == 0)
	    angle = M_PI * 0.5;
	else if (angle == M_PI * -0.25)
	    angle = M_PI * 0.75;
	else if (angle == M_PI * -0.5)
	    angle = M_PI;
	break;
    default:
	UNREACHABLE();
    }
    return angle;
}

/* compassPoint:
 * Compute compass points for non-trivial shapes.
 * It finds where the ray ((0,0),(x,y)) hits the boundary and
 * returns it.
 * Assumes ictxt and ictxt->n are non-NULL.
 *
 * bezier_clip uses the shape's _inside function, which assumes the input
 * point is in the rotated coordinate system (as determined by rankdir), so
 * it rotates the point counterclockwise based on rankdir to get the node's
 * coordinate system.
 * To handle this, if rankdir is set, we rotate (x,y) clockwise, and then
 * rotate the answer counterclockwise.
 */
static pointf compassPoint(inside_t * ictxt, double y, double x)
{
    pointf curve[4];		/* Bézier control points for a straight line */
    node_t *n = ictxt->s.n;
    graph_t* g = agraphof(n);
    int rd = GD_rankdir(g);
    pointf p;

    p.x = x;
    p.y = y;
    if (rd)
	p = cwrotatepf(p, 90 * rd);

    curve[0].x = curve[0].y = 0;
    curve[1] = curve[0];
    curve[3] = curve[2] = p;

    bezier_clip(ictxt, ND_shape(n)->fns->insidefn, curve, true);

    if (rd)
	curve[0] = ccwrotatepf(curve[0], 90 * rd);
    return curve[0];
}

/* compassPort:
 * Attach a compass point to a port pp, and fill in remaining fields.
 * n is the corresponding node; bp is the bounding box of the port.
 * compass is the compass point
 * Return 1 if unrecognized compass point, in which case we
 * use the center.
 *
 * This function also finishes initializing the port structure,
 * even if no compass point is involved.
 * The sides value gives the set of sides shared by the port. This
 * is used with a compass point to indicate if the port is exposed, to
 * set the port's side value.
 *
 * If ictxt is NULL, we are working with a simple rectangular shape (node or
 * port of record of HTML label), so compass points are trivial. If ictxt is
 * not NULL, it provides shape information so that the compass point can be
 * calculated based on the shape.
 *
 * The code assumes the node has its unrotated shape to find the points,
 * angles, etc. At the end, the parameters are adjusted to take into account
 * the rankdir attribute. In particular, the first if-else statement flips
 * the already adjusted ND_ht, ND_lw and ND_rw back to non-flipped values.
 *
 */
static int compassPort(node_t *n, boxf *bp, port *pp, const char *compass,
                       unsigned char sides, inside_t * ictxt) {
    boxf b;
    pointf p, ctr;
    int rv = 0;
    double theta = 0.0;
    bool constrain = false;
    bool dyna = false;
    unsigned char side = 0;
    bool clip = true;
    bool defined;
    double maxv;  /* sufficiently large value outside of range of node */

    if (bp) {
	b = *bp;
	p = (pointf){(b.LL.x + b.UR.x) / 2, (b.LL.y + b.UR.y) / 2};
	defined = true;
    } else {
	p.x = p.y = 0.;
	if (GD_flip(agraphof(n))) {
	    b.UR.x = ND_ht(n) / 2.;
	    b.LL.x = -b.UR.x;
	    b.UR.y = ND_lw(n);
	    b.LL.y = -b.UR.y;
	} else {
	    b.UR.y = ND_ht(n) / 2.;
	    b.LL.y = -b.UR.y;
	    b.UR.x = ND_lw(n);
	    b.LL.x = -b.UR.x;
	}
	defined = false;
    }
    maxv = fmax(b.UR.x, b.UR.y);
    maxv *= 4.0;
    ctr = p;
    if (compass && *compass) {
	switch (*compass++) {
	case 'e':
	    if (*compass)
		rv = 1;
	    else {
                if (ictxt)
                    p = compassPoint(ictxt, ctr.y, maxv);
                else
		    p.x = b.UR.x;
		theta = 0.0;
		constrain = true;
		defined = true;
		clip = false;
		side = sides & RIGHT;
	    }
	    break;
	case 's':
	    p.y = b.LL.y;
	    constrain = true;
	    clip = false;
	    switch (*compass) {
	    case '\0':
		theta = -M_PI * 0.5;
		defined = true;
                if (ictxt)
                    p = compassPoint(ictxt, -maxv, ctr.x);
                else
                    p.x = ctr.x;
		side = sides & BOTTOM;
		break;
	    case 'e':
		theta = -M_PI * 0.25;
		defined = true;
		if (ictxt)
		    p = compassPoint(ictxt, -maxv, maxv);
		else
		    p.x = b.UR.x;
		side = sides & (BOTTOM | RIGHT);
		break;
	    case 'w':
		theta = -M_PI * 0.75;
		defined = true;
		if (ictxt)
		    p = compassPoint(ictxt, -maxv, -maxv);
		else
		    p.x = b.LL.x;
		side = sides & (BOTTOM | LEFT);
		break;
	    default:
		p.y = ctr.y;
		constrain = false;
		clip = true;
		rv = 1;
		break;
	    }
	    break;
	case 'w':
	    if (*compass)
		rv = 1;
	    else {
                if (ictxt)
                    p = compassPoint(ictxt, ctr.y, -maxv);
                else
		    p.x = b.LL.x;
		theta = M_PI;
		constrain = true;
		defined = true;
		clip = false;
		side = sides & LEFT;
	    }
	    break;
	case 'n':
	    p.y = b.UR.y;
	    constrain = true;
	    clip = false;
	    switch (*compass) {
	    case '\0':
		defined = true;
		theta = M_PI * 0.5;
                if (ictxt)
                    p = compassPoint(ictxt, maxv, ctr.x);
                else
                    p.x = ctr.x;
		side = sides & TOP;
		break;
	    case 'e':
		defined = true;
		theta = M_PI * 0.25;
		if (ictxt)
		    p = compassPoint(ictxt, maxv, maxv);
		else
		    p.x = b.UR.x;
		side = sides & (TOP | RIGHT);
		break;
	    case 'w':
		defined = true;
		theta = M_PI * 0.75;
		if (ictxt)
		    p = compassPoint(ictxt, maxv, -maxv);
		else
		    p.x = b.LL.x;
		side = sides & (TOP | LEFT);
		break;
	    default:
		p.y = ctr.y;
		constrain = false;
		clip = true;
		rv = 1;
		break;
	    }
	    break;
	case '_':
	    dyna = true;
	    side = sides;
	    break;
	case 'c':
	    break;
	default:
	    rv = 1;
	    break;
	}
    }
    p = cwrotatepf(p, 90 * GD_rankdir(agraphof(n)));
    if (dyna)
	pp->side = side;
    else
	pp->side = invflip_side(side, GD_rankdir(agraphof(n)));
    pp->bp = bp;
    pp->p = p;
    pp->theta = invflip_angle(theta, GD_rankdir(agraphof(n)));
    if (p.x == 0 && p.y == 0)
	pp->order = MC_SCALE / 2;
    else {
	/* compute angle with 0 at north pole, increasing CCW */
	double angle = atan2(p.y, p.x) + 1.5 * M_PI;
	if (angle >= 2 * M_PI)
	    angle -= 2 * M_PI;
	pp->order = (int) (MC_SCALE * angle / (2 * M_PI));
    }
    pp->constrained = constrain;
    pp->defined = defined;
    pp->clip = clip;
    pp->dyna = dyna;
    return rv;
}

static port poly_port(node_t * n, char *portname, char *compass)
{
    port rv;
    boxf *bp;
    unsigned char sides; // bitmap of which sides the port lies along

    if (portname[0] == '\0')
	return Center;

    if (compass == NULL)
	compass = "_";
    sides = BOTTOM | RIGHT | TOP | LEFT;
    if (ND_label(n)->html && (bp = html_port(n, portname, &sides))) {
	if (compassPort(n, bp, &rv, compass, sides, NULL)) {
	    agwarningf(
		  "node %s, port %s, unrecognized compass point '%s' - ignored\n",
		  agnameof(n), portname, compass);
	}
    } else {
	inside_t *ictxtp;
	inside_t ictxt = {0};

	if (IS_BOX(n))
	    ictxtp = NULL;
	else {
	    ictxt.s.n = n;
	    ictxt.s.bp = NULL;
	    ictxtp = &ictxt;
	}
	if (compassPort(n, NULL, &rv, portname, sides, ictxtp))
	    unrecognized(n, portname);
    }

    rv.name = NULL;
    return rv;
}

static bool multicolor(const char *f) {
  return strchr(f, ':') != NULL;
}

/* generic polygon gencode routine */
static void poly_gencode(GVJ_t * job, node_t * n)
{
    obj_state_t *obj = job->obj;
    polygon_t *poly;
    double xsize, ysize;
    pointf P, *vertices;
    int filled;
    bool usershape_p;
    bool pfilled;		/* true if fill not handled by user shape */
    char *color, *name;
    int doMap = (obj->url || obj->explicit_tooltip);
    char* fillcolor=NULL;
    char* pencolor=NULL;

    if (doMap && !(job->flags & EMIT_CLUSTERS_LAST))
	gvrender_begin_anchor(job,
			      obj->url, obj->tooltip, obj->target,
			      obj->id);

    poly = ND_shape_info(n);
    vertices = poly->vertices;
    const size_t sides = poly->sides;
    size_t peripheries = poly->peripheries;
    pointf *AF = gv_calloc(sides + 5, sizeof(pointf));

    /* nominal label position in the center of the node */
    ND_label(n)->pos = ND_coord(n);

    xsize = (ND_lw(n) + ND_rw(n)) / INCH2PS(ND_width(n));
    ysize = ND_ht(n) / INCH2PS(ND_height(n));

    const graphviz_polygon_style_t style = stylenode(job, n);

    char *clrs[2] = {0};
    if (ND_gui_state(n) & GUI_STATE_ACTIVE) {
	pencolor = DEFAULT_ACTIVEPENCOLOR;
	gvrender_set_pencolor(job, pencolor);
	color = DEFAULT_ACTIVEFILLCOLOR;
	gvrender_set_fillcolor(job, color);
	filled = FILL;
    } else if (ND_gui_state(n) & GUI_STATE_SELECTED) {
	pencolor = DEFAULT_SELECTEDPENCOLOR;
	gvrender_set_pencolor(job, pencolor);
	color = DEFAULT_SELECTEDFILLCOLOR;
	gvrender_set_fillcolor(job, color);
	filled = FILL;
    } else if (ND_gui_state(n) & GUI_STATE_DELETED) {
	pencolor = DEFAULT_DELETEDPENCOLOR;
	gvrender_set_pencolor(job, pencolor);
	color = DEFAULT_DELETEDFILLCOLOR;
	gvrender_set_fillcolor(job, color);
	filled = FILL;
    } else if (ND_gui_state(n) & GUI_STATE_VISITED) {
	pencolor = DEFAULT_VISITEDPENCOLOR;
	gvrender_set_pencolor(job, pencolor);
	color = DEFAULT_VISITEDFILLCOLOR;
	gvrender_set_fillcolor(job, color);
	filled = FILL;
    } else {
	if (style.filled) {
	    double frac;
	    fillcolor = findFill (n);
	    if (findStopColor (fillcolor, clrs, &frac)) {
        	gvrender_set_fillcolor(job, clrs[0]);
		if (clrs[1])
		    gvrender_set_gradient_vals(job,clrs[1],late_int(n,N_gradientangle,0,0), frac);
		else
		    gvrender_set_gradient_vals(job,DEFAULT_COLOR,late_int(n,N_gradientangle,0,0), frac);
		if (style.radial)
		    filled = RGRADIENT;
	 	else
		    filled = GRADIENT;
	    }
	    else {
        	gvrender_set_fillcolor(job, fillcolor);
		filled = FILL;
	    }
	}
	else if (style.striped || style.wedged)  {
	    fillcolor = findFill (n);
	    filled = 1;
	}
	else {
	    filled = 0;
	}
	pencolor = penColor(job, n);	/* emit pen color */
    }

    pfilled = !ND_shape(n)->usershape || streq(ND_shape(n)->name, "custom");

    /* if no boundary but filled, set boundary color to transparent */
    if (peripheries == 0 && filled != 0 && pfilled) {
	peripheries = 1;
	gvrender_set_pencolor(job, "transparent");
    }

    /* draw peripheries first */
    size_t j;
    for (j = 0; j < peripheries; j++) {
	for (size_t i = 0; i < sides; i++) {
	    P = vertices[i + j * sides];
	    AF[i].x = P.x * xsize + ND_coord(n).x;
	    AF[i].y = P.y * ysize + ND_coord(n).y;
	}
	if (sides <= 2) {
	    if (style.wedged && j == 0 && multicolor(fillcolor)) {
		int rv = wedgedEllipse (job, AF, fillcolor);
		if (rv > 1)
		    agerr (AGPREV, "in node %s\n", agnameof(n));
		filled = 0;
	    }
	    gvrender_ellipse(job, AF, filled);
	    if (style.diagonals) {
		Mcircle_hack(job, n);
	    }
	} else if (style.striped) {
	    if (j == 0) {
		int rv = stripedBox (job, AF, fillcolor, 1);
		if (rv > 1)
		    agerr (AGPREV, "in node %s\n", agnameof(n));
	    }
	    gvrender_polygon(job, AF, sides, 0);
	} else if (style.underline) {
	    gvrender_set_pencolor(job, "transparent");
	    gvrender_polygon(job, AF, sides, filled);
	    gvrender_set_pencolor(job, pencolor);
	    gvrender_polyline(job, AF+2, 2);
	} else if (SPECIAL_CORNERS(style)) {
	    round_corners(job, AF, sides, style, filled);
	} else {
	    gvrender_polygon(job, AF, sides, filled);
	}
	/* fill innermost periphery only */
	filled = 0;
    }

    usershape_p = false;
    if (ND_shape(n)->usershape) {
	name = ND_shape(n)->name;
	if (streq(name, "custom")) {
	    if ((name = agget(n, "shapefile")) && name[0])
		usershape_p = true;
	} else
	    usershape_p = true;
    } else if ((name = agget(n, "image")) && name[0]) {
	usershape_p = true;
    }
    if (usershape_p) {
	/* get coords of innermost periphery */
	for (size_t i = 0; i < sides; i++) {
	    P = vertices[i];
	    AF[i].x = P.x * xsize + ND_coord(n).x;
	    AF[i].y = P.y * ysize + ND_coord(n).y;
	}
	/* lay down fill first */
	if (filled != 0 && pfilled) {
	    if (sides <= 2) {
		if (style.wedged && j == 0 && multicolor(fillcolor)) {
		    int rv = wedgedEllipse (job, AF, fillcolor);
		    if (rv > 1)
			agerr (AGPREV, "in node %s\n", agnameof(n));
		    filled = 0;
		}
		gvrender_ellipse(job, AF, filled);
		if (style.diagonals) {
		    Mcircle_hack(job, n);
		}
	    } else if (style.striped) {
		int rv = stripedBox (job, AF, fillcolor, 1);
		if (rv > 1)
		    agerr (AGPREV, "in node %s\n", agnameof(n));
		gvrender_polygon(job, AF, sides, 0);
	    } else if (style.rounded || style.diagonals) {
		round_corners(job, AF, sides, style, filled);
	    } else {
		gvrender_polygon(job, AF, sides, filled);
	    }
	}
	gvrender_usershape(job, name, AF, sides, filled != 0,
			   late_string(n, N_imagescale, "false"),
			   late_string(n, N_imagepos, "mc"));
	filled = 0;		/* with user shapes, we have done the fill if needed */
    }

    free(AF);
    free (clrs[0]);
    free (clrs[1]);

    emit_label(job, EMIT_NLABEL, ND_label(n));
    if (doMap) {
	if (job->flags & EMIT_CLUSTERS_LAST)
	    gvrender_begin_anchor(job,
				  obj->url, obj->tooltip, obj->target,
				  obj->id);
	gvrender_end_anchor(job);
    }
}

/*=======================end poly======================================*/

/*===============================point start========================*/

/* point_init:
 * shorthand for shape=circle, style=filled, width=0.05, label=""
 */
static void point_init(node_t * n)
{
    polygon_t *poly = gv_alloc(sizeof(polygon_t));
    size_t sides, outp, peripheries = ND_shape(n)->polygon->peripheries;
    double sz;
    pointf P, *vertices;
    size_t i, j;
    double w, h;

    /* set width and height, and make them equal
     * if user has set weight or height, use it.
     * if both are set, use smallest.
     * if neither, use default
     */
    w = late_double(n, N_width, DBL_MAX, MIN_NODEWIDTH);
    h = late_double(n, N_height, DBL_MAX, MIN_NODEHEIGHT);
    w = fmin(w, h);
    if (is_exactly_equal(w, DBL_MAX) &&
        is_exactly_equal(h, DBL_MAX)) // neither defined
	ND_width(n) = ND_height(n) = DEF_POINT;
    else {
	w = fmin(w, h);
	/* If w == 0, use it; otherwise, make w no less than MIN_POINT due
         * to the restrictions mentioned above.
         */
	if (w > 0.0)
	    w = fmax(w, MIN_POINT);
	ND_width(n) = ND_height(n) = w;
    }

    sz = ND_width(n) * POINTS_PER_INCH;
    peripheries = (size_t)late_int(n, N_peripheries, (int)peripheries, 0);
    if (peripheries < 1)
	outp = 1;
    else
	outp = peripheries;
    sides = 2;
    const double penwidth = late_double(n, N_penwidth, DEFAULT_NODEPENWIDTH,
                                        MIN_NODEPENWIDTH);
    if (peripheries >= 1 && penwidth > 0) {
        // allocate extra vertices representing the outline, i.e., the outermost
        // periphery with penwidth taken into account
        ++outp;
    }
    vertices = gv_calloc(outp * sides, sizeof(pointf));
    P.y = P.x = sz / 2.;
    vertices[0].x = -P.x;
    vertices[0].y = -P.y;
    vertices[1] = P;
    if (peripheries > 1) {
	for (j = 1, i = 2; j < peripheries; j++) {
	    P.x += GAP;
	    P.y += GAP;
	    vertices[i].x = -P.x;
	    vertices[i].y = -P.y;
	    i++;
	    vertices[i].x = P.x;
	    vertices[i].y = P.y;
	    i++;
	}
	sz = 2. * P.x;
    } else {
        i = sides;
    }

    if (peripheries >= 1 && penwidth > 0 && outp > peripheries) {
      // add an outline at half the penwidth outside the outermost periphery
      P.x += penwidth / 2;
      P.y += penwidth / 2;
      vertices[i].x = -P.x;
      vertices[i].y = -P.y;
      i++;
      vertices[i].x = P.x;
      vertices[i].y = P.y;
      i++;
    }
    const double sz_outline = 2. * P.x;

    poly->regular = true;
    poly->peripheries = peripheries;
    poly->sides = 2;
    poly->orientation = 0;
    poly->skew = 0;
    poly->distortion = 0;
    poly->vertices = vertices;

    ND_height(n) = ND_width(n) = PS2INCH(sz);
    ND_outline_height(n) = ND_outline_width(n) = PS2INCH(sz_outline);
    ND_shape_info(n) = poly;
}

static bool point_inside(inside_t * inside_context, pointf p)
{
    pointf P;
    node_t *n;

    if (!inside_context) {
	return false;
    }

    n = inside_context->s.n;
    P = ccwrotatepf(p, 90 * GD_rankdir(agraphof(n)));

    if (n != inside_context->s.lastn) {
	size_t outp;
	polygon_t *poly = ND_shape_info(n);
	const size_t sides = 2;
	const double penwidth = late_double(n, N_penwidth, DEFAULT_NODEPENWIDTH,
	                                    MIN_NODEPENWIDTH);

	if (poly->peripheries >= 1 && penwidth > 0) {
	    /* index to outline, i.e., the outer-periphery with penwidth taken into account */
	    outp = sides * (poly->peripheries + 1 - 1);
	} else if (poly->peripheries < 1) {
	    outp = 0;
	} else {
	    /* index to outer-periphery */
	    outp = sides * (poly->peripheries - 1);
	}

	inside_context->s.radius = poly->vertices[outp + 1].x;
	inside_context->s.lastn = n;
    }

    /* inside bounding box? */
    if (fabs(P.x) > inside_context->s.radius ||
        fabs(P.y) > inside_context->s.radius)
	return false;

    return hypot(P.x, P.y) <= inside_context->s.radius;
}

static void point_gencode(GVJ_t * job, node_t * n)
{
    obj_state_t *obj = job->obj;
    polygon_t *poly;
    pointf P, *vertices;
    bool filled;
    char *color;
    int doMap = obj->url || obj->explicit_tooltip;

    if (doMap && !(job->flags & EMIT_CLUSTERS_LAST))
	gvrender_begin_anchor(job,
			      obj->url, obj->tooltip, obj->target,
			      obj->id);

    poly = ND_shape_info(n);
    vertices = poly->vertices;
    const size_t sides = poly->sides;
    size_t peripheries = poly->peripheries;

    graphviz_polygon_style_t style = {0};
    checkStyle(n, &style);
    if (style.invisible)
	gvrender_set_style(job, point_style);
    else
	gvrender_set_style(job, &point_style[1]);
    if (N_penwidth)
	gvrender_set_penwidth(job, late_double(n, N_penwidth, 1.0, 0.0));

    if (ND_gui_state(n) & GUI_STATE_ACTIVE) {
	color = DEFAULT_ACTIVEPENCOLOR;
	gvrender_set_pencolor(job, color);
	color = DEFAULT_ACTIVEFILLCOLOR;
	gvrender_set_fillcolor(job, color);
    } else if (ND_gui_state(n) & GUI_STATE_SELECTED) {
	color = DEFAULT_SELECTEDPENCOLOR;
	gvrender_set_pencolor(job, color);
	color = DEFAULT_SELECTEDFILLCOLOR;
	gvrender_set_fillcolor(job, color);
    } else if (ND_gui_state(n) & GUI_STATE_DELETED) {
	color = DEFAULT_DELETEDPENCOLOR;
	gvrender_set_pencolor(job, color);
	color = DEFAULT_DELETEDFILLCOLOR;
	gvrender_set_fillcolor(job, color);
    } else if (ND_gui_state(n) & GUI_STATE_VISITED) {
	color = DEFAULT_VISITEDPENCOLOR;
	gvrender_set_pencolor(job, color);
	color = DEFAULT_VISITEDFILLCOLOR;
	gvrender_set_fillcolor(job, color);
    } else {
	color = findFillDflt(n, "black");
	gvrender_set_fillcolor(job, color);	/* emit fill color */
	penColor(job, n);	/* emit pen color */
    }
    filled = true;

    /* if no boundary but filled, set boundary color to fill color */
    if (peripheries == 0) {
	peripheries = 1;
	if (color[0])
	    gvrender_set_pencolor(job, color);
    }

    for (size_t j = 0; j < peripheries; j++) {
	enum {A_size = 2};
	pointf AF[A_size] = {{0}};
	for (size_t i = 0; i < sides; i++) {
	    P = vertices[i + j * sides];
	    if (i < A_size) {
	      AF[i].x = P.x + ND_coord(n).x;
	      AF[i].y = P.y + ND_coord(n).y;
	    }
	}
	gvrender_ellipse(job, AF, filled);
	/* fill innermost periphery only */
	filled = false;
    }

    if (doMap) {
	if (job->flags & EMIT_CLUSTERS_LAST)
	    gvrender_begin_anchor(job,
				  obj->url, obj->tooltip, obj->target,
				  obj->id);
	gvrender_end_anchor(job);
    }
}

/* the "record" shape is a rudimentary table formatter */

#define HASTEXT 1
#define HASPORT 2
#define HASTABLE 4
#define INTEXT 8
#define INPORT 16

static bool ISCTRL(int c) {
  return c == '{' || c == '}' || c == '|' || c == '<' || c == '>';
}

static char *reclblp;

static void free_field(field_t * f)
{
    int i;

    for (i = 0; i < f->n_flds; i++) {
	free_field(f->fld[i]);
    }

    free(f->id);
    free_label(f->lp);
    free(f->fld);
    free(f);
}

/* parse_error:
 * Clean up memory allocated in parse_reclbl, then return NULL
 */
static field_t *parse_error(field_t *rv, char *portname) {
    free_field(rv);
    free(portname);
    return NULL;
}

static field_t *parse_reclbl(node_t *n, bool LR, bool flag, char *text) {
    field_t *fp, *rv = gv_alloc(sizeof(field_t));
    char *tsp, *psp=NULL, *hstsp, *hspsp=NULL, *sp;
    char *tmpport = NULL;
    int cnt, mode, fi;
    textlabel_t *lbl = ND_label(n);
    unsigned char uc;

    fp = NULL;
    size_t maxf;
    for (maxf = 1, cnt = 0, sp = reclblp; *sp; sp++) {
	if (*sp == '\\') {
	    sp++;
	    if (*sp && (*sp == '{' || *sp == '}' || *sp == '|' || *sp == '\\'))
		continue;
	}
	if (*sp == '{')
	    cnt++;
	else if (*sp == '}')
	    cnt--;
	else if (*sp == '|' && cnt == 0)
	    maxf++;
	if (cnt < 0)
	    break;
    }
    rv->fld = gv_calloc(maxf, sizeof(field_t*));
    rv->LR = LR;
    mode = 0;
    fi = 0;
    hstsp = tsp = text;
    bool wflag = true;
    bool ishardspace = false;
    while (wflag) {
	if ((uc = *(unsigned char*)reclblp) && uc < ' ') {    /* Ignore non-0 control characters */
	    reclblp++;
	    continue;
	}
	switch (*reclblp) {
	case '<':
	    if (mode & (HASTABLE | HASPORT))
		return parse_error(rv, tmpport);
	    if (lbl->html)
		goto dotext;
	    mode |= (HASPORT | INPORT);
	    reclblp++;
	    hspsp = psp = text;
	    break;
	case '>':
	    if (lbl->html)
		goto dotext;
	    if (!(mode & INPORT))
		return parse_error(rv, tmpport);
	    if (psp > text + 1 && psp - 1 != hspsp && *(psp - 1) == ' ')
		psp--;
	    *psp = '\000';
	    tmpport = gv_strdup(text);
	    mode &= ~INPORT;
	    reclblp++;
	    break;
	case '{':
	    reclblp++;
	    if (mode != 0 || !*reclblp)
		return parse_error(rv, tmpport);
	    mode = HASTABLE;
	    if (!(rv->fld[fi++] = parse_reclbl(n, !LR, false, text)))
		return parse_error(rv, tmpport);
	    break;
	case '}':
	case '|':
	case '\000':
	    if ((!*reclblp && !flag) || (mode & INPORT))
		return parse_error(rv, tmpport);
	    if (!(mode & HASTABLE))
		fp = rv->fld[fi++] = gv_alloc(sizeof(field_t));
	    if (tmpport) {
		fp->id = tmpport;
		tmpport = NULL;
	    }
	    if (!(mode & (HASTEXT | HASTABLE))) {
		mode |= HASTEXT;
		*tsp++ = ' ';
	    }
	    if (mode & HASTEXT) {
		if (tsp > text + 1 && tsp - 1 != hstsp && *(tsp - 1) == ' ')
		    tsp--;
		*tsp = '\000';
		fp->lp = make_label(n, text, lbl->html, false,
			       lbl->fontsize, lbl->fontname, lbl->fontcolor);
		fp->LR = true;
		hstsp = tsp = text;
	    }
	    if (*reclblp) {
		if (*reclblp == '}') {
		    reclblp++;
		    rv->n_flds = fi;
		    return rv;
		}
		mode = 0;
		reclblp++;
	    } else
		wflag = false;
	    break;
	case '\\':
	    if (*(reclblp + 1)) {
		if (ISCTRL(*(reclblp + 1))) {
		    // nothing
		} else if (*(reclblp + 1) == ' ' && !lbl->html)
		    ishardspace = true;
		else {
		    *tsp++ = '\\';
		    mode |= INTEXT | HASTEXT;
		}
		reclblp++;
	    }
	    /* fall through */
	default:
	  dotext:
	    if ((mode & HASTABLE) && *reclblp != ' ')
		return parse_error(rv, tmpport);
	    if (!(mode & (INTEXT | INPORT)) && *reclblp != ' ')
		mode |= INTEXT | HASTEXT;
	    if (mode & INTEXT) {
		if (!(*reclblp == ' ' && !ishardspace && *(tsp - 1) == ' '
		     && !lbl->html))
		    *tsp++ = *reclblp;
		if (ishardspace)
		    hstsp = tsp - 1;
	    } else if (mode & INPORT) {
		if (!(*reclblp == ' ' && !ishardspace &&
		      (psp == text || *(psp - 1) == ' ')))
		    *psp++ = *reclblp;
		if (ishardspace)
		    hspsp = psp - 1;
	    }
	    reclblp++;
	    while ((*reclblp & 0xc0) == 0x80)
		*tsp++ = *reclblp++;
	    break;
	}
    }
    rv->n_flds = fi;
    return rv;
}

static pointf size_reclbl(node_t * n, field_t * f)
{
    int i;
    char *p;
    double marginx, marginy;
    pointf d, d0;
    pointf dimen;

    if (f->lp) {
	dimen = f->lp->dimen;

	/* minimal whitespace around label */
	if (dimen.x > 0.0 || dimen.y > 0.0) {
	    /* padding */
	    if ((p = agget(n, "margin"))) {
		i = sscanf(p, "%lf,%lf", &marginx, &marginy);
		if (i > 0) {
		    dimen.x += 2 * INCH2PS(marginx);
		    if (i > 1)
			dimen.y += 2 * INCH2PS(marginy);
		    else
			dimen.y += 2 * INCH2PS(marginx);
		} else
		    PAD(dimen);
	    } else
		PAD(dimen);
	}
	d = dimen;
    } else {
	d.x = d.y = 0;
	for (i = 0; i < f->n_flds; i++) {
	    d0 = size_reclbl(n, f->fld[i]);
	    if (f->LR) {
		d.x += d0.x;
		d.y = fmax(d.y, d0.y);
	    } else {
		d.y += d0.y;
		d.x = fmax(d.x, d0.x);
	    }
	}
    }
    f->size = d;
    return d;
}

static void resize_reclbl(field_t *f, pointf sz, bool nojustify_p) {
    int i, amt;
    double inc;
    pointf d;
    pointf newsz;
    field_t *sf;

    /* adjust field */
    d.x = sz.x - f->size.x;
    d.y = sz.y - f->size.y;
    f->size = sz;

    /* adjust text area */
    if (f->lp && !nojustify_p) {
	f->lp->space.x += d.x;
	f->lp->space.y += d.y;
    }

    /* adjust children */
    if (f->n_flds) {

	if (f->LR)
	    inc = d.x / f->n_flds;
	else
	    inc = d.y / f->n_flds;
	for (i = 0; i < f->n_flds; i++) {
	    sf = f->fld[i];
	    amt = (int)((i + 1) * inc) - (int)(i * inc);
	    if (f->LR)
		newsz = (pointf){sf->size.x + amt, sz.y};
	    else
		newsz = (pointf){sz.x, sf->size.y + amt};
	    resize_reclbl(sf, newsz, nojustify_p);
	}
    }
}

/* pos_reclbl:
 * Assign position info for each field. Also, set
 * the sides attribute, which indicates which sides of the
 * record are accessible to the field.
 */
static void pos_reclbl(field_t *f, pointf ul, unsigned char sides) {
    int i, last;
    unsigned char mask;

    f->sides = sides;
    f->b.LL = (pointf){ul.x, ul.y - f->size.y};
    f->b.UR = (pointf){ul.x + f->size.x, ul.y};
    last = f->n_flds - 1;
    for (i = 0; i <= last; i++) {
	if (sides) {
	    if (f->LR) {
		if (i == 0) {
		    if (i == last)
			mask = TOP | BOTTOM | RIGHT | LEFT;
		    else
			mask = TOP | BOTTOM | LEFT;
		} else if (i == last)
		    mask = TOP | BOTTOM | RIGHT;
		else
		    mask = TOP | BOTTOM;
	    } else {
		if (i == 0) {
		    if (i == last)
			mask = TOP | BOTTOM | RIGHT | LEFT;
		    else
			mask = TOP | RIGHT | LEFT;
		} else if (i == last)
		    mask = LEFT | BOTTOM | RIGHT;
		else
		    mask = LEFT | RIGHT;
	    }
	} else
	    mask = 0;
	pos_reclbl(f->fld[i], ul, (unsigned char)(sides & mask));
	if (f->LR)
	    ul.x = ul.x + f->fld[i]->size.x;
	else
	    ul.y = ul.y - f->fld[i]->size.y;
    }
}

#if defined(DEBUG) && DEBUG > 1
static void indent(int l)
{
    int i;
    for (i = 0; i < l; i++)
	fputs("  ", stderr);
}

static void prbox(boxf b)
{
    fprintf(stderr, "((%.5g,%.5g),(%.5g,%.5g))\n", b.LL.x, b.LL.y, b.UR.x,
	    b.UR.y);
}

static void dumpL(field_t * info, int level)
{
    int i;

    indent(level);
    if (info->n_flds == 0) {
	fprintf(stderr, "Label \"%s\" ", info->lp->text);
	prbox(info->b);
    } else {
	fprintf(stderr, "Tbl ");
	prbox(info->b);
	for (i = 0; i < info->n_flds; i++) {
	    dumpL(info->fld[i], level + 1);
	}
    }
}
#endif

/* syntax of labels: foo|bar|baz or foo|(recursive|label)|baz */
static void record_init(node_t * n)
{
    field_t *info;
    pointf sz;
    int flip;
    size_t len;
    unsigned char sides = BOTTOM | RIGHT | TOP | LEFT;

    /* Always use rankdir to determine how records are laid out */
    flip = !GD_realflip(agraphof(n));
    reclblp = ND_label(n)->text;
    len = strlen(reclblp);
    /* For some forgotten reason, an empty label is parsed into a space, so
     * we need at least two bytes in textbuf, as well as accounting for the
     * error path involving "\\N" below.
     */
    len = MAX(MAX(len, 1), strlen("\\N"));
    char *textbuf = gv_calloc(len + 1, sizeof(char)); // temp buffer for storing labels
    if (!(info = parse_reclbl(n, flip, true, textbuf))) {
	agerrorf("bad label format %s\n", ND_label(n)->text);
	reclblp = "\\N";
	info = parse_reclbl(n, flip, true, textbuf);
    }
    free(textbuf);
    size_reclbl(n, info);
    sz.x = INCH2PS(ND_width(n));
    sz.y = INCH2PS(ND_height(n));
    if (mapbool(late_string(n, N_fixed, "false"))) {
	if (sz.x < info->size.x || sz.y < info->size.y) {
/* should check that the record really won't fit, e.g., there may be no text.
			agwarningf("node '%s' size may be too small\n", agnameof(n));
*/
	}
    } else {
	sz.x = fmax(info->size.x, sz.x);
	sz.y = fmax(info->size.y, sz.y);
    }
    resize_reclbl(info, sz, mapbool(late_string(n, N_nojustify, "false")));
    pointf ul = {-sz.x / 2., sz.y / 2.};	/* FIXME - is this still true:    suspected to introduce rounding error - see Kluge below */
    pos_reclbl(info, ul, sides);
    ND_width(n) = PS2INCH(info->size.x);
    ND_height(n) = PS2INCH(info->size.y + 1);	/* Kluge!!  +1 to fix rounding diff between layout and rendering
						   otherwise we can get -1 coords in output */
    ND_shape_info(n) = info;
}

static void record_free(node_t * n)
{
    field_t *p = ND_shape_info(n);

    free_field(p);
}

static field_t *map_rec_port(field_t * f, char *str)
{
    field_t *rv;
    int sub;

    if (f->id && streq(f->id, str))
	rv = f;
    else {
	rv = NULL;
	for (sub = 0; sub < f->n_flds; sub++)
	    if ((rv = map_rec_port(f->fld[sub], str)))
		break;
    }
    return rv;
}

static port record_port(node_t * n, char *portname, char *compass)
{
    field_t *f;
    field_t *subf;
    port rv;
    unsigned char sides; // bitmap of which sides the port lies along

    if (portname[0] == '\0')
	return Center;
    sides = BOTTOM | RIGHT | TOP | LEFT;
    if (compass == NULL)
	compass = "_";
    f = ND_shape_info(n);
    if ((subf = map_rec_port(f, portname))) {
	if (compassPort(n, &subf->b, &rv, compass, subf->sides, NULL)) {
	    agwarningf(
		  "node %s, port %s, unrecognized compass point '%s' - ignored\n",
		  agnameof(n), portname, compass);
	}
    } else if (compassPort(n, &f->b, &rv, portname, sides, NULL)) {
	unrecognized(n, portname);
    }

    return rv;
}

/* record_inside:
 * Note that this does not handle Mrecords correctly. It assumes
 * everything is a rectangle.
 */
static bool record_inside(inside_t * inside_context, pointf p)
{

    field_t *fld0;
    boxf *bp = inside_context->s.bp;
    node_t *n = inside_context->s.n;
    boxf bbox;

    /* convert point to node coordinate system */
    p = ccwrotatepf(p, 90 * GD_rankdir(agraphof(n)));

    if (bp == NULL) {
	fld0 = ND_shape_info(n);
	bbox = fld0->b;
    } else
	bbox = *bp;

    // adjust bbox to outline, i.e., the periphery with penwidth taken into account
    const double penwidth = late_double(n, N_penwidth, DEFAULT_NODEPENWIDTH,
                                        MIN_NODEPENWIDTH);
    const pointf extension = {penwidth / 2, penwidth / 2};
    bbox.LL = sub_pointf(bbox.LL, extension);
    bbox.UR = add_pointf(bbox.UR, extension);

    return INSIDE(p, bbox);
}

/* record_path:
 * Generate box path from port to border.
 * See poly_path for constraints.
 */
static int record_path(node_t * n, port * prt, int side, boxf rv[],
		       int *kptr)
{
    int i;
    double ls, rs;
    pointf p;
    field_t *info;

    if (!prt->defined)
	return 0;
    p = prt->p;
    info = ND_shape_info(n);

    for (i = 0; i < info->n_flds; i++) {
	if (!GD_flip(agraphof(n))) {
	    ls = info->fld[i]->b.LL.x;
	    rs = info->fld[i]->b.UR.x;
	} else {
	    ls = info->fld[i]->b.LL.y;
	    rs = info->fld[i]->b.UR.y;
	}
	if (BETWEEN(ls, p.x, rs)) {
	    /* FIXME: I don't understand this code */
	    if (GD_flip(agraphof(n))) {
		rv[0] = flip_rec_boxf(info->fld[i]->b, ND_coord(n));
	    } else {
		rv[0].LL.x = ND_coord(n).x + ls;
		rv[0].LL.y = ND_coord(n).y - (ND_ht(n) / 2);
		rv[0].UR.x = ND_coord(n).x + rs;
	    }
	    rv[0].UR.y = ND_coord(n).y + (ND_ht(n) / 2);
	    *kptr = 1;
	    break;
	}
    }
    return side;
}

static void gen_fields(GVJ_t * job, node_t * n, field_t * f)
{
    int i;
    pointf AF[2], coord;

    if (f->lp) {
	f->lp->pos = add_pointf(mid_pointf(f->b.LL, f->b.UR), ND_coord(n));
	emit_label(job, EMIT_NLABEL, f->lp);
	penColor(job, n);
    }

    coord = ND_coord(n);
    for (i = 0; i < f->n_flds; i++) {
	if (i > 0) {
	    if (f->LR) {
		AF[0] = f->fld[i]->b.LL;
		AF[1].x = AF[0].x;
		AF[1].y = f->fld[i]->b.UR.y;
	    } else {
		AF[1] = f->fld[i]->b.UR;
		AF[0].x = f->fld[i]->b.LL.x;
		AF[0].y = AF[1].y;
	    }
	    AF[0] = add_pointf(AF[0], coord);
	    AF[1] = add_pointf(AF[1], coord);
	    gvrender_polyline(job, AF, 2);
	}
	gen_fields(job, n, f->fld[i]);
    }
}

static void record_gencode(GVJ_t * job, node_t * n)
{
    obj_state_t *obj = job->obj;
    boxf BF;
    pointf AF[4];
    field_t *f;
    int doMap = obj->url || obj->explicit_tooltip;
    int filled;

    f = ND_shape_info(n);
    BF = f->b;
    BF.LL.x += ND_coord(n).x;
    BF.LL.y += ND_coord(n).y;
    BF.UR.x += ND_coord(n).x;
    BF.UR.y += ND_coord(n).y;

    if (doMap && !(job->flags & EMIT_CLUSTERS_LAST))
	gvrender_begin_anchor(job,
			      obj->url, obj->tooltip, obj->target,
			      obj->id);
    graphviz_polygon_style_t style = stylenode(job, n);
    penColor(job, n);
    char *clrs[2] = {0};
    if (style.filled) {
	char* fillcolor = findFill (n);
	double frac;

	if (findStopColor (fillcolor, clrs, &frac)) {
            gvrender_set_fillcolor(job, clrs[0]);
	    if (clrs[1])
		gvrender_set_gradient_vals(job,clrs[1],late_int(n,N_gradientangle,0,0), frac);
	    else
		gvrender_set_gradient_vals(job,DEFAULT_COLOR,late_int(n,N_gradientangle,0,0), frac);
	    if (style.radial)
		filled = RGRADIENT;
	    else
		filled = GRADIENT;
	}
	else {
	    filled = FILL;
            gvrender_set_fillcolor(job, fillcolor);
	}
    }
    else filled = 0;

    if (streq(ND_shape(n)->name, "Mrecord"))
	style.rounded = true;
    if (SPECIAL_CORNERS(style)) {
	AF[0] = BF.LL;
	AF[2] = BF.UR;
	AF[1].x = AF[2].x;
	AF[1].y = AF[0].y;
	AF[3].x = AF[0].x;
	AF[3].y = AF[2].y;
	round_corners(job, AF, 4, style, filled);
    } else {
	gvrender_box(job, BF, filled);
    }

    gen_fields(job, n, f);

    free (clrs[0]);
    free(clrs[1]);

    if (doMap) {
	if (job->flags & EMIT_CLUSTERS_LAST)
	    gvrender_begin_anchor(job,
				  obj->url, obj->tooltip, obj->target,
				  obj->id);
	gvrender_end_anchor(job);
    }
}

static shape_desc **UserShape;
static size_t N_UserShape;

shape_desc *find_user_shape(const char *name)
{
    if (UserShape) {
	for (size_t i = 0; i < N_UserShape; i++) {
	    if (streq(UserShape[i]->name, name))
		return UserShape[i];
	}
    }
    return NULL;
}

static shape_desc *user_shape(char *name)
{
    shape_desc *p;

    if ((p = find_user_shape(name)))
	return p;
    size_t i = N_UserShape++;
    UserShape = gv_recalloc(UserShape, N_UserShape - 1, N_UserShape, sizeof(shape_desc *));
    p = UserShape[i] = gv_alloc(sizeof(shape_desc));
    *p = Shapes[0];
    p->name = strdup(name);
    if (Lib == NULL && !streq(name, "custom")) {
	agwarningf("using %s for unknown shape %s\n", Shapes[0].name,
	      p->name);
	p->usershape = false;
    } else {
	p->usershape = true;
    }
    return p;
}

shape_desc *bind_shape(char *name, node_t * np)
{
    shape_desc *ptr, *rv = NULL;
    const char *str;

    str = safefile(agget(np, "shapefile"));
    /* If shapefile is defined and not epsf, set shape = custom */
    if (str && !streq(name, "epsf"))
	name = "custom";
    if (!streq(name, "custom")) {
	for (ptr = Shapes; ptr->name; ptr++) {
	    if (streq(ptr->name, name)) {
		rv = ptr;
		break;
	    }
	}
    }
    if (rv == NULL)
	rv = user_shape(name);
    return rv;
}

static bool epsf_inside(inside_t * inside_context, pointf p)
{
    pointf P;
    double x2;
    node_t *n = inside_context->s.n;

    P = ccwrotatepf(p, 90 * GD_rankdir(agraphof(n)));
    x2 = ND_ht(n) / 2;
    return P.y >= -x2 && P.y <= x2 && P.x >= -ND_lw(n) && P.x <= ND_rw(n);
}

static void epsf_gencode(GVJ_t * job, node_t * n)
{
    obj_state_t *obj = job->obj;
    epsf_t *desc;
    int doMap = obj->url || obj->explicit_tooltip;

    desc = ND_shape_info(n);
    if (!desc)
	return;

    if (doMap && !(job->flags & EMIT_CLUSTERS_LAST))
	gvrender_begin_anchor(job,
			      obj->url, obj->tooltip, obj->target,
			      obj->id);
    if (desc)
	fprintf(job->output_file,
		"%.5g %.5g translate newpath user_shape_%d\n",
		ND_coord(n).x + desc->offset.x,
		ND_coord(n).y + desc->offset.y, desc->macro_id);
    ND_label(n)->pos = ND_coord(n);

    emit_label(job, EMIT_NLABEL, ND_label(n));
    if (doMap) {
	if (job->flags & EMIT_CLUSTERS_LAST)
	    gvrender_begin_anchor(job,
				  obj->url, obj->tooltip, obj->target,
				  obj->id);
	gvrender_end_anchor(job);
    }
}

#define alpha   (M_PI/10.0)
#define alpha2  (2*alpha)
#define alpha3  (3*alpha)
#define alpha4  (2*alpha2)

static pointf star_size (pointf sz0)
{
    pointf sz;
    double r, rx, ry;

    rx = sz0.x/(2*cos(alpha));
    ry = sz0.y/(sin(alpha) + sin(alpha3));
    const double r0 = fmax(rx, ry);
    r = r0 * sin(alpha4) * cos(alpha2) / (cos(alpha) * cos(alpha4));

    sz.x = 2*r*cos(alpha);
    sz.y = r*(1 + sin(alpha3));
    return sz;
}

static void star_vertices (pointf* vertices, pointf* bb)
{
    int i;
    pointf sz = *bb;
    double offset, a, aspect = (1 + sin(alpha3))/(2*cos(alpha));
    double r, r0, theta = alpha;

    /* Scale up width or height to required aspect ratio */
    a = sz.y/sz.x;
    if (a > aspect) {
	sz.x = sz.y/aspect;
    }
    else if (a < aspect) {
	sz.y = sz.x*aspect;
    }

    /* for given sz, get radius */
    r = sz.x/(2*cos(alpha));
    r0 = r * cos(alpha) * cos(alpha4) / (sin(alpha4) * cos(alpha2));

    /* offset is the y shift of circle center from bb center */
    offset = (r*(1 - sin(alpha3)))/2;

    for (i = 0; i < 10; i += 2) {
	vertices[i].x = r*cos(theta);
	vertices[i].y = r*sin(theta) - offset;
	theta += alpha2;
	vertices[i+1].x = r0*cos(theta);
	vertices[i+1].y = r0*sin(theta) - offset;
	theta += alpha2;
    }

    *bb = sz;
}

static bool star_inside(inside_t * inside_context, pointf p)
{
    size_t sides;
    pointf *vertex;
    const pointf O = {0};

    if (!inside_context) {
	return false;
    }
    boxf *bp = inside_context->s.bp;
    node_t *n = inside_context->s.n;
    pointf P, Q, R;
    int outcnt;

    P = ccwrotatepf(p, 90 * GD_rankdir(agraphof(n)));

    /* Quick test if port rectangle is target */
    if (bp) {
	boxf bbox = *bp;
	return INSIDE(P, bbox);
    }

    if (n != inside_context->s.lastn) {
	inside_context->s.last_poly = ND_shape_info(n);
	vertex = inside_context->s.last_poly->vertices;
	sides = inside_context->s.last_poly->sides;

	const double penwidth = late_double(n, N_penwidth, DEFAULT_NODEPENWIDTH,
	                                    MIN_NODEPENWIDTH);
	if (inside_context->s.last_poly->peripheries >= 1 && penwidth > 0) {
	    /* index to outline, i.e., the outer-periphery with penwidth taken into account */
	    inside_context->s.outp = (inside_context->s.last_poly->peripheries + 1 - 1)
	                           * sides;
	} else if (inside_context->s.last_poly->peripheries < 1) {
	    inside_context->s.outp = 0;
	} else {
	    /* index to outer-periphery */
	    inside_context->s.outp = (inside_context->s.last_poly->peripheries - 1)
	                           * sides;
	}
	inside_context->s.lastn = n;
    } else {
	vertex = inside_context->s.last_poly->vertices;
	sides = inside_context->s.last_poly->sides;
    }

    outcnt = 0;
    for (size_t i = 0; i < sides; i += 2) {
	Q = vertex[i + inside_context->s.outp];
	R = vertex[(i + 4) % sides + inside_context->s.outp];
	if (!(same_side(P, O, Q, R))) {
	    outcnt++;
	}
	if (outcnt == 2) {
	    return false;
	}
    }
    return true;
}

/* cylinder:
 * Code based on PostScript version by Brandon Rhodes.
 * http://rhodesmill.org/brandon/2007/a-database-symbol-for-graphviz/
 */
static pointf cylinder_size (pointf sz)
{
    sz.y *= 1.375;
    return sz;
}

static void cylinder_vertices (pointf* vertices, pointf* bb)
{
    double x = bb->x/2;
    double y = bb->y/2;
    double yr = bb->y/11;

    vertices[0].x = x;
    vertices[0].y = y-yr;
    vertices[1].x = x;
    vertices[1].y = y-(1-0.551784)*yr;
    vertices[2].x = 0.551784*x;
    vertices[2].y = y;
    vertices[3].x = 0;
    vertices[3].y = y;
    vertices[4].x = -0.551784*x;
    vertices[4].y = y;
    vertices[5].x = -x;
    vertices[5].y = vertices[1].y;
    vertices[6].x = -x;
    vertices[6].y = y-yr;
    vertices[7] = vertices[6];
    vertices[8].x = -x;
    vertices[8].y = yr-y;
    vertices[9] = vertices[8];
    vertices[10].x = -x;
    vertices[10].y = -vertices[1].y;
    vertices[11].x = vertices[4].x;
    vertices[11].y = -vertices[4].y;
    vertices[12].x = vertices[3].x;
    vertices[12].y = -vertices[3].y;
    vertices[13].x = vertices[2].x;
    vertices[13].y = -vertices[2].y;
    vertices[14].x = vertices[1].x;
    vertices[14].y = -vertices[1].y;
    vertices[15].x = vertices[0].x;
    vertices[15].y = -vertices[0].y;
    vertices[16] = vertices[15];
    vertices[18] = vertices[17] = vertices[0];
}

static void cylinder_draw(GVJ_t *job, pointf *AF, size_t sides, int filled) {
    pointf vertices[7];
    double y0 = AF[0].y;
    double y02 = y0+y0;

    vertices[0] = AF[0];
    vertices[1].x = AF[1].x;
    vertices[1].y = y02 - AF[1].y;
    vertices[2].x = AF[2].x;
    vertices[2].y = y02 - AF[2].y;
    vertices[3].x = AF[3].x;
    vertices[3].y = y02 - AF[3].y;
    vertices[4].x = AF[4].x;
    vertices[4].y = y02 - AF[4].y;
    vertices[5].x = AF[5].x;
    vertices[5].y = y02 - AF[5].y;
    vertices[6] = AF[6];

    gvrender_beziercurve(job, AF, sides, filled);
    gvrender_beziercurve(job, vertices, 7, 0);
}

static const char *side_port[] = {"s", "e", "n", "w"};

static pointf cvtPt(pointf p, int rankdir) {
    pointf q = { 0, 0 };

    switch (rankdir) {
    case RANKDIR_TB:
	q = p;
	break;
    case RANKDIR_BT:
	q.x = p.x;
	q.y = -p.y;
	break;
    case RANKDIR_LR:
	q.y = p.x;
	q.x = -p.y;
	break;
    case RANKDIR_RL:
	q.y = p.x;
	q.x = p.y;
	break;
    default:
	UNREACHABLE();
    }
    return q;
}

/* closestSide:
 * Resolve unspecified compass-point port to best available port.
 * At present, this finds the available side closest to the center
 * of the other port.
 *
 * This could be improved:
 *  - if other is unspecified, do them together
 *  - if dot, bias towards bottom of one to top of another, if possible
 *  - if line segment from port centers uses available sides, use these
 *     or center. (This latter may require spline routing to cooperate.)
 */
static const char *closestSide(node_t *n, node_t *other, port *oldport) {
    boxf b;
    int rkd = GD_rankdir(agraphof(n)->root);
    pointf p = {0};
    const pointf pt = cvtPt(ND_coord(n), rkd);
    const pointf opt = cvtPt(ND_coord(other), rkd);
    int sides = oldport->side;
    const char *rv = NULL;

    if (sides == 0 || sides == (TOP | BOTTOM | LEFT | RIGHT))
	return rv;		/* use center */

    if (oldport->bp) {
	b = *oldport->bp;
    } else {
	if (GD_flip(agraphof(n))) {
	    b.UR.x = ND_ht(n) / 2;
	    b.LL.x = -b.UR.x;
	    b.UR.y = ND_lw(n);
	    b.LL.y = -b.UR.y;
	} else {
	    b.UR.y = ND_ht(n) / 2;
	    b.LL.y = -b.UR.y;
	    b.UR.x = ND_lw(n);
	    b.LL.x = -b.UR.x;
	}
    }

    double mind = 0;
    for (int i = 0; i < 4; i++) {
	if ((sides & (1 << i)) == 0)
	    continue;
	switch (i) {
	case BOTTOM_IX:
	    p.y = b.LL.y;
	    p.x = (b.LL.x + b.UR.x) / 2;
	    break;
	case RIGHT_IX:
	    p.x = b.UR.x;
	    p.y = (b.LL.y + b.UR.y) / 2;
	    break;
	case TOP_IX:
	    p.y = b.UR.y;
	    p.x = (b.LL.x + b.UR.x) / 2;
	    break;
	case LEFT_IX:
	    p.x = b.LL.x;
	    p.y = (b.LL.y + b.UR.y) / 2;
	    break;
	default:
	    UNREACHABLE();
	}
	p.x += pt.x;
	p.y += pt.y;
	const double d = DIST2(p, opt);
	if (!rv || d < mind) {
	    mind = d;
	    rv = side_port[i];
	}
    }
    return rv;
}

port resolvePort(node_t * n, node_t * other, port * oldport)
{
    port rv;
    const char *compass = closestSide(n, other, oldport);

    /* transfer name pointer; all other necessary fields will be regenerated */
    rv.name = oldport->name;
    compassPort(n, oldport->bp, &rv, compass, oldport->side, NULL);

    return rv;
}

void resolvePorts(edge_t * e)
{
    if (ED_tail_port(e).dyna)
	ED_tail_port(e) =
	    resolvePort(agtail(e), aghead(e), &ED_tail_port(e));
    if (ED_head_port(e).dyna)
	ED_head_port(e) =
	    resolvePort(aghead(e), agtail(e), &ED_head_port(e));
}
