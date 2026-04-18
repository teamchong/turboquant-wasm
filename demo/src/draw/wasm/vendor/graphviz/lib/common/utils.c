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

#include "config.h"

#include <common/render.h>
#include <common/geomprocs.h>
#include <common/htmltable.h>
#include <common/entities.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <gvc/gvc.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/gv_ctype.h>
#include <util/gv_math.h>
#include <util/list.h>
#include <util/path.h>
#include <util/startswith.h>
#include <util/strcasecmp.h>
#include <util/streq.h>
#include <util/strview.h>
#include <util/tokenize.h>

int late_int(void *obj, attrsym_t *attr, int defaultValue, int minimum) {
    if (attr == NULL)
        return defaultValue;
    char *p = agxget(obj, attr);
    if (!p || p[0] == '\0')
        return defaultValue;
    char *endp;
    long rv = strtol(p, &endp, 10);
    if (p == endp || rv > INT_MAX)
        return defaultValue; /* invalid int format */
    if (rv < minimum)
        return minimum;
    return (int)rv;
}

double late_double(void *obj, attrsym_t *attr, double defaultValue,
                   double minimum) {
    if (!attr || !obj)
        return defaultValue;
    char *p = agxget(obj, attr);
    if (!p || p[0] == '\0')
        return defaultValue;
    char *endp;
    double rv = strtod(p, &endp);
    if (p == endp)
        return defaultValue; /* invalid double format */
    if (rv < minimum)
        return minimum;
    return rv;
}

/** Return value for PSinputscale. If this is > 0, it has been set on the
 * command line and this value is used.
 * Otherwise, we check the graph's inputscale attribute. If this is not set
 * or has a bad value, we return -1.
 * If the value is 0, we return the default. Otherwise, we return the value.
 * Set but negative values are treated like 0.
 */
double get_inputscale(graph_t *g) {
    if (PSinputscale > 0) return PSinputscale;  /* command line flag prevails */
    double d = late_double(g, agfindgraphattr(g, "inputscale"), -1, 0);
    if (is_exactly_zero(d)) return POINTS_PER_INCH;
    return d;
}

char *late_string(void *obj, attrsym_t *attr, char *defaultValue) {
    if (!attr || !obj)
        return defaultValue;
    return agxget(obj, attr);
}

char *late_nnstring(void *obj, attrsym_t *attr, char *defaultValue) {
    char *rv = late_string(obj, attr, defaultValue);
    if (!rv || (rv[0] == '\0'))
        return defaultValue;
    return rv;
}

bool late_bool(void *obj, attrsym_t *attr, bool defaultValue) {
    if (attr == NULL)
        return defaultValue;

    return mapbool(agxget(obj, attr));
}

node_t *UF_find(node_t * n)
{
    while (ND_UF_parent(n) && ND_UF_parent(n) != n) {
	if (ND_UF_parent(ND_UF_parent(n)))
	    ND_UF_parent(n) = ND_UF_parent(ND_UF_parent(n));
	n = ND_UF_parent(n);
    }
    return n;
}

node_t *UF_union(node_t * u, node_t * v)
{
    if (u == v)
	return u;
    if (ND_UF_parent(u) == NULL) {
	ND_UF_parent(u) = u;
	ND_UF_size(u) = 1;
    } else
	u = UF_find(u);
    if (ND_UF_parent(v) == NULL) {
	ND_UF_parent(v) = v;
	ND_UF_size(v) = 1;
    } else
	v = UF_find(v);
	/* if we have two copies of the same node, their union is just that node */
    if (u == v)
	return u;
    if (ND_id(u) > ND_id(v)) {
	ND_UF_parent(u) = v;
	ND_UF_size(v) += ND_UF_size(u);
    } else {
	ND_UF_parent(v) = u;
	ND_UF_size(u) += ND_UF_size(v);
	v = u;
    }
    return v;
}

void UF_singleton(node_t * u)
{
    ND_UF_size(u) = 1;
    ND_UF_parent(u) = NULL;
    ND_ranktype(u) = NORMAL;
}

void UF_setname(node_t * u, node_t * v)
{
    assert(u == UF_find(u));
    ND_UF_parent(u) = v;
    ND_UF_size(v) += ND_UF_size(u);
}

pointf coord(node_t * n)
{
    pointf r;

    r.x = POINTS_PER_INCH * ND_pos(n)[0];
    r.y = POINTS_PER_INCH * ND_pos(n)[1];
    return r;
}

/* from Glassner's Graphics Gems */
#define W_DEGREE 5

/*
 *	Evaluate a Bézier curve at a particular parameter value
 *      Fill in control points for resulting sub-curves if "Left" and
 *	"Right" are non-null.
 *
 */
pointf Bezier(const pointf *V, double t, pointf *Left, pointf *Right) {
    const int degree = 3;
    int i, j;			/* Index variables      */
    pointf Vtemp[W_DEGREE + 1][W_DEGREE + 1];

    /* Copy control points  */
    for (j = 0; j <= degree; j++) {
	Vtemp[0][j] = V[j];
    }

    /* Triangle computation */
    for (i = 1; i <= degree; i++) {
	for (j = 0; j <= degree - i; j++) {
	    Vtemp[i][j].x =
		(1.0 - t) * Vtemp[i - 1][j].x + t * Vtemp[i - 1][j + 1].x;
	    Vtemp[i][j].y =
		(1.0 - t) * Vtemp[i - 1][j].y + t * Vtemp[i - 1][j + 1].y;
	}
    }

    if (Left != NULL)
	for (j = 0; j <= degree; j++)
	    Left[j] = Vtemp[j][0];
    if (Right != NULL)
	for (j = 0; j <= degree; j++)
	    Right[j] = Vtemp[degree - j][j];

    return Vtemp[degree][0];
}

#ifdef DEBUG
edge_t *debug_getedge(graph_t * g, char *s0, char *s1)
{
    node_t *n0, *n1;
    n0 = agfindnode(g, s0);
    n1 = agfindnode(g, s1);
    if (n0 && n1)
	return agfindedge(g, n0, n1);
    return NULL;
}
Agraphinfo_t* GD_info(graph_t * g) { return ((Agraphinfo_t*)AGDATA(g));}
Agnodeinfo_t* ND_info(node_t * n) { return ((Agnodeinfo_t*)AGDATA(n));}
#endif

/* safefile:
 * Check to make sure it is okay to read in files.
 * It returns NULL if the filename is trivial.
 *
 * If the application has set the SERVER_NAME environment variable,
 * this indicates it is web-active.
 *
 * If filename contains multiple components, the user is
 * warned, once, that everything to the left is ignored.
 *
 * For non-server applications, we use the path list in Gvimagepath to
 * resolve relative pathnames.
 *
 * N.B. safefile uses a fixed buffer, so functions using it should use the
 * value immediately or make a copy.
 */
#ifdef _WIN32
#define PATHSEP ";"
#else
#define PATHSEP ":"
#endif

typedef LIST(strview_t) strviews_t;

static strviews_t mkDirlist(const char *list) {
    strviews_t dirs = {0};

    for (tok_t t = tok(list, PATHSEP); !tok_end(&t); tok_next(&t)) {
        strview_t dir = tok_get(&t);
        LIST_APPEND(&dirs, dir);
    }
    return dirs;
}

static char *findPath(const strviews_t dirs, const char *str) {
    static agxbuf safefilename;

    for (size_t i = 0; i < LIST_SIZE(&dirs); ++i) {
	const strview_t d = LIST_GET(&dirs, i);
	agxbprint(&safefilename, "%.*s%c%s", (int)d.size, d.data, PATH_SEPARATOR, str);
	char *filename = agxbuse(&safefilename);
	if (access(filename, R_OK) == 0)
	    return filename;
    }
    return NULL;
}

const char *safefile(const char *filename)
{
    static bool onetime = true;
    static char *pathlist = NULL;
    static strviews_t dirs;

    if (!filename || !filename[0])
	return NULL;

    if (HTTPServerEnVar) {   /* If used as a server */
	if (onetime) {
	    agwarningf(
		      "file loading is disabled because the environment contains SERVER_NAME=\"%s\"\n",
		      HTTPServerEnVar);
	    onetime = false;
	}
	return NULL;
    }

    if (Gvfilepath != NULL) {
	if (pathlist == NULL) {
	    LIST_FREE(&dirs);
	    pathlist = Gvfilepath;
	    dirs = mkDirlist(pathlist);
	}

	const char *str = filename;
	for (const char *sep = "/\\:"; *sep != '\0'; ++sep) {
	    const char *p = strrchr(str, *sep);
	    if (p != NULL) {
	        str = ++p;
	    }
	}

	return findPath(dirs, str);
    }

    if (pathlist != Gvimagepath) {
	LIST_FREE(&dirs);
	pathlist = Gvimagepath;
	if (pathlist && *pathlist)
	    dirs = mkDirlist(pathlist);
    }

    if (*filename == PATH_SEPARATOR || LIST_IS_EMPTY(&dirs))
	return filename;

    return findPath(dirs, filename);
}

int maptoken(char *p, char **name, int *val) {
    char *q;

    int i = 0;
    for (; (q = name[i]) != 0; i++)
        if (p && streq(p, q))
            break;
    return val[i];
}

bool mapBool(const char *p, bool defaultValue) {
    if (!p || *p == '\0')
        return defaultValue;
    if (!strcasecmp(p, "false"))
	return false;
    if (!strcasecmp(p, "no"))
	return false;
    if (!strcasecmp(p, "true"))
	return true;
    if (!strcasecmp(p, "yes"))
	return true;
    if (gv_isdigit(*p))
	return atoi(p) != 0;
    return defaultValue;
}

bool mapbool(const char *p)
{
    return mapBool(p, false);
}

pointf dotneato_closest(splines * spl, pointf pt)
{
    double d2, dlow2, dhigh2; /* squares of distances */
    double low, high, t;
    pointf c[4], pt2;
    bezier bz;

    size_t besti = SIZE_MAX;
    size_t bestj = SIZE_MAX;
    double bestdist2 = DBL_MAX;
    for (size_t i = 0; i < spl->size; i++) {
	bz = spl->list[i];
	for (size_t j = 0; j < bz.size; j++) {
	    pointf b;

	    b.x = bz.list[j].x;
	    b.y = bz.list[j].y;
	    d2 = DIST2(b, pt);
	    if (bestj == SIZE_MAX || d2 < bestdist2) {
		besti = i;
		bestj = j;
		bestdist2 = d2;
	    }
	}
    }

    bz = spl->list[besti];
    /* Pick best Bézier. If bestj is the last point in the B-spline, decrement.
     * Then set j to be the first point in the corresponding Bézier by dividing
     * then multiplying be 3. Thus, 0,1,2 => 0; 3,4,5 => 3, etc.
     */
    if (bestj == bz.size-1)
	bestj--;
    const size_t j = 3 * (bestj / 3);
    for (size_t k = 0; k < 4; k++) {
	c[k].x = bz.list[j + k].x;
	c[k].y = bz.list[j + k].y;
    }
    low = 0.0;
    high = 1.0;
    dlow2 = DIST2(c[0], pt);
    dhigh2 = DIST2(c[3], pt);
    do {
	t = (low + high) / 2.0;
	pt2 = Bezier(c, t, NULL, NULL);
	if (fabs(dlow2 - dhigh2) < 1.0)
	    break;
	if (fabs(high - low) < .00001)
	    break;
	if (dlow2 < dhigh2) {
	    high = t;
	    dhigh2 = DIST2(pt2, pt);
	} else {
	    low = t;
	    dlow2 = DIST2(pt2, pt);
	}
    } while (1);
    return pt2;
}

static int Tflag;
void gvToggle(int s)
{
    (void)s;
    Tflag = !Tflag;
#if !defined(_WIN32) && !defined(__wasm__)
    signal(SIGUSR1, gvToggle);
#endif
}

int test_toggle(void)
{
    return Tflag;
}

struct fontinfo {
    double fontsize;
    char *fontname;
    char *fontcolor;
};

void common_init_node(node_t * n)
{
    struct fontinfo fi;
    char *str;
    ND_width(n) =
	late_double(n, N_width, DEFAULT_NODEWIDTH, MIN_NODEWIDTH);
    ND_height(n) =
	late_double(n, N_height, DEFAULT_NODEHEIGHT, MIN_NODEHEIGHT);
    ND_shape(n) =
	bind_shape(late_nnstring(n, N_shape, DEFAULT_NODESHAPE), n);
    str = agxget(n, N_label);
    fi.fontsize = late_double(n, N_fontsize, DEFAULT_FONTSIZE, MIN_FONTSIZE);
    fi.fontname = late_nnstring(n, N_fontname, DEFAULT_FONTNAME);
    fi.fontcolor = late_nnstring(n, N_fontcolor, DEFAULT_COLOR);
    ND_label(n) = make_label(n, str, aghtmlstr(str), shapeOf(n) == SH_RECORD,
		fi.fontsize, fi.fontname, fi.fontcolor);
    if (N_xlabel && (str = agxget(n, N_xlabel)) && str[0]) {
	ND_xlabel(n) = make_label(n, str, aghtmlstr(str), false,
				fi.fontsize, fi.fontname, fi.fontcolor);
	GD_has_labels(agraphof(n)) |= NODE_XLABEL;
    }

    {
	const int showboxes = imin(late_int(n, N_showboxes, 0, 0), UCHAR_MAX);
	ND_showboxes(n) = (unsigned char)showboxes;
    }
    ND_shape(n)->fns->initfn(n);
}

static void initFontEdgeAttr(edge_t * e, struct fontinfo *fi)
{
    fi->fontsize = late_double(e, E_fontsize, DEFAULT_FONTSIZE, MIN_FONTSIZE);
    fi->fontname = late_nnstring(e, E_fontname, DEFAULT_FONTNAME);
    fi->fontcolor = late_nnstring(e, E_fontcolor, DEFAULT_COLOR);
}

static void
initFontLabelEdgeAttr(edge_t * e, struct fontinfo *fi,
		      struct fontinfo *lfi)
{
    if (!fi->fontname) initFontEdgeAttr(e, fi);
    lfi->fontsize = late_double(e, E_labelfontsize, fi->fontsize, MIN_FONTSIZE);
    lfi->fontname = late_nnstring(e, E_labelfontname, fi->fontname);
    lfi->fontcolor = late_nnstring(e, E_labelfontcolor, fi->fontcolor);
}

/// Return true if head/tail end of edge should not be clipped to node.
static bool
noClip(edge_t *e, attrsym_t* sym)
{
    char		*str;
    bool rv = false;

    if (sym) {	/* mapbool isn't a good fit, because we want "" to mean true */
	str = agxget(e,sym);
	if (str && str[0]) rv = !mapbool(str);
	else rv = false;
    }
    return rv;
}

static port
chkPort (port (*pf)(node_t*, char*, char*), node_t* n, char* s)
{
    port pt;
	char* cp=NULL;
	if(s)
		cp= strchr(s,':');
    if (cp) {
	*cp = '\0';
	pt = pf(n, s, cp+1);
	*cp = ':';
	pt.name = cp+1;
    }
    else {
	pt = pf(n, s, NULL);
	pt.name = s;
    }
    return pt;
}

/* return true if edge has label */
void common_init_edge(edge_t *e) {
    char *str;
    struct fontinfo fi;
    struct fontinfo lfi;
    graph_t *sg = agraphof(agtail(e));

    fi.fontname = NULL;
    lfi.fontname = NULL;
    if (E_label && (str = agxget(e, E_label)) && str[0]) {
	initFontEdgeAttr(e, &fi);
	ED_label(e) = make_label(e, str, aghtmlstr(str), false,
				fi.fontsize, fi.fontname, fi.fontcolor);
	GD_has_labels(sg) |= EDGE_LABEL;
	ED_label_ontop(e) = mapbool(late_string(e, E_label_float, "false"));
    }

    if (E_xlabel && (str = agxget(e, E_xlabel)) && str[0]) {
	if (!fi.fontname)
	    initFontEdgeAttr(e, &fi);
	ED_xlabel(e) = make_label(e, str, aghtmlstr(str), false,
				fi.fontsize, fi.fontname, fi.fontcolor);
	GD_has_labels(sg) |= EDGE_XLABEL;
    }

    if (E_headlabel && (str = agxget(e, E_headlabel)) && str[0]) {
	initFontLabelEdgeAttr(e, &fi, &lfi);
	ED_head_label(e) = make_label(e, str, aghtmlstr(str), false,
				lfi.fontsize, lfi.fontname, lfi.fontcolor);
	GD_has_labels(sg) |= HEAD_LABEL;
    }
    if (E_taillabel && (str = agxget(e, E_taillabel)) && str[0]) {
	if (!lfi.fontname)
	    initFontLabelEdgeAttr(e, &fi, &lfi);
	ED_tail_label(e) = make_label(e, str, aghtmlstr(str), false,
				lfi.fontsize, lfi.fontname, lfi.fontcolor);
	GD_has_labels(sg) |= TAIL_LABEL;
    }

    /* We still accept ports beginning with colons but this is deprecated
     * That is, we allow tailport = ":abc" as well as the preferred
     * tailport = "abc".
     */
    str = agget(e, TAIL_ID);
    /* libgraph always defines tailport/headport; libcgraph doesn't */
    if (!str) str = "";
    if (str && str[0])
	ND_has_port(agtail(e)) = true;
    ED_tail_port(e) = chkPort (ND_shape(agtail(e))->fns->portfn, agtail(e), str);
    if (noClip(e, E_tailclip))
	ED_tail_port(e).clip = false;
    str = agget(e, HEAD_ID);
    /* libgraph always defines tailport/headport; libcgraph doesn't */
    if (!str) str = "";
    if (str && str[0])
	ND_has_port(aghead(e)) = true;
    ED_head_port(e) = chkPort(ND_shape(aghead(e))->fns->portfn, aghead(e), str);
    if (noClip(e, E_headclip))
	ED_head_port(e).clip = false;
}

static boxf addLabelBB(boxf bb, textlabel_t * lp, bool flipxy)
{
    double width, height;
    pointf p = lp->pos;
    double min, max;

    if (flipxy) {
	height = lp->dimen.x;
	width = lp->dimen.y;
    }
    else {
	width = lp->dimen.x;
	height = lp->dimen.y;
    }
    min = p.x - width / 2.;
    max = p.x + width / 2.;
    if (min < bb.LL.x)
	bb.LL.x = min;
    if (max > bb.UR.x)
	bb.UR.x = max;

    min = p.y - height / 2.;
    max = p.y + height / 2.;
    if (min < bb.LL.y)
	bb.LL.y = min;
    if (max > bb.UR.y)
	bb.UR.y = max;

    return bb;
}

/** Compute the bounding box of a polygon.
 * We only need to use the outer periphery.
 */
boxf
polyBB (polygon_t* poly)
{
    const size_t sides = poly->sides;
    const size_t peris = MAX(poly->peripheries, (size_t)1);
    pointf* verts = poly->vertices + (peris-1)*sides;
    boxf bb;

    bb.LL = bb.UR = verts[0];
    for (size_t i = 1; i < sides; i++) {
	bb.LL.x = MIN(bb.LL.x,verts[i].x);
	bb.LL.y = MIN(bb.LL.y,verts[i].y);
	bb.UR.x = MAX(bb.UR.x,verts[i].x);
	bb.UR.y = MAX(bb.UR.y,verts[i].y);
    }
    return bb;
}

/** Reset graph's bounding box to include bounding box of the given label.
 * Assume the label's position has been set.
 */
void updateBB(graph_t * g, textlabel_t * lp)
{
    GD_bb(g) = addLabelBB(GD_bb(g), lp, GD_flip(g));
}

/** Compute bounding box of g using nodes, splines, and clusters.
 * Assumes bb of clusters already computed.
 * store in GD_bb.
 */
void compute_bb(graph_t * g)
{
    node_t *n;
    edge_t *e;
    boxf b, bb;
    boxf BF;
    pointf ptf, s2;

    if (agnnodes(g) == 0 && GD_n_cluster(g) == 0) {
	bb.LL = (pointf){0};
	bb.UR = (pointf){0};
	return;
    }

    bb.LL = (pointf){INT_MAX, INT_MAX};
    bb.UR = (pointf){-INT_MAX, -INT_MAX};
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	ptf = coord(n);
	s2.x = ND_xsize(n) / 2.0;
	s2.y = ND_ysize(n) / 2.0;
	b.LL = sub_pointf(ptf, s2);
	b.UR = add_pointf(ptf, s2);

	EXPANDBB(&bb, b);
	if (ND_xlabel(n) && ND_xlabel(n)->set) {
	    bb = addLabelBB(bb, ND_xlabel(n), GD_flip(g));
	}
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    if (ED_spl(e) == 0)
		continue;
	    for (size_t i = 0; i < ED_spl(e)->size; i++) {
		for (size_t j = 0; j < (((Agedgeinfo_t*)AGDATA(e))->spl)->list[i].size; j++) {
		    ptf = ED_spl(e)->list[i].list[j];
		    expandbp(&bb, ptf);
		}
	    }
	    if (ED_label(e) && ED_label(e)->set) {
		bb = addLabelBB(bb, ED_label(e), GD_flip(g));
	    }
	    if (ED_head_label(e) && ED_head_label(e)->set) {
		bb = addLabelBB(bb, ED_head_label(e), GD_flip(g));
	    }
	    if (ED_tail_label(e) && ED_tail_label(e)->set) {
		bb = addLabelBB(bb, ED_tail_label(e), GD_flip(g));
	    }
	    if (ED_xlabel(e) && ED_xlabel(e)->set) {
		bb = addLabelBB(bb, ED_xlabel(e), GD_flip(g));
	    }
	}
    }

    for (int i = 1; i <= GD_n_cluster(g); i++) {
	B2BF(GD_bb(GD_clust(g)[i]), BF);
	EXPANDBB(&bb, BF);
    }
    if (GD_label(g) && GD_label(g)->set) {
	bb = addLabelBB(bb, GD_label(g), GD_flip(g));
    }

    GD_bb(g) = bb;
}

bool is_a_cluster (Agraph_t* g)
{
  return g == g->root || !strncasecmp(agnameof(g), "cluster", 7) ||
         mapbool(agget(g, "cluster"));
}

/** Sets object's name attribute to the given value.
 * Creates the attribute if not already set.
 */
Agsym_t *setAttr(graph_t * g, void *obj, char *name, char *value,
			Agsym_t * ap)
{
    if (ap == NULL) {
	switch (agobjkind(obj)) {
	case AGRAPH:
	    ap = agattr_text(g, AGRAPH,name, "");
	    break;
	case AGNODE:
	    ap = agattr_text(g,AGNODE, name, "");
	    break;
	case AGEDGE:
	    ap = agattr_text(g,AGEDGE, name, "");
	    break;
	}
    }
    agxset(obj, ap, value);
    return ap;
}

/** Generate a special cluster node representing the end node
 * of an edge to the cluster cg. n is a node whose name is the same
 * as the cluster cg. clg is the subgraph of all of
 * the original nodes, which will be deleted later.
 */
static node_t *clustNode(node_t * n, graph_t * cg, agxbuf * xb,
			 graph_t * clg)
{
    node_t *cn;
    static int idx = 0;

    agxbprint(xb, "__%d:%s", idx++, agnameof(cg));

    cn = agnode(agroot(cg), agxbuse(xb), 1);
    agbindrec(cn, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);

    SET_CLUST_NODE(cn);
	agsubnode(cg,cn,1);
	agsubnode(clg,n,1);

    /* set attributes */
    N_label = setAttr(agraphof(cn), cn, "label", "", N_label);
    N_style = setAttr(agraphof(cn), cn, "style", "invis", N_style);
    N_shape = setAttr(agraphof(cn), cn, "shape", "box", N_shape);

    return cn;
}

typedef struct {
    Dtlink_t link;		/* cdt data */
    void *p[2];			/* key */
    node_t *t;
    node_t *h;
} item;

static int cmpItem(void *pp1, void *pp2) {
    const void **p1 = pp1;
    const void **p2 = pp2;
    if ((uintptr_t)p1[0] < (uintptr_t)p2[0])
	return -1;
    if ((uintptr_t)p1[0] > (uintptr_t)p2[0])
	return 1;
    if ((uintptr_t)p1[1] < (uintptr_t)p2[1])
	return -1;
    if ((uintptr_t)p1[1] > (uintptr_t)p2[1])
	return 1;
    return 0;
}

static void *newItem(void *p, Dtdisc_t *disc) {
    item *objp = p;
    item *newp = gv_alloc(sizeof(item));

    (void)disc;
    newp->p[0] = objp->p[0];
    newp->p[1] = objp->p[1];
    newp->t = objp->t;
    newp->h = objp->h;

    return newp;
}

static Dtdisc_t mapDisc = {
    .key = offsetof(item, p),
    .size = sizeof(2 * sizeof(void *)),
    .link = offsetof(item, link),
    .makef = newItem,
    .freef = free,
    .comparf = cmpItem,
};

/// Make a copy of e in e's graph but using ct and ch as nodes
static edge_t *cloneEdge(edge_t * e, node_t * ct, node_t * ch)
{
    graph_t *g = agraphof(ct);
    edge_t *ce = agedge(g, ct, ch,NULL,1);
    agbindrec(ce, "Agedgeinfo_t", sizeof(Agedgeinfo_t), true);
    agcopyattr(e, ce);
    ED_compound(ce) = true;

    return ce;
}

static void insertEdge(Dt_t * map, void *t, void *h, edge_t * e)
{
    item dummy1 = {.p = {t, h}, .t = agtail(e), .h = aghead(e)};
    dtinsert(map, &dummy1);

    item dummy2 = {.p = {h, t}, .t = aghead(e), .h = agtail(e)};
    dtinsert(map, &dummy2);
}

/// Check if we already have cluster edge corresponding to t->h, and return it.
static item *mapEdge(Dt_t * map, edge_t * e)
{
    void *key[] = {agtail(e), aghead(e)};
    return dtmatch(map, &key);
}

static graph_t *mapc(Dt_t *cmap, node_t *n) {
  if (startswith(agnameof(n), "cluster")) {
    return findCluster(cmap, agnameof(n));
  }
  return NULL;
}

/** If endpoint names a cluster, mark for temporary deletion and create
 * special node and insert into cluster. Then clone the edge. Real edge
 * will be deleted when we delete the original node.
 * Invariant: new edge has same sense as old. That is, given t->h with
 * t and h mapped to ct and ch, the new edge is ct->ch.
 *
 * In the current model, we create a cluster node for each cluster edge
 * between the cluster and some other node or cluster, treating the
 * cluster node as a port on the cluster. This should help with better
 * routing to avoid edge crossings. At present, this is not implemented,
 * so we could use a simpler model in which we create a single cluster
 * node for each cluster used in a cluster edge.
 *
 * Return 1 if cluster edge is created.
 */
static int
checkCompound(edge_t * e, graph_t * clg, agxbuf * xb, Dt_t * map, Dt_t* cmap)
{
    node_t *cn;
    node_t *cn1;
    node_t *t = agtail(e);
    node_t *h = aghead(e);
    edge_t *ce;
    item *ip;

    if (IS_CLUST_NODE(h)) return 0;
    graph_t *const tg = mapc(cmap, t);
    graph_t *const hg = mapc(cmap, h);
    if (!tg && !hg)
	return 0;
    if (tg == hg) {
	agwarningf("cluster cycle %s -- %s not supported\n", agnameof(t),
	      agnameof(t));
	return 0;
    }
    ip = mapEdge(map, e);
    if (ip) {
	cloneEdge(e, ip->t, ip->h);
	return 1;
    }

    if (hg) {
	if (tg) {
	    if (agcontains(hg, tg)) {
		agwarningf("tail cluster %s inside head cluster %s\n",
		      agnameof(tg), agnameof(hg));
		return 0;
	    }
	    if (agcontains(tg, hg)) {
		agwarningf("head cluster %s inside tail cluster %s\n",
		      agnameof(hg),agnameof(tg));
		return 0;
	    }
	    cn = clustNode(t, tg, xb, clg);
	    cn1 = clustNode(h, hg, xb, clg);
	    ce = cloneEdge(e, cn, cn1);
	    insertEdge(map, t, h, ce);
	} else {
	    if (agcontains(hg, t)) {
		agwarningf("tail node %s inside head cluster %s\n",
		      agnameof(t), agnameof(hg));
		return 0;
	    }
	    cn = clustNode(h, hg, xb, clg);
	    ce = cloneEdge(e, t, cn);
	    insertEdge(map, t, h, ce);
	}
    } else {
	if (agcontains(tg, h)) {
	    agwarningf("head node %s inside tail cluster %s\n", agnameof(h),
		  agnameof(tg));
	    return 0;
	}
	cn = clustNode(t, tg, xb, clg);
	ce = cloneEdge(e, cn, h);
	insertEdge(map, t, h, ce);
    }
    return 1;
}

typedef struct {
    Agrec_t hdr;
    int n_cluster_edges;
} cl_edge_t;

static int
num_clust_edges(graph_t * g)
{
    cl_edge_t* cl_info = (cl_edge_t*)HAS_CLUST_EDGE(g);
    if (cl_info)
	return cl_info->n_cluster_edges;
    return 0;
}

/** Look for cluster edges. Replace cluster edge endpoints
 * corresponding to a cluster with special cluster nodes.
 * Delete original nodes.
 * If cluster edges are found, a cl_edge_t record will be
 * attached to the graph, containing the count of such edges.
 */
void processClusterEdges(graph_t * g)
{
    int num_cl_edges = 0;
    node_t *n;
    node_t *nxt;
    edge_t *e;
    graph_t *clg;
    agxbuf xb = {0};
    Dt_t *map;
    Dt_t *cmap = mkClustMap (g);

    map = dtopen(&mapDisc, Dtoset);
    clg = agsubg(g, "__clusternodes",1);
    agbindrec(clg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if (IS_CLUST_NODE(n)) continue;
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    num_cl_edges += checkCompound(e, clg, &xb, map, cmap);
	}
    }
    agxbfree(&xb);
    dtclose(map);
    for (n = agfstnode(clg); n; n = nxt) {
	nxt = agnxtnode(clg, n);
	agdelete(g, n);
    }
    agclose(clg);
    if (num_cl_edges) {
	cl_edge_t* cl_info;
	cl_info = agbindrec(g, CL_EDGE_TAG, sizeof(cl_edge_t), false);
	cl_info->n_cluster_edges = num_cl_edges;
    }
    dtclose(cmap);
}

/** Convert cluster nodes back to ordinary nodes
 * If n is already ordinary, return it.
 * Otherwise, we know node's name is "__i:xxx"
 * where i is some number and xxx is the nodes's original name.
 * Create new node of name xxx if it doesn't exist and add n to clg
 * for later deletion.
 */
static node_t *mapN(node_t * n, graph_t * clg)
{
    node_t *nn;
    char *name;
    graph_t *g = agraphof(n);
    Agsym_t *sym;

    if (!IS_CLUST_NODE(n))
	return n;
    agsubnode(clg, n, 1);
    name = strchr(agnameof(n), ':');
    assert(name);
    name++;
    if ((nn = agfindnode(g, name)))
	return nn;
    nn = agnode(g, name, 1);
    agbindrec(nn, "Agnodeinfo_t", sizeof(Agnodeinfo_t), true);
    SET_CLUST_NODE(nn);

    /* Set all attributes to default */
    for (sym = agnxtattr(g, AGNODE, NULL); sym;  (sym = agnxtattr(g, AGNODE, sym))) {
	if (agxget(nn, sym) != sym->defval)
	    agxset(nn, sym, sym->defval);
    }
    return nn;
}

static void undoCompound(edge_t * e, graph_t * clg)
{
    node_t *t = agtail(e);
    node_t *h = aghead(e);
    node_t *ntail;
    node_t *nhead;
    edge_t* ce;

    ntail = mapN(t, clg);
    nhead = mapN(h, clg);
    ce = cloneEdge(e, ntail, nhead);

    /* transfer drawing information */
    ED_spl(ce) = ED_spl(e);
    ED_spl(e) = NULL;
    ED_label(ce) = ED_label(e);
    ED_label(e) = NULL;
    ED_xlabel(ce) = ED_xlabel(e);
    ED_xlabel(e) = NULL;
    ED_head_label(ce) = ED_head_label(e);
    ED_head_label(e) = NULL;
    ED_tail_label(ce) = ED_tail_label(e);
    ED_tail_label(e) = NULL;
    gv_cleanup_edge(e);
}

/** Replace cluster nodes with originals. Make sure original has
 * no attributes. Replace original edges. Delete cluster nodes,
 * which will also delete cluster edges.
 */
void undoClusterEdges(graph_t * g)
{
    node_t *n;
    node_t *nextn;
    edge_t *e;
    graph_t *clg;
    int ecnt = num_clust_edges(g);
    int i = 0;

    if (!ecnt) return;
    clg = agsubg(g, "__clusternodes",1);
    agbindrec(clg, "Agraphinfo_t", sizeof(Agraphinfo_t), true);
    edge_t **edgelist = gv_calloc(ecnt, sizeof(edge_t*));
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    if (ED_compound(e))
		edgelist[i++] = e;
	}
    }
    assert(i == ecnt);
    for (i = 0; i < ecnt; i++)
	undoCompound(edgelist[i], clg);
    free (edgelist);
    for (n = agfstnode(clg); n; n = nextn) {
	nextn = agnxtnode(clg, n);
	gv_cleanup_node(n);
	agdelete(g, n);
    }
    agclose(clg);
}

/** Find the attribute belonging to graph g for objects like obj
 * with given name. If one does not exist, create it with the
 * default value defaultValue.
 */
attrsym_t *safe_dcl(graph_t *g, int obj_kind, char *name, char *defaultValue) {
    attrsym_t *a = agattr_text(g,obj_kind,name, NULL);
    if (!a)	/* attribute does not exist */
        a = agattr_text(g, obj_kind, name, defaultValue);
    return a;
}

static int comp_entities(const void *e1, const void *e2) {
  const strview_t *key = e1;
  const struct entities_s *candidate = e2;
  return strview_cmp(*key, strview(candidate->name, '\0'));
}

/** Scan non-numeric entity, convert to &#...; form and store in xbuf.
 * t points to first char after '&'. Return after final semicolon.
 * If unknown, we return t and let libexpat flag the error.
 *     */
char* scanEntity (char* t, agxbuf* xb)
{
    const strview_t key = strview(t, ';');
    struct entities_s *res;

    agxbputc(xb, '&');
    if (key.data[key.size] == '\0') return t;
    if (key.size > ENTITY_NAME_LENGTH_MAX || key.size < 2) return t;
    res = bsearch(&key, entities, NR_OF_ENTITIES,
        sizeof(entities[0]), comp_entities);
    if (!res) return t;
    agxbprint(xb, "#%d;", res->value);
    return t + key.size + 1;
}

/** Check for an HTML entity for a special character.
 * Assume *s points to first byte after '&'.
 * If successful, return the corresponding value and update s to
 * point after the terminating ';'.
 * On failure, return 0 and leave s unchanged.
 */
static int
htmlEntity (char** s)
{
    struct entities_s *res;
    unsigned char* str = *(unsigned char**)s;
    unsigned int byte;
    int i, n = 0;

    byte = *str;
    if (byte == '#') {
	byte = *(str + 1);
	if (byte == 'x' || byte == 'X') {
	    for (i = 2; i < 8; i++) {
		byte = *(str + i);
		if (byte >= 'A' && byte <= 'F')
                    byte = byte - 'A' + 10;
		else if (byte >= 'a' && byte <= 'f')
                    byte = byte - 'a' + 10;
		else if (byte >= '0' && byte <= '9')
                    byte = byte - '0';
		else
                    break;
		n = n * 16 + (int)byte;
	    }
	}
	else {
	    for (i = 1; i < 8; i++) {
		byte = *(str + i);
		if (byte >= '0' && byte <= '9')
		    n = n * 10 + ((int)byte - '0');
		else
		    break;
	    }
	}
	if (byte == ';') {
	    str += i+1;
	}
	else {
	    n = 0;
	}
    }
    else {
	strview_t key = {.data = (char *)str};
	for (i = 0; i <  ENTITY_NAME_LENGTH_MAX; i++) {
	    byte = *(str + i);
	    if (byte == '\0') break;
	    if (byte == ';') {
		res = bsearch(&key, entities, NR_OF_ENTITIES,
		    sizeof(entities[0]), comp_entities);
		if (res) {
		    n = res->value;
		    str += i+1;
		}
		break;
	    }
	    ++key.size;
	}
    }
    *s = (char*)str;
    return n;
}

static unsigned char
cvtAndAppend (unsigned char c, agxbuf* xb)
{
    char buf[] = {c, '\0'};
    char *s = latin1ToUTF8(buf);
    char *p = s;
    size_t len = strlen(s);
    while (len-- > 1)
	agxbputc(xb, *p++);
    c = *p;
    free (s);
    return c;
}

/** substitute html entities like: &#123; and: &amp; with the UTF8 equivalents
 * check for invalid utf8. If found, treat a single byte as Latin-1, convert it to
 * utf8 and warn the user.
 */
char* htmlEntityUTF8 (char* s, graph_t* g)
{
    static graph_t* lastg;
    static atomic_flag warned;
    unsigned char c;
    unsigned int v;

    int uc;
    int ui;

    if (lastg != g) {
	lastg = g;
	atomic_flag_clear(&warned);
    }

    agxbuf xb = {0};

    while ((c = *(unsigned char*)s++)) {
        if (c < 0xC0)
	    /*
	     * Handles properly formed UTF-8 characters between
	     * 0x01 and 0x7F.  Also treats \0 and naked trail
	     * bytes 0x80 to 0xBF as valid characters representing
	     * themselves.
	     */
            uc = 0;
        else if (c < 0xE0)
            uc = 1;
        else if (c < 0xF0)
            uc = 2;
        else if (c < 0xF8)
            uc = 3;
        else {
            uc = -1;
            if (!atomic_flag_test_and_set(&warned)) {
                agwarningf("UTF8 codes > 4 bytes are not currently supported (graph %s) - treated as Latin-1. Perhaps \"-Gcharset=latin1\" is needed?\n", agnameof(g));
            }
            c = cvtAndAppend (c, &xb);
        }

	    if (uc == 0 && c == '&') {
		/* replace html entity sequences like: &amp;
		 * and: &#123; with their UTF8 equivalents */
	        v = htmlEntity (&s);
	        if (v) {
		    if (v < 0x7F) /* entity needs 1 byte in UTF8 */
			c = v;
		    else if (v < 0x07FF) { /* entity needs 2 bytes in UTF8 */
			agxbputc(&xb, (char)((v >> 6) | 0xC0));
			c = (v & 0x3F) | 0x80;
		    }
		    else { /* entity needs 3 bytes in UTF8 */
			agxbputc(&xb, (char)((v >> 12) | 0xE0));
			agxbputc(&xb, (char)(((v >> 6) & 0x3F) | 0x80));
			c = (v & 0x3F) | 0x80;
		    }
		    }
        }
        else /* copy n byte UTF8 characters */
            for (ui = 0; ui < uc; ++ui)
                if ((*s & 0xC0) == 0x80) {
                    agxbputc(&xb, (char)c);
                    c = *(unsigned char*)s++;
                }
                else {
		            if (!atomic_flag_test_and_set(&warned)) {
		                agwarningf("Invalid %d-byte UTF8 found in input of graph %s - treated as Latin-1. Perhaps \"-Gcharset=latin1\" is needed?\n", uc + 1, agnameof(g));
		            }
		            c = cvtAndAppend (c, &xb);
                    break;
	            }
        agxbputc(&xb, (char)c);
    }
    return agxbdisown(&xb);
}

/// Converts string from Latin1 encoding to utf8. Also translates HTML entities.
char* latin1ToUTF8 (char* s)
{
    agxbuf xb = {0};
    unsigned int v;

    /* Values are either a byte (<= 256) or come from htmlEntity, whose
     * values are all less than 0x07FF, so we need at most 3 bytes.
     */
    while ((v = *(unsigned char*)s++)) {
	if (v == '&') {
	    v = htmlEntity (&s);
	    if (!v) v = '&';
        }
	if (v < 0x7F)
	    agxbputc(&xb, (char)v);
	else if (v < 0x07FF) {
	    agxbputc(&xb, (char)((v >> 6) | 0xC0));
	    agxbputc(&xb, (char)((v & 0x3F) | 0x80));
	}
	else {
	    agxbputc(&xb, (char)((v >> 12) | 0xE0));
	    agxbputc(&xb, (char)(((v >> 6) & 0x3F) | 0x80));
	    agxbputc(&xb, (char)((v & 0x3F) | 0x80));
	}
    }
    return agxbdisown(&xb);
}

/** Converts string from utf8 encoding to Latin1
 * Note that it does not attempt to reproduce HTML entities.
 * We assume the input string comes from latin1ToUTF8.
 */
char*
utf8ToLatin1 (char* s)
{
    agxbuf xb = {0};
    unsigned char c;

    while ((c = *(unsigned char*)s++)) {
	if (c < 0x7F)
	    agxbputc(&xb, (char)c);
	else {
            unsigned char outc = (c & 0x03) << 6;
            c = *(unsigned char *)s++;
            outc = outc | (c & 0x3F);
	    agxbputc(&xb, (char)outc);
	}
    }
    return agxbdisown(&xb);
}

bool overlap_node(node_t *n, boxf b) {
    if (! OVERLAP(b, ND_bb(n)))
        return false;

    /*  FIXME - need to do something better about CLOSEENOUGH */
    pointf p = sub_pointf(ND_coord(n), mid_pointf(b.UR, b.LL));

    inside_t ictxt = {.s.n = n};

    return ND_shape(n)->fns->insidefn(&ictxt, p);
}

bool overlap_label(textlabel_t *lp, boxf b)
{
    const pointf s = {.x = lp->dimen.x / 2.0, .y = lp->dimen.y / 2.0};
    boxf bb = {.LL = sub_pointf(lp->pos, s), .UR = add_pointf(lp->pos, s)};
    return OVERLAP(b, bb);
}

static bool overlap_arrow(pointf p, pointf u, double scale, boxf b)
{
    // FIXME - check inside arrow shape
    return OVERLAP(b, arrow_bb(p, u, scale));
}

static bool overlap_bezier(bezier bz, boxf b) {
    assert(bz.size);
    pointf u = bz.list[0];
    for (size_t i = 1; i < bz.size; i++) {
        pointf p = bz.list[i];
        if (lineToBox(p, u, b) != -1)
            return true;
	u = p;
    }

    /* check arrows */
    if (bz.sflag) {
	if (overlap_arrow(bz.sp, bz.list[0], 1, b))
	    return true;
    }
    if (bz.eflag) {
	if (overlap_arrow(bz.ep, bz.list[bz.size - 1], 1, b))
	    return true;
    }
    return false;
}

bool overlap_edge(edge_t *e, boxf b)
{
    splines *spl = ED_spl(e);
    if (spl && boxf_overlap(spl->bb, b))
        for (size_t i = 0; i < spl->size; i++)
            if (overlap_bezier(spl->list[i], b))
                return true;

    textlabel_t *lp = ED_label(e);
    if (lp && overlap_label(lp, b))
        return true;

    return false;
}

/// Convert string to edge type.
static int edgeType(const char *s, int defaultValue) {
    if (s == NULL || strcmp(s, "") == 0) {
        return defaultValue;
    }

    if (*s == '0') { /* false */
	return EDGETYPE_LINE;
    } else if (*s >= '1' && *s <= '9') { /* true */
	return EDGETYPE_SPLINE;
    } else if (strcasecmp(s, "curved") == 0) {
	return EDGETYPE_CURVED;
    } else if (strcasecmp(s, "compound") == 0) {
	return EDGETYPE_COMPOUND;
    } else if (strcasecmp(s, "false") == 0) {
	return EDGETYPE_LINE;
    } else if (strcasecmp(s, "line") == 0) {
	return EDGETYPE_LINE;
    } else if (strcasecmp(s, "none") == 0) {
	return EDGETYPE_NONE;
    } else if (strcasecmp(s, "no") == 0) {
	return EDGETYPE_LINE;
    } else if (strcasecmp(s, "ortho") == 0) {
	return EDGETYPE_ORTHO;
    } else if (strcasecmp(s, "polyline") == 0) {
	return EDGETYPE_PLINE;
    } else if (strcasecmp(s, "spline") == 0) {
	return EDGETYPE_SPLINE;
    } else if (strcasecmp(s, "true") == 0) {
	return EDGETYPE_SPLINE;
    } else if (strcasecmp(s, "yes") == 0) {
	return EDGETYPE_SPLINE;
    }

    agwarningf("Unknown \"splines\" value: \"%s\" - ignored\n", s);
    return defaultValue;
}

/** Sets graph's edge type based on the "splines" attribute.
 * If the attribute is not defined, use defaultValue.
 * If the attribute is "", use NONE.
 * If attribute value matches (case indepedent), use match.
 *   ortho => EDGETYPE_ORTHO
 *   none => EDGETYPE_NONE
 *   line => EDGETYPE_LINE
 *   polyline => EDGETYPE_PLINE
 *   spline => EDGETYPE_SPLINE
 * If attribute is boolean, true means EDGETYPE_SPLINE, false means
 * EDGETYPE_LINE. Else warn and use default.
 */
void setEdgeType(graph_t *g, int defaultValue) {
    char* s = agget(g, "splines");
    int et;

    if (!s) {
        et = defaultValue;
    }
    else if (*s == '\0') {
	et = EDGETYPE_NONE;
    } else {
        et = edgeType(s, defaultValue);
    }
    GD_flags(g) |= et;
}

/** Evaluates the extreme points of an ellipse or polygon
 * Determines the point at the center of the extreme points
 * If isRadial is true,sets the inner radius to half the distance to the min point;
 * else uses the angle parameter to identify two points on a line that defines the
 * gradient direction
 * By default, this assumes a left-hand coordinate system (for svg); if RHS = 2 flag
 * is set, use standard coordinate system.
 */
void get_gradient_points(pointf *A, pointf *G, size_t n, double angle, int flags) {
    pointf min,max,center;
    int isRadial = flags & 1;
    int isRHS = flags & 2;

    if (n == 2) {
        double rx = A[1].x - A[0].x;
        double ry = A[1].y - A[0].y;
        min.x = A[0].x - rx;
        max.x = A[0].x + rx;
        min.y = A[0].y - ry;
        max.y = A[0].y + ry;
    }
    else {
      min.x = max.x = A[0].x;
      min.y = max.y = A[0].y;
      for (size_t i = 0; i < n; i++) {
            min.x = MIN(A[i].x, min.x);
            min.y = MIN(A[i].y, min.y);
            max.x = MAX(A[i].x, max.x);
            max.y = MAX(A[i].y, max.y);
      }
    }
      center.x = min.x + (max.x - min.x)/2;
      center.y = min.y + (max.y - min.y)/2;
    if (isRadial) {
	double inner_r, outer_r;
	outer_r = hypot(center.x - min.x, center.y - min.y);
	inner_r = outer_r /4.;
	if (isRHS) {
	    G[0].y = center.y;
	}
	else {
	    G[0].y = -center.y;
	}
	G[0].x = center.x;
	G[1].x = inner_r;
	G[1].y = outer_r;
    }
    else {
	double half_x = max.x - center.x;
	double half_y = max.y - center.y;
	double sina = sin(angle);
	double cosa = cos(angle);
	if (isRHS) {
	    G[0].y = center.y - half_y * sina;
	    G[1].y = center.y + half_y * sina;
	}
	else {
	    G[0].y = -center.y + (max.y - center.y) * sin(angle);
	    G[1].y = -center.y - (center.y - min.y) * sin(angle);
	}
	G[0].x = center.x - half_x * cosa;
	G[1].x = center.x + half_x * cosa;
    }
}

void gv_free_splines(edge_t *e) {
    if (ED_spl(e)) {
        for (size_t i = 0; i < ED_spl(e)->size; i++)
            free(ED_spl(e)->list[i].list);
        free(ED_spl(e)->list);
        free(ED_spl(e));
    }
    ED_spl(e) = NULL;
}

void gv_cleanup_edge(edge_t * e)
{
    free(ED_path(e).ps);
    gv_free_splines(e);
    free_label(ED_label(e));
    free_label(ED_xlabel(e));
    free_label(ED_head_label(e));
    free_label(ED_tail_label(e));
    /*FIX HERE , shallow cleaning may not be enough here */
    agdelrec(e, "Agedgeinfo_t");
}

void gv_cleanup_node(node_t * n)
{
    free(ND_pos(n));
    if (ND_shape(n))
        ND_shape(n)->fns->freefn(n);
    free_label(ND_label(n));
    free_label(ND_xlabel(n));
    /*FIX HERE , shallow cleaning may not be enough here */
    agdelrec(n, "Agnodeinfo_t");
}

void gv_nodesize(node_t *n, bool flip) {
    if (flip) {
        double w = INCH2PS(ND_height(n));
        ND_lw(n) = ND_rw(n) = w / 2;
        ND_ht(n) = INCH2PS(ND_width(n));
    }
    else {
        double w = INCH2PS(ND_width(n));
        ND_lw(n) = ND_rw(n) = w / 2;
        ND_ht(n) = INCH2PS(ND_height(n));
    }
}

#ifndef HAVE_DRAND48
double drand48(void)
{
    double d;
    d = rand();
    d = d / RAND_MAX;
    return d;
}
#endif
typedef struct {
    Dtlink_t link;
    char* name;
    Agraph_t* clp;
} clust_t;

static Dtdisc_t strDisc = {
    .key = offsetof(clust_t, name),
    .size = -1,
    .link = offsetof(clust_t, link),
    .freef = free,
};

static void fillMap (Agraph_t* g, Dt_t* map)
{
    for (int c = 1; c <= GD_n_cluster(g); c++) {
        Agraph_t *cl = GD_clust(g)[c];
        char *s = agnameof(cl);
        if (dtmatch(map, s)) {
            agwarningf("Two clusters named %s - the second will be ignored\n", s);
        } else {
            clust_t *ip = gv_alloc(sizeof(clust_t));
            ip->name = s;
            ip->clp = cl;
	    dtinsert (map, ip);
        }
        fillMap (cl, map);
    }
}

/** Generates a dictionary mapping cluster names to corresponding cluster.
 * Used with cgraph as the latter does not support a flat namespace of clusters.
 * Assumes G has already built a cluster tree using GD_n_cluster and GD_clust.
 */
Dt_t* mkClustMap (Agraph_t* g)
{
    Dt_t* map = dtopen (&strDisc, Dtoset);

    fillMap (g, map);

    return map;
}

Agraph_t*
findCluster (Dt_t* map, char* name)
{
    clust_t* clp = dtmatch (map, name);
    if (clp)
	return clp->clp;
    return NULL;
}

/**
 * @dir lib/common
 * @brief common code for layout engines
 * @ingroup engines
 *
 * @ref common_utils
 *
 * @defgroup common_utils utilities
 * @brief low level utilities for layout engines and rendering
 * @ingroup engines
 */
