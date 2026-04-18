/// @file
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

#include <common/render.h>
#include <common/htmltable.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/streq.h>

static char *strdup_and_subst_obj0 (char *str, void *obj, int escBackslash);

static void storeline(GVC_t *gvc, textlabel_t *lp, char *line,
                      char terminator) {
    pointf size;
    textspan_t *span;
    size_t oldsz = lp->u.txt.nspans + 1;

    lp->u.txt.span = gv_recalloc(lp->u.txt.span, oldsz, oldsz + 1,
                                 sizeof(textspan_t));
    span = &lp->u.txt.span[lp->u.txt.nspans];
    span->str = line;
    span->just = terminator;
    if (line && line[0]) {
	textfont_t tf = {0};
	tf.name = lp->fontname;
	tf.size = lp->fontsize;
	span->font = dtinsert(gvc->textfont_dt, &tf);
        size = textspan_size(gvc, span);
    }
    else {
	size.x = 0.0;
	span->size.y = size.y = (int)(lp->fontsize * LINESPACING);
    }

    lp->u.txt.nspans++;
    /* width = max line width */
    lp->dimen.x = MAX(lp->dimen.x, size.x);
    /* accumulate height */
    lp->dimen.y += size.y;
}

/* compiles <str> into a label <lp> */
void make_simple_label(GVC_t * gvc, textlabel_t * lp)
{
    lp->dimen.x = lp->dimen.y = 0.0;
    if (*lp->text == '\0')
	return;

    agxbuf line = {0};
    for (char c, *p = lp->text; (c = *p++);) {
	unsigned char byte = (unsigned char) c;
	/* wingraphviz allows a combination of ascii and big-5. The latter
         * is a two-byte encoding, with the first byte in 0xA1-0xFE, and
         * the second in 0x40-0x7e or 0xa1-0xfe. We assume that the input
         * is well-formed, but check that we don't go past the ending '\0'.
         */
	if (lp->charset == CHAR_BIG5 && 0xA1 <= byte && byte <= 0xFE) {
	    agxbputc(&line, c);
	    c = *p++;
	    agxbputc(&line, c);
	    if (!c) /* NB. Protect against unexpected string end here */
		break;
	} else {
	    if (c == '\\') {
		switch (*p) {
		case 'n':
		case 'l':
		case 'r':
		    storeline(gvc, lp, agxbdisown(&line), *p);
		    break;
		default:
		    agxbputc(&line, *p);
		}
		if (*p)
		    p++;
		/* tcldot can enter real linend characters */
	    } else if (c == '\n') {
		storeline(gvc, lp, agxbdisown(&line), 'n');
	    } else {
		agxbputc(&line, c);
	    }
	}
    }

    if (agxblen(&line) > 0) {
	storeline(gvc, lp, agxbdisown(&line), 'n');
    }

    agxbfree(&line);
    lp->space = lp->dimen;
}

/* Assume str is freshly allocated for this instance, so it
 * can be freed in free_label.
 */
textlabel_t *make_label(void *obj, char *str, bool is_html, bool is_record,
                        double fontsize, char *fontname, char *fontcolor) {
    textlabel_t *rv = gv_alloc(sizeof(textlabel_t));
    graph_t *g = NULL, *sg = NULL;
    node_t *n = NULL;
    edge_t *e = NULL;
        char *s;

    // disregard HTML intent for empty string labels
    is_html &= !streq(str, "");

    switch (agobjkind(obj)) {
    case AGRAPH:
        sg = obj;
	g = sg->root;
	break;
    case AGNODE:
        n = obj;
	g = agroot(agraphof(n));
	break;
    case AGEDGE:
        e = obj;
	g = agroot(agraphof(aghead(e)));
	break;
    }
    rv->fontname = fontname;
    rv->fontcolor = fontcolor;
    rv->fontsize = fontsize;
    rv->charset = GD_charset(g);
    if (is_record) {
	rv->text = gv_strdup(str);
        if (is_html) {
	    rv->html = true;
	}
    }
    else if (is_html) {
	rv->text = gv_strdup(str);
	rv->html = true;
	if (make_html_label(obj, rv)) {
	    switch (agobjkind(obj)) {
	    case AGRAPH:
	        agerr(AGPREV, "in label of graph %s\n",agnameof(sg));
		break;
	    case AGNODE:
	        agerr(AGPREV, "in label of node %s\n", agnameof(n));
		break;
	    case AGEDGE:
		agerr(AGPREV, "in label of edge %s %s %s\n",
		        agnameof(agtail(e)), agisdirected(g)?"->":"--", agnameof(aghead(e)));
		break;
	    }
	}
    }
    else {
        assert(!is_record);
        assert(!is_html);
	/* This call just processes the graph object based escape sequences. The formatting escape
         * sequences (\n, \l, \r) are processed in make_simple_label. That call also replaces \\ with \.
         */
	rv->text = strdup_and_subst_obj0(str, obj, 0);
        switch (rv->charset) {
	case CHAR_LATIN1:
	    s = latin1ToUTF8(rv->text);
	    break;
	default: /* UTF8 */
	    s = htmlEntityUTF8(rv->text, g);
	    break;
	}
        free(rv->text);
        rv->text = s;
	make_simple_label(GD_gvc(g), rv);
    }
    return rv;
}

/* Free resources related to textspan_t.
 * tl is an array of cnt textspan_t's.
 * It is also assumed that the text stored in the str field
 * is all stored in one large buffer shared by all of the textspan_t,
 * so only the first one needs to free its tlp->str.
 */
void free_textspan(textspan_t *tl, size_t cnt) {
    textspan_t* tlp = tl;

    if (!tl) return;
    for (size_t i = 0; i < cnt; i++) {
	free(tlp->str);
	if (tlp->layout && tlp->free_layout)
	    tlp->free_layout (tlp->layout);
	tlp++;
    }
    free(tl);
}

void free_label(textlabel_t * p)
{
    if (p) {
	free(p->text);
	if (p->html) {
	    if (p->u.html) free_html_label(p->u.html, 1);
	} else {
	    free_textspan(p->u.txt.span, p->u.txt.nspans);
	}
	free(p);
    }
}

void emit_label(GVJ_t * job, emit_state_t emit_state, textlabel_t * lp)
{
    obj_state_t *obj = job->obj;
    pointf p;
    emit_state_t old_emit_state;

    old_emit_state = obj->emit_state;
    obj->emit_state = emit_state;

    if (lp->html) {
	emit_html_label(job, lp->u.html, lp);
	obj->emit_state = old_emit_state;
	return;
    }

    /* make sure that there is something to do */
    if (lp->u.txt.nspans < 1)
	return;

    gvrender_begin_label(job, LABEL_PLAIN);
    gvrender_set_pencolor(job, lp->fontcolor);

    /* position for first span */
    switch (lp->valign) {
	case 't':
    	    p.y = lp->pos.y + lp->space.y / 2.0 - lp->fontsize;
	    break;
	case 'b':
    	    p.y = lp->pos.y - lp->space.y / 2.0 + lp->dimen.y - lp->fontsize;
	    break;
	case 'c':
	default:	
    	    p.y = lp->pos.y + lp->dimen.y / 2.0 - lp->fontsize;
	    break;
    }
    if (obj->labeledgealigned)
	p.y -= lp->pos.y;
    for (size_t i = 0; i < lp->u.txt.nspans; i++) {
	switch (lp->u.txt.span[i].just) {
	case 'l':
	    p.x = lp->pos.x - lp->space.x / 2.0;
	    break;
	case 'r':
	    p.x = lp->pos.x + lp->space.x / 2.0;
	    break;
	default:
	case 'n':
	    p.x = lp->pos.x;
	    break;
	}
	gvrender_textspan(job, p, &lp->u.txt.span[i]);

	/* UL position for next span */
	p.y -= lp->u.txt.span[i].size.y;
    }

    gvrender_end_label(job);
    obj->emit_state = old_emit_state;
}

/* Replace various escape sequences with the name of the associated
 * graph object. A double backslash \\ can be used to avoid a replacement.
 * If escBackslash is true, convert \\ to \; else leave alone. All other dyads 
 * of the form \. are passed through unchanged.
 */
static char *strdup_and_subst_obj0 (char *str, void *obj, int escBackslash)
{
    char c, *s;
    char *tp_str = "", *hp_str = "";
    char *g_str = "\\G", *n_str = "\\N", *e_str = "\\E",
	*h_str = "\\H", *t_str = "\\T", *l_str = "\\L";
    bool has_hp = false;
    bool has_tp = false;
    int isEdge = 0;
    textlabel_t *tl;
    port pt;

    /* prepare substitution strings */
    switch (agobjkind(obj)) {
	case AGRAPH:
	    g_str = agnameof(obj);
	    tl = GD_label(obj);
	    if (tl) {
		l_str = tl->text;
	    }
	    break;
	case AGNODE:
	    g_str = agnameof(agraphof(obj));
	    n_str = agnameof(obj);
	    tl = ND_label(obj);
	    if (tl) {
		l_str = tl->text;
	    }
	    break;
	case AGEDGE:
	    isEdge = 1;
	    g_str = agnameof(agroot(agraphof(agtail(((edge_t *)obj)))));
	    t_str = agnameof(agtail(((edge_t *)obj)));
	    pt = ED_tail_port(obj);
	    if ((tp_str = pt.name))
	        has_tp = *tp_str != '\0';
	    h_str = agnameof(aghead(((edge_t *)obj)));
	    pt = ED_head_port(obj);
	    if ((hp_str = pt.name))
		has_hp = *hp_str != '\0';
	    tl = ED_label(obj);
	    if (tl) {
		l_str = tl->text;
	    }
	    if (agisdirected(agroot(agraphof(agtail(((edge_t*)obj))))))
		e_str = "->";
	    else
		e_str = "--";
	    break;
    }

    /* allocate a dynamic buffer that we will use to construct the result */
    agxbuf buf = {0};

    /* assemble new string */
    for (s = str; (c = *s++);) {
	if (c == '\\' && *s != '\0') {
	    switch (c = *s++) {
	    case 'G':
		agxbput(&buf, g_str);
		break;
	    case 'N':
		agxbput(&buf, n_str);
		break;
	    case 'E':
		if (isEdge) {
		    agxbput(&buf, t_str);
		    if (has_tp) {
			agxbprint(&buf, ":%s", tp_str);
		    }
		    agxbprint(&buf, "%s%s", e_str, h_str);
		    if (has_hp) {
			agxbprint(&buf, ":%s", hp_str);
		    }
		}
		break;
	    case 'T':
		agxbput(&buf, t_str);
		break;
	    case 'H':
		agxbput(&buf, h_str);
		break;
	    case 'L':
		agxbput(&buf, l_str);
		break;
	    case '\\':
		if (escBackslash) {
		    agxbputc(&buf, '\\');
		    break; 
		}
		/* Fall through */
	    default:  /* leave other escape sequences unmodified, e.g. \n \l \r */
		agxbprint(&buf, "\\%c", c);
		break;
	    }
	} else {
	    agxbputc(&buf, c);
	}
    }

    /* extract the final string with replacements applied */
    return agxbdisown(&buf);
}

/// Processes graph object escape sequences; also collapses \\ to \.
char *strdup_and_subst_obj(char *str, void *obj)
{
    return strdup_and_subst_obj0 (str, obj, 1);
}
