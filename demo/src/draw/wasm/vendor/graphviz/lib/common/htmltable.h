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

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <util/list.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FIXED_FLAG 1
#define HALIGN_RIGHT (1 << 1)
#define HALIGN_LEFT (1 << 2)
#define HALIGN_MASK (HALIGN_RIGHT | HALIGN_LEFT)
#define HALIGN_TEXT HALIGN_MASK
#define VALIGN_TOP (1 << 3)
#define VALIGN_BOTTOM (1 << 4)
#define VALIGN_MASK (VALIGN_TOP | VALIGN_BOTTOM)
#define BORDER_SET (1 << 5)
#define PAD_SET (1 << 6)
#define SPACE_SET (1 << 7)
#define BALIGN_RIGHT (1 << 8)
#define BALIGN_LEFT (1 << 9)
#define BALIGN_MASK (BALIGN_RIGHT | BALIGN_LEFT)
#define BORDER_LEFT (1 << 10)
#define BORDER_TOP (1 << 11)
#define BORDER_RIGHT (1 << 12)
#define BORDER_BOTTOM (1 << 13)
#define BORDER_MASK (BORDER_LEFT|BORDER_TOP|BORDER_RIGHT|BORDER_BOTTOM)

#define UNSET_ALIGN 0

    /* spans of text within a cell
     * NOTE: As required, the str field in span is utf-8.
     * This translation is done when libexpat scans the input.
     */
	
    /* line of textspan_t's */
    typedef struct {
	textspan_t *items;
	size_t nitems;
	char just;
	double size;   /* width of span */
	double lfsize; /* offset from previous baseline to current one */
    } htextspan_t;
	
    typedef struct {
	htextspan_t *spans;
	size_t nspans;
	char simple;
	boxf box;
    } htmltxt_t;

    typedef struct {
	boxf box;
	char *src;
	char *scale;
    } htmlimg_t;

    typedef struct {
	bool radial: 1;
	bool rounded: 1;
	bool invisible: 1;
	bool dotted: 1;
	bool dashed: 1;
    } htmlstyle_t;

    typedef struct {
	char *href;		/* pointer to an external resource */
	char *port;
	char *target;
	char *title;
	char *id;
	char *bgcolor;
	char *pencolor;
	int gradientangle;
	signed char space;
	unsigned char border;
	unsigned char pad;
	unsigned char sides;    /* set of sides exposed to field */
	unsigned short flags;
	unsigned short width;
	unsigned short height;
	htmlstyle_t style;
	boxf box;		/* its geometric placement in points */
    } htmldata_t;

typedef enum { HTML_UNSET = 0, HTML_TBL, HTML_TEXT, HTML_IMAGE } label_type_t;

    typedef struct htmlcell_t htmlcell_t;
    typedef struct htmltbl_t htmltbl_t;
	
/* During parsing, table contents are stored as rows of cells.
 * A row is a list of cells
 * Rows is a list of rows.
 */

typedef struct {
  LIST(htmlcell_t *) rp;
  bool ruled;
} row_t;

/// Free row. This closes and frees row’s list, then the item itself is freed.
static inline void free_ritem(row_t *p) {
  LIST_FREE(&p->rp);
  free (p);
}

typedef LIST(row_t *) rows_t;

    struct htmltbl_t {
	htmldata_t data;
	union {
	    struct {
		htmlcell_t *parent;	/* enclosing cell */
		htmlcell_t **cells;	/* cells */
	    };
	    struct {
		htmltbl_t *prev;	/* stack */
		rows_t rows; ///< cells
	    };
	};
	int8_t cellborder;
	double *heights; ///< heights of the rows
	double *widths; ///< widths of the columns
	size_t row_count; ///< number of rows
	size_t column_count; ///< number of columns
	textfont_t *font;	/* font info */
	bool hrule:1; ///< horizontal rule
	bool vrule:1; ///< vertical rule
    };

    struct htmllabel_t {
	union {
	    htmltbl_t *tbl;
	    htmltxt_t *txt;
	    htmlimg_t *img;
	} u;
	label_type_t kind;
    };

    struct htmlcell_t {
	htmldata_t data;
	uint16_t colspan;
	uint16_t rowspan;
	uint16_t col;
	uint16_t row;
	htmllabel_t child;
	htmltbl_t *parent;
	bool vruled: 1; ///< vertically ruled?
	bool hruled: 1; ///< horizontally ruled?
    };

    typedef struct {
        pointf pos;
        textfont_t finfo;
        void *obj;
        graph_t *g;
        char *imgscale;
        char *objid;
        bool objid_set;
    } htmlenv_t;

    extern htmllabel_t *parseHTML(char *, int *, htmlenv_t *);

    extern int make_html_label(void *obj, textlabel_t * lp);
    extern void emit_html_label(GVJ_t * job, htmllabel_t * lp, textlabel_t *);

    extern void free_html_label(htmllabel_t *, int);
    extern void free_html_data(htmldata_t *);
    extern void free_html_text(htmltxt_t *);

    extern boxf *html_port(node_t *n, char *pname, unsigned char *sides);

#ifdef __cplusplus
}
#endif
