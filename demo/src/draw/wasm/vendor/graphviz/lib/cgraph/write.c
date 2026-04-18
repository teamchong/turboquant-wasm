/**
 * @file
 * @brief implements @ref agwrite, @ref agstrcanon,
 * and @ref agcanonStr
 *
 * @ingroup cgraph_graph
 * @ingroup cgraph_core
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

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>		/* need sprintf() */
#include <stdlib.h>
#include <ctype.h>
#include <cgraph/agstrcanon.h>
#include <cgraph/cghdr.h>
#include <inttypes.h>
#include <util/gv_ctype.h>
#include <util/strcasecmp.h>

#define EMPTY(s)		(((s) == 0) || (s)[0] == '\0')
#define CHKRV(v)     {if ((v) == EOF) return EOF;}

typedef void iochan_t;

static int ioput(Agraph_t * g, iochan_t * ofile, char *str)
{
    return AGDISC(g, io)->putstr(ofile, str);

}

#define MAX_OUTPUTLINE		128
#define MIN_OUTPUTLINE		 60
static int Max_outputline = MAX_OUTPUTLINE;
static Agsym_t *Tailport, *Headport;

typedef struct {
	uint64_t *preorder_number;	// of a graph or subgraph
	uint64_t *node_last_written;	// postorder number of subg when node was last written
	uint64_t *edge_last_written;	// postorder number of subg when edge was last written
	int level;                      // indentation level
} write_info_t;

static int write_body(Agraph_t *g, iochan_t *ofile, write_info_t *wr_info);

static write_info_t before_write(Agraph_t *);
static void after_write(write_info_t);

static int indent(Agraph_t *g, iochan_t *ofile, const write_info_t wr_info) {
    int i;
    for (i = wr_info.level; i > 0; i--)
	CHKRV(ioput(g, ofile, "\t"));
    return 0;
}

// alphanumeric, '.', '-', or non-ascii; basically, chars used in unquoted ids
static bool is_id_char(char c) {
  return gv_isalnum(c) || c == '.' || c == '-' || !isascii(c);
}

// is the prefix of this string a recognized Graphviz escape sequence?
// https://graphviz.org/docs/attr-types/escString/
static bool is_escape(const char *str) {
  assert(str != NULL);

  if (*str != '\\')
    return false;

  if (str[1] == 'E')
    return true;
  if (str[1] == 'G')
    return true;
  if (str[1] == 'H')
    return true;
  if (str[1] == 'L')
    return true;
  if (str[1] == 'N')
    return true;
  if (str[1] == 'T')
    return true;

  if (str[1] == 'l')
    return true;
  if (str[1] == 'n')
    return true;
  if (str[1] == 'r')
    return true;

  if (str[1] == '\\')
    return true;

  if (str[1] == '"')
    return true;

  return false;
}

/* Canonicalize ordinary strings. 
 * Assumes buf is large enough to hold output.
 */
static char *_agstrcanon(char *arg, char *buf)
{
    char *s, *p;
    char uc;
    int cnt = 0, dotcnt = 0;
    bool needs_quotes = false;
    bool part_of_escape = false;
    bool maybe_num;
    bool backslash_pending = false;
    static const char *tokenlist[]	/* must agree with scan.l */
	= { "node", "edge", "strict", "graph", "digraph", "subgraph",
	NULL
    };
    const char **tok;

    if (EMPTY(arg))
	return "\"\"";
    s = arg;
    p = buf;
    *p++ = '\"';
    uc = *s++;
    maybe_num = gv_isdigit(uc) || uc == '.' || uc == '-';
    while (uc) {
	if (uc == '\"' && !part_of_escape) {
	    *p++ = '\\';
	    needs_quotes = true;
	} else if (!part_of_escape && is_escape(&s[-1])) {
	    needs_quotes = true;
	    part_of_escape = true;
	} else if (maybe_num) {
	    if (uc == '-') {
		if (cnt) {
		    maybe_num = false;
		    needs_quotes = true;
		}
	    }
	    else if (uc == '.') {
		if (dotcnt++) {
		    maybe_num = false;
		    needs_quotes = true;
		}
	    }
	    else if (!gv_isdigit(uc)) {
		maybe_num = false;
		needs_quotes = true;
	    }
	    part_of_escape = false;
	}
	else if (!(gv_isalnum(uc) || uc == '_' || !isascii(uc))) {
	    needs_quotes = true;
	    part_of_escape = false;
	} else {
	    part_of_escape = false;
	}
	*p++ = uc;
	uc = *s++;
	cnt++;
	
	/* If breaking long strings into multiple lines, only allow breaks after a non-id char, not a backslash, where the next char is an
 	 * id char.
	 */
	if (Max_outputline) {
            if (uc && backslash_pending && !(is_id_char(p[-1]) || p[-1] == '\\') && is_id_char(uc)) {
        	*p++ = '\\';
        	*p++ = '\n';
        	needs_quotes = true;
        	backslash_pending = false;
		cnt = 0;
            } else if (uc && (cnt >= Max_outputline)) {
        	if (!(is_id_char(p[-1]) || p[-1] == '\\') && is_id_char(uc)) {
	            *p++ = '\\';
    	            *p++ = '\n';
	            needs_quotes = true;
		    cnt = 0;
        	} else {
                    backslash_pending = true;
        	}
	    }
	}
    }
    *p++ = '\"';
    *p = '\0';
    if (needs_quotes || (cnt == 1 && (*arg == '.' || *arg == '-')))
	return buf;

    /* Use quotes to protect tokens (example, a node named "node") */
    /* It would be great if it were easier to use flex here. */
    for (tok = tokenlist; *tok; tok++)
	if (!strcasecmp(*tok, arg))
	    return buf;
    return arg;
}

/**
 * Canonicalize html strings. 
 */
static char *agcanonhtmlstr(const char *arg, char *buf)
{
    sprintf(buf, "<%s>", arg);
    return buf;
}

/**
 * canonicalize a string for printing.
 * Unsafe if buffer is not large enough.
 */
char *agstrcanon(char *arg, char *buf)
{
    if (aghtmlstr(arg))
	return agcanonhtmlstr(arg, buf);
    else
	return _agstrcanon(arg, buf);
}

static int _write_canonstr(Agraph_t *g, iochan_t *ofile, char *str, bool chk) {

    // maximum bytes required for canonicalized string
    const size_t required = agstrcanon_bytes(str);

    // allocate space to stage the canonicalized string
    char *const scratch = malloc(required);
    if (scratch == NULL) {
	return EOF;
    }

    char *const canonicalized =
      chk ? agstrcanon(str, scratch) : _agstrcanon(str, scratch);
    const int rc = ioput(g, ofile, canonicalized);
    free(scratch);
    return rc;
}

/// @param known Is `str` already known to be a reference-counted string?
static int write_canonstr(Agraph_t *g, iochan_t *ofile, char *str, bool known) {
    char *s;

    /* str may not have been allocated by agstrdup, so we first need to turn it
     * into a valid refstr
     */
    s = known ? str : agstrdup(g, str);

    int r = _write_canonstr(g, ofile, s, true);

    if (!known) {
	agstrfree(g, s, false);
    }
    return r;
}

static int write_dict(Agraph_t * g, iochan_t * ofile, char *name,
                      Dict_t * dict, bool top, write_info_t *wr_info) {
    int cnt = 0;
    Dict_t *view;
    Agsym_t *sym, *psym;

    if (!top)
	view = dtview(dict, NULL);
    else
	view = 0;
    for (sym = dtfirst(dict); sym; sym = dtnext(dict, sym)) {
	if (EMPTY(sym->defval) && !sym->print) {	/* try to skip empty str (default) */
	    if (view == NULL)
		continue;	/* no parent */
	    psym = dtsearch(view, sym);
	    assert(psym);
	    if (EMPTY(psym->defval) && psym->print)
		continue;	/* also empty in parent */
	}
	if (cnt++ == 0) {
	    CHKRV(indent(g, ofile, *wr_info));
	    CHKRV(ioput(g, ofile, name));
	    CHKRV(ioput(g, ofile, " ["));
	    wr_info->level++;
	} else {
	    CHKRV(ioput(g, ofile, ",\n"));
	    CHKRV(indent(g, ofile, *wr_info));
	}
	CHKRV(write_canonstr(g, ofile, sym->name, true));
	CHKRV(ioput(g, ofile, "="));
	CHKRV(write_canonstr(g, ofile, sym->defval, true));
    }
    if (cnt > 0) {
	wr_info->level--;
	if (cnt > 1) {
	    CHKRV(ioput(g, ofile, "\n"));
	    CHKRV(indent(g, ofile, *wr_info));
	}
	CHKRV(ioput(g, ofile, "];\n"));
    }
    if (!top)
	dtview(dict, view);	/* restore previous view */
    return 0;
}

static int write_dicts(Agraph_t *g, iochan_t *ofile, bool top,
                       write_info_t *wr_info) {
    Agdatadict_t *def;
    if ((def = agdatadict(g, false))) {
	CHKRV(write_dict(g, ofile, "graph", def->dict.g, top, wr_info));
	CHKRV(write_dict(g, ofile, "node", def->dict.n, top, wr_info));
	CHKRV(write_dict(g, ofile, "edge", def->dict.e, top, wr_info));
    }
    return 0;
}

static int write_hdr(Agraph_t *g, iochan_t *ofile, bool top,
                     write_info_t *wr_info) {
    char *name, *sep, *kind, *strict;
    bool root = false;
    bool hasName = true;

    strict = "";
    if (!top && agparent(g))
	kind = "sub";
    else {
	root = true;
	if (g->desc.directed)
	    kind = "di";
	else
	    kind = "";
	if (agisstrict(g))
	    strict = "strict ";
	Tailport = agattr_text(g, AGEDGE, TAILPORT_ID, NULL);
	Headport = agattr_text(g, AGEDGE, HEADPORT_ID, NULL);
    }
    name = agnameof(g);
    sep = " ";
    if (!name || name[0] == LOCALNAMEPREFIX) {
	sep = name = "";
	hasName = false;
    }
    CHKRV(indent(g, ofile, *wr_info));
    CHKRV(ioput(g, ofile, strict));

    /* output "<kind>graph" only for root graphs or graphs with names */
    if (root || hasName) {
	CHKRV(ioput(g, ofile, kind));
	CHKRV(ioput(g, ofile, "graph "));
    }
    if (hasName)
	CHKRV(write_canonstr(g, ofile, name, false));
    CHKRV(ioput(g, ofile, sep));
    CHKRV(ioput(g, ofile, "{\n"));
    wr_info->level++;
    CHKRV(write_dicts(g, ofile, top, wr_info));
    AGATTRWF(g) = true;
    return 0;
}

static int write_trl(Agraph_t *g, iochan_t *ofile, write_info_t *wr_info) {
    wr_info->level--;
    CHKRV(indent(g, ofile, *wr_info));
    CHKRV(ioput(g, ofile, "}\n"));
    return 0;
}

/// is this graph unnamed?
///
/// @param g Graph to inspect
/// @return True if this graph was given no explicit name
static bool is_anonymous(Agraph_t *g) {
  assert(g != NULL);

  // handle the common case inline for performance
  if (AGDISC(g, id) == &AgIdDisc) {
    // replicate `idprint`
    const IDTYPE id = AGID(g);
    if (id % 2 != 0) {
      return true;
    }
    return *(char *)(uintptr_t)id == LOCALNAMEPREFIX;
  }

  const char *const name = agnameof(g);
  return name == NULL || name[0] == LOCALNAMEPREFIX;
}

static bool irrelevant_subgraph(Agraph_t * g)
{
    int i, n;
    Agattr_t *sdata, *pdata, *rdata;
    Agdatadict_t *dd;

    if (!is_anonymous(g))
	return false;
    if ((sdata = agattrrec(g)) && (pdata = agattrrec(agparent(g)))) {
	rdata = agattrrec(agroot(g));
	n = dtsize(rdata->dict);
	for (i = 0; i < n; i++)
	    if (sdata->str[i] && pdata->str[i]
		&& strcmp(sdata->str[i], pdata->str[i]))
		return false;
    }
    dd = agdatadict(g, false);
    if (!dd)
	return true;
    if (dtsize(dd->dict.n) > 0 || dtsize(dd->dict.e) > 0)
	return false;
    return true;
}

static bool has_no_edges(Agraph_t * g, Agnode_t * n)
{
    return agfstin(g, n) == NULL && agfstout(g, n) == NULL;
}

static bool not_default_attrs(Agraph_t * g, Agnode_t * n)
{
    Agattr_t *data;
    Agsym_t *sym;

    (void)g;
    if ((data = agattrrec(n))) {
	for (sym = dtfirst(data->dict); sym; sym = dtnext(data->dict, sym)) {
	    if (data->str[sym->id] != sym->defval)
		return true;
	}
    }
    return false;
}

static int write_subgs(Agraph_t *g, iochan_t *ofile, write_info_t *wr_info) {
    Agraph_t *subg;

    for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	if (irrelevant_subgraph(subg)) {
	    write_subgs(subg, ofile, wr_info);
	}
	else {
	    CHKRV(write_hdr(subg, ofile, false, wr_info));
	    CHKRV(write_body(subg, ofile, wr_info));
	    CHKRV(write_trl(subg, ofile, wr_info));
	}
    }
    return 0;
}

static int write_edge_name(Agedge_t *e, iochan_t *ofile, bool terminate,
                           write_info_t *wr_info) {
    char *p;
    Agraph_t *g;

    p = agnameof(e);
    g = agraphof(e);
    if (!EMPTY(p)) {
	if (!terminate) {
	    wr_info->level++;
	}
	CHKRV(ioput(g, ofile, "\t[key="));
	CHKRV(write_canonstr(g, ofile, p, false));
	if (terminate)
	    CHKRV(ioput(g, ofile, "]"));
	return 1;
    }
    return 0;
}


static int write_nondefault_attrs(void *obj, iochan_t * ofile,
				  Dict_t *defdict, write_info_t *wr_info) {
    Agattr_t *data;
    Agsym_t *sym;
    Agraph_t *g;
    int cnt = 0;
    int rv;

    if (AGTYPE(obj) == AGINEDGE || AGTYPE(obj) == AGOUTEDGE) {
	CHKRV(rv = write_edge_name(obj, ofile, false, wr_info));
	if (rv)
	    cnt++;
    }
    data = agattrrec(obj);
    g = agraphof(obj);
    if (data)
	for (sym = dtfirst(defdict); sym; sym = dtnext(defdict, sym)) {
	    if (AGTYPE(obj) == AGINEDGE || AGTYPE(obj) == AGOUTEDGE) {
		if (Tailport && sym->id == Tailport->id)
		    continue;
		if (Headport && sym->id == Headport->id)
		    continue;
	    }
	    if (data->str[sym->id] != sym->defval) {
		if (cnt++ == 0) {
		    CHKRV(ioput(g, ofile, "\t["));
		    wr_info->level++;
		} else {
		    CHKRV(ioput(g, ofile, ",\n"));
		    CHKRV(indent(g, ofile, *wr_info));
		}
		CHKRV(write_canonstr(g, ofile, sym->name, true));
		CHKRV(ioput(g, ofile, "="));
		CHKRV(write_canonstr(g, ofile, data->str[sym->id], true));
	    }
	}
    if (cnt > 0) {
	CHKRV(ioput(g, ofile, "]"));
	wr_info->level--;
    }
    AGATTRWF(obj) = true;
    return 0;
}

static int write_nodename(Agnode_t * n, iochan_t * ofile)
{
    char *name;
    Agraph_t *g;

    name = agnameof(n);
    g = agraphof(n);
    if (name) {
	CHKRV(write_canonstr(g, ofile, name, false));
    } else {
	char buf[sizeof("__SUSPECT") + 20];
	snprintf(buf, sizeof(buf), "_%" PRIu64 "_SUSPECT", AGID(n));	/* could be deadly wrong */
	CHKRV(ioput(g, ofile, buf));
    }
    return 0;
}

static int attrs_written(void *obj)
{
    return AGATTRWF(obj);
}

static int write_node(Agraph_t *subg, Agnode_t *n, iochan_t *ofile, Dict_t *d,
                      write_info_t *wr_info) {
    Agraph_t *g;

    g = agraphof(n);
    CHKRV(indent(g, ofile, *wr_info));
    CHKRV(write_nodename(n, ofile));
    if (!attrs_written(n))
	CHKRV(write_nondefault_attrs(n, ofile, d, wr_info));
    wr_info->node_last_written[AGSEQ(n)] =
	wr_info->preorder_number[AGSEQ(subg)];
    return ioput(g, ofile, ";\n");
}

/* node must be written if it wasn't already emitted because of
 * a subgraph or one of its predecessors, and if it is a singleton
 * or has non-default attributes.
 */
static bool write_node_test(Agraph_t *g, Agnode_t *n, write_info_t *wr_info) {
    /* test if node was already written in g or a subgraph of g */
    if (wr_info->node_last_written[AGSEQ(n)] >=
	wr_info->preorder_number[AGSEQ(g)]) return false;

    if (has_no_edges(g, n) || not_default_attrs(g, n))
	return true;
    return false;
}

static int write_port(Agedge_t * e, iochan_t * ofile, Agsym_t * port)
{
    char *val;
    Agraph_t *g;

    if (!port)
	return 0;
    g = agraphof(e);
    val = agxget(e, port);
    if (val[0] == '\0')
	return 0;

    CHKRV(ioput(g, ofile, ":"));
    if (aghtmlstr(val)) {
	CHKRV(write_canonstr(g, ofile, val, true));
    } else {
	char *s = strchr(val, ':');
	if (s) {
	    *s = '\0';
	    CHKRV(_write_canonstr(g, ofile, val, false));
	    CHKRV(ioput(g, ofile, ":"));
	    CHKRV(_write_canonstr(g, ofile, s + 1, false));
	    *s = ':';
	} else {
	    CHKRV(_write_canonstr(g, ofile, val, false));
	}
    }
    return 0;
}

static bool write_edge_test(Agraph_t *g, Agedge_t *e, write_info_t *wr_info) {
    if (wr_info->edge_last_written[AGSEQ(e)] >=
        wr_info->preorder_number[AGSEQ(g)]) return false;
    return true;
}

static int write_edge(Agraph_t *subg, Agedge_t *e, iochan_t *ofile, Dict_t *d,
                      write_info_t *wr_info) {
    Agnode_t *t, *h;
    Agraph_t *g;

    t = AGTAIL(e);
    h = AGHEAD(e);
    g = agraphof(t);
    CHKRV(indent(g, ofile, *wr_info));
    CHKRV(write_nodename(t, ofile));
    CHKRV(write_port(e, ofile, Tailport));
    CHKRV(ioput(g, ofile, (agisdirected(agraphof(t)) ? " -> " : " -- ")));
    CHKRV(write_nodename(h, ofile));
    CHKRV(write_port(e, ofile, Headport));
    if (!attrs_written(e)) {
	CHKRV(write_nondefault_attrs(e, ofile, d, wr_info));
    } else {
	CHKRV(write_edge_name(e, ofile, true, wr_info));
    }
    wr_info->edge_last_written[AGSEQ(e)] =
	wr_info->preorder_number[AGSEQ(subg)];
    return ioput(g, ofile, ";\n");
}

static int write_body(Agraph_t *g, iochan_t *ofile, write_info_t *wr_info) {
    Agnode_t *n, *prev;
    Agedge_t *e;
    Agdatadict_t *dd;

    CHKRV(write_subgs(g, ofile, wr_info));
    dd = agdatadict(g, false);
    for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	if (write_node_test(g, n, wr_info))
	    CHKRV(write_node(g, n, ofile, dd ? dd->dict.n : 0, wr_info));
	prev = n;
	for (e = agfstout(g, n); e; e = agnxtout(g, e)) {
	    if (prev != aghead(e) && write_node_test(g, aghead(e), wr_info)) {
		CHKRV(write_node(g, aghead(e), ofile, dd ? dd->dict.n : 0, wr_info));
		prev = aghead(e);
	    }
	    if (write_edge_test(g, e, wr_info))
		CHKRV(write_edge(g, e, ofile, dd ? dd->dict.e : 0, wr_info));
	}

	}
    return 0;
}

static void set_attrwf(Agraph_t * g, bool toplevel, bool value)
{
    Agraph_t *subg;
    Agnode_t *n;
    Agedge_t *e;

    AGATTRWF(g) = value;
    for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	set_attrwf(subg, false, value);
    }
    if (toplevel) {
	for (n = agfstnode(g); n; n = agnxtnode(g, n)) {
	    AGATTRWF(n) = value;
	    for (e = agfstout(g, n); e; e = agnxtout(g, e))
		AGATTRWF(e) = value;
	}
    }
}

/// Return 0 on success, EOF on failure
int agwrite(Agraph_t * g, void *ofile)
{
    char* s;
    s = agget(g, "linelength");
    if (s != NULL && gv_isdigit(*s)) {
	unsigned long len = strtoul(s, NULL, 10);
	if ((len == 0 || len >= MIN_OUTPUTLINE) && len <= INT_MAX)
	    Max_outputline = (int)len;
    }
    write_info_t wr_info = before_write(g);
    if (write_hdr(g, ofile, true, &wr_info) == EOF) {
	after_write(wr_info);
	return EOF;
    }
    if (write_body(g, ofile, &wr_info) == EOF) {
	after_write(wr_info);
	return EOF;
    }
    if (write_trl(g, ofile, &wr_info) == EOF) {
	after_write(wr_info);
	return EOF;
    }
    after_write(wr_info);
    Max_outputline = MAX_OUTPUTLINE;
    return AGDISC(g, io)->flush(ofile);
}

static uint64_t subgdfs(Agraph_t *g, uint64_t ix, write_info_t *wr_info) {
    uint64_t ix0 = ix;
    Agraph_t *subg;

    wr_info->preorder_number[AGSEQ(g)] = ix0;
    for (subg = agfstsubg(g); subg; subg = agnxtsubg(subg)) {
	ix0 = subgdfs(subg, ix0, wr_info);
    }
    return ix0 + 1;
}

static write_info_t before_write(Agraph_t *g) {
    write_info_t wr_info = {0};
    set_attrwf(g, true, false);
    
    wr_info.preorder_number = gv_calloc(g->clos->seq[AGRAPH] + 1, sizeof(uint64_t));
    wr_info.node_last_written = gv_calloc(g->clos->seq[AGNODE] + 1, sizeof(uint64_t));
    wr_info.edge_last_written = gv_calloc(g->clos->seq[AGEDGE] + 1, sizeof(uint64_t));
    subgdfs(g, 1, &wr_info);
    return wr_info;
}

static void after_write(write_info_t wr_info) {
  free(wr_info.preorder_number);
  free(wr_info.node_last_written);
  free(wr_info.edge_last_written);
}
