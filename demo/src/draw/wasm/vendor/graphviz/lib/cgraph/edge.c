/**
 * @file
 * @ingroup cgraph_edge
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
#include <cgraph/cghdr.h>
#include <cgraph/node_set.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/unused.h>

/* return first outedge of <n> */
Agedge_t *agfstout(Agraph_t * g, Agnode_t * n)
{
    Agsubnode_t *sn;
    Agedge_t *e = NULL;

    sn = agsubrep(g, n);
    if (sn) {
		dtrestore(g->e_seq, sn->out_seq);
		e = dtfirst(g->e_seq);
		sn->out_seq = dtextract(g->e_seq);
	}
    return e;
}

/* return outedge that follows <e> of <n> */
Agedge_t *agnxtout(Agraph_t * g, Agedge_t * e)
{
    Agnode_t *n;
    Agsubnode_t *sn;
    Agedge_t *f = NULL;

    n = AGTAIL(e);
    sn = agsubrep(g, n);
    if (sn) {
		dtrestore(g->e_seq, sn->out_seq);
		f = dtnext(g->e_seq, e);
		sn->out_seq = dtextract(g->e_seq);
	}
    return f;
}

Agedge_t *agfstin(Agraph_t * g, Agnode_t * n)
{
    Agsubnode_t *sn;
    Agedge_t *e = NULL;

    sn = agsubrep(g, n);
	if (sn) {
		dtrestore(g->e_seq, sn->in_seq);
		e = dtfirst(g->e_seq);
		sn->in_seq = dtextract(g->e_seq);
	}
    return e;
}

Agedge_t *agnxtin(Agraph_t * g, Agedge_t * e)
{
    Agnode_t *n;
    Agsubnode_t *sn;
    Agedge_t *f = NULL;

    n = AGHEAD(e);
    sn = agsubrep(g, n);
	if (sn) {
		dtrestore(g->e_seq, sn->in_seq);
		f = dtnext(g->e_seq, e);
		sn->in_seq = dtextract(g->e_seq);
	}
	return f;
}

Agedge_t *agfstedge(Agraph_t * g, Agnode_t * n)
{
    Agedge_t *rv;
    rv = agfstout(g, n);
    if (rv == NULL)
	rv = agfstin(g, n);
    return rv;
}

Agedge_t *agnxtedge(Agraph_t * g, Agedge_t * e, Agnode_t * n)
{
    Agedge_t *rv;

    if (AGTYPE(e) == AGOUTEDGE) {
	rv = agnxtout(g, e);
	if (rv == NULL) {
	    do {
		rv = !rv ? agfstin(g, n) : agnxtin(g,rv);
	    } while (rv && rv->node == n);
	}
    } else {
	do {
	    rv = agnxtin(g, e);		/* so that we only see each edge once, */
		e = rv;
	} while (rv && rv->node == n);	/* ignore loops as in-edges */
    }
    return rv;
}

/* internal edge set lookup */
static Agedge_t *agfindedge_by_key(Agraph_t * g, Agnode_t * t, Agnode_t * h,
			    Agtag_t key)
{
    Agedge_t *e, template;
    Agsubnode_t *sn;

    if (t == NULL || h == NULL)
	return NULL;
    template.base.tag = key;
    template.node = t;		/* guess that fan-in < fan-out */
    sn = agsubrep(g, h);
    if (!sn) e = 0;
    else {
	    dtrestore(g->e_id, sn->in_id);
	    e = dtsearch(g->e_id, &template);
	    sn->in_id = dtextract(g->e_id);
    }
    return e;
}

static Agedge_t *agfindedge_by_id(Agraph_t * g, Agnode_t * t, Agnode_t * h,
                  IDTYPE id)
{
    Agtag_t tag = {.objtype = AGEDGE, .id = id};
    return agfindedge_by_key(g, t, h, tag);
}

Agsubnode_t *agsubrep(Agraph_t * g, Agnode_t * n)
{
  return g == n->root ? &n->mainsub : node_set_find(g->n_id, n->base.tag.id);
}

static void ins(Dict_t * d, Dtlink_t ** set, Agedge_t * e)
{
    dtrestore(d, *set);
    dtinsert(d, e);
    *set = dtextract(d);
}

static void del(Dict_t * d, Dtlink_t ** set, Agedge_t * e)
{
    dtrestore(d, *set);
    void *x UNUSED = dtdelete(d, e);
    assert(x);
    *set = dtextract(d);
}

static void installedge(Agraph_t * g, Agedge_t * e)
{
    Agnode_t *t, *h;
    Agedge_t *out, *in;
    Agsubnode_t *sn;

    out = AGMKOUT(e);
    in = AGMKIN(e);
    t = agtail(e);
    h = aghead(e);
    while (g) {
	if (agfindedge_by_key(g, t, h, AGTAG(e))) break;
	sn = agsubrep(g, t);
	ins(g->e_seq, &sn->out_seq, out);
	ins(g->e_id, &sn->out_id, out);
	sn = agsubrep(g, h);
	ins(g->e_seq, &sn->in_seq, in);
	ins(g->e_id, &sn->in_id, in);
	g = agparent(g);
    }
}

static void subedge(Agraph_t * g, Agedge_t * e)
{
    installedge(g, e);
    /* might an init method call be needed here? */
}

static Agedge_t *newedge(Agraph_t * g, Agnode_t * t, Agnode_t * h,
             IDTYPE id)
{
    Agedge_t *in, *out;

    (void)agsubnode(g, t, 1);
    (void)agsubnode(g, h, 1);
    Agedgepair_t *e2 = gv_alloc(sizeof(Agedgepair_t));
    in = &e2->in;
    out = &e2->out;
    uint64_t seq = agnextseq(g, AGEDGE);
    assert((seq & SEQ_MASK) == seq && "sequence ID overflow");
    AGTYPE(in) = AGINEDGE;
    AGTYPE(out) = AGOUTEDGE;
    AGID(in) = AGID(out) = id;
    AGSEQ(in) = AGSEQ(out) = seq & SEQ_MASK;
    in->node = t;
    out->node = h;

    installedge(g, out);
    if (g->desc.has_attrs) {
	(void)agbindrec(out, AgDataRecName, sizeof(Agattr_t), false);
	agedgeattr_init(g, out);
    }
    agmethod_init(g, out);
    return out;
}

/* edge creation predicate */
static bool ok_to_make_edge(Agraph_t *g, Agnode_t *t, Agnode_t *h) {
    Agtag_t key = {0};

    /* protect against self, multi-edges in strict graphs */
    if (agisstrict(g)) {
	key.objtype = 0;	/* wild card */
	if (agfindedge_by_key(g, t, h, key))
	    return false;
    }
    if (g->desc.no_loop && (t == h)) /* simple graphs */
	return false;
    return true;
}

Agedge_t *agidedge(Agraph_t * g, Agnode_t * t, Agnode_t * h,
           IDTYPE id, int cflag)
{
    Agraph_t *root;
    Agedge_t *e;

    e = agfindedge_by_id(g, t, h, id);
    if (e == NULL && agisundirected(g))
	e = agfindedge_by_id(g, h, t, id);
    if (e == NULL && cflag && ok_to_make_edge(g, t, h)) {
	root = agroot(g);
	if (g != root && (e = agfindedge_by_id(root, t, h, id))) {
	    subedge(g, e);	/* old */
	}
    }
    return e;
}

Agedge_t *agedge(Agraph_t * g, Agnode_t * t, Agnode_t * h, char *name,
		 int cflag)
{
    Agedge_t *e;
    IDTYPE my_id;

    int have_id = agmapnametoid(g, AGEDGE, name, &my_id, false);
    if (have_id || (name == NULL && (!cflag || agisstrict(g)))) {
	/* probe for pre-existing edge */
	Agtag_t key = {0};
	if (have_id) {
	    key.id = my_id;
	    key.objtype = AGEDGE;
	} else {
	    key.id = key.objtype = 0;
	}

	/* might already exist locally */
	e = agfindedge_by_key(g, t, h, key);
	if (e == NULL && agisundirected(g))
	    e = agfindedge_by_key(g, h, t, key);
	if (e)
	    return e;
	if (cflag) {
	    e = agfindedge_by_key(agroot(g), t, h, key);
	    if (e == NULL && agisundirected(g))
		e = agfindedge_by_key(agroot(g), h, t, key);
	    if (e) {
		subedge(g,e);
		return e;
	    }
	}
    }

    if (cflag && ok_to_make_edge(g, t, h)
	&& agmapnametoid(g, AGEDGE, name, &my_id, true)) { /* reserve id */
	e = newedge(g, t, h, my_id);
	agregister(g, AGEDGE, e); /* register new object in external namespace */
    }
    else
	e = NULL;
    return e;
}

void agdeledgeimage(Agraph_t *g, Agobj_t *edge, void *ignored) {
    Agedge_t *e = (Agedge_t *)((char *)edge - offsetof(Agedge_t, base));
    Agedge_t *in, *out;
    Agnode_t *t, *h;
    Agsubnode_t *sn;

    (void)ignored;
    if (AGTYPE(e) == AGINEDGE) {
	in = e;
	out = AGIN2OUT(e);
    } else {
	out = e;
	in = AGOUT2IN(e);
    }
    t = in->node;
    h = out->node;
    sn = agsubrep(g, t);
    del(g->e_seq, &sn->out_seq, out);
    del(g->e_id, &sn->out_id, out);
    sn = agsubrep(g, h);
    del(g->e_seq, &sn->in_seq, in);
    del(g->e_id, &sn->in_id, in);
#ifdef DEBUG
    for (e = agfstin(g,h); e; e = agnxtin(g,e))
	assert(e != in);
    for (e = agfstout(g,t); e; e = agnxtout(g,e))
	assert(e != out);
#endif
}

int agdeledge(Agraph_t * g, Agedge_t * e)
{
    e = AGMKOUT(e);
    if (agfindedge_by_key(g, agtail(e), aghead(e), AGTAG(e)) == NULL)
	return FAILURE;

    if (g == agroot(g)) {
	if (g->desc.has_attrs)
	    agedgeattr_delete(e);
	agmethod_delete(g, e);
	agrecclose(&e->base);
	agfreeid(g, AGEDGE, AGID(e));
    }
    if (agapply(g, &e->base, agdeledgeimage, NULL, false) == SUCCESS) {
	if (g == agroot(g))
		free(e);
	return SUCCESS;
    } else
	return FAILURE;
}

Agedge_t *agsubedge(Agraph_t * g, Agedge_t * e, int cflag)
{
    Agnode_t *t;
    Agedge_t *rv;

    rv = NULL;
    t = agsubnode(g, AGTAIL(e), cflag);
    if (t != NULL || cflag) {
	Agnode_t *const h = agsubnode(g, AGHEAD(e), cflag);
	if (t != NULL && h != NULL) {
	    rv = agfindedge_by_key(g, t, h, AGTAG(e));
	    if (cflag && rv == NULL) {
	        installedge(g, e);
	        rv = e;
	    }
	    if (rv && AGTYPE(rv) != AGTYPE(e))
	        rv = AGOPP(rv);
	}
    }
    return rv;
}

/* edge comparison.  AGTYPE(e) == 0 means ID is a wildcard. */
static int agedgeidcmpf(void *arg_e0, void *arg_e1) {
    Agedge_t *e0 = arg_e0;
    Agedge_t *e1 = arg_e1;

    if (AGID(e0->node) < AGID(e1->node)) return -1;
    if (AGID(e0->node) > AGID(e1->node)) return 1;
    /* same node */
    if (AGTYPE(e0) != 0 && AGTYPE(e1) != 0) {
        if (AGID(e0) < AGID(e1)) return -1;
        if (AGID(e0) > AGID(e1)) return 1;
    }
    return 0;
}

/* edge comparison.  for ordered traversal. */
static int agedgeseqcmpf(void *arg_e0, void *arg_e1) {
    Agedge_t *e0 = arg_e0;
    Agedge_t *e1 = arg_e1;
    assert(arg_e0 && arg_e1);

    if (e0->node != e1->node) {
        if (AGSEQ(e0->node) < AGSEQ(e1->node)) return -1;
        if (AGSEQ(e0->node) > AGSEQ(e1->node)) return 1;
    }
    else {
        if (AGSEQ(e0) < AGSEQ(e1)) return -1;
        if (AGSEQ(e0) > AGSEQ(e1)) return 1;
    }
    return 0;
}

/* indexing for ordered traversal */
Dtdisc_t Ag_mainedge_seq_disc = {
    .link = offsetof(Agedge_t, seq_link), // use internal links
    .comparf = agedgeseqcmpf,
};

Dtdisc_t Ag_subedge_seq_disc = {
    .link = -1, // use external holder objects
    .comparf = agedgeseqcmpf,
};

/* indexing for random search */
Dtdisc_t Ag_mainedge_id_disc = {
    .link = offsetof(Agedge_t, id_link), // use internal links
    .comparf = agedgeidcmpf,
};

Dtdisc_t Ag_subedge_id_disc = {
    .link = -1, // use external holder objects
    .comparf = agedgeidcmpf,
};

/* expose macros as functions for ease of debugging
and to expose them to foreign languages without C preprocessor. */
#ifdef ageqedge
#undef ageqedge
#endif
CGRAPH_API int ageqedge(Agedge_t *e, Agedge_t *f);
CGRAPH_API int ageqedge(Agedge_t * e, Agedge_t * f)
{
    return AGEQEDGE(e, f);
}

#ifdef agmkout
#undef agmkout
#endif
CGRAPH_API Agedge_t *agmkout(Agedge_t *e);
CGRAPH_API Agedge_t *agmkout(Agedge_t * e)
{
    return AGMKOUT(e);
}

#ifdef agmkin
#undef agmkin
#endif
CGRAPH_API Agedge_t *agmkin(Agedge_t *e);
CGRAPH_API Agedge_t *agmkin(Agedge_t * e)
{
    return AGMKIN(e);
}

#ifdef agtail
#undef agtail
#endif
CGRAPH_API Agnode_t *agtail(Agedge_t *e);
CGRAPH_API Agnode_t *agtail(Agedge_t * e)
{
    return AGTAIL(e);
}

#ifdef aghead
#undef aghead
#endif
CGRAPH_API Agnode_t *aghead(Agedge_t *e);
CGRAPH_API Agnode_t *aghead(Agedge_t * e)
{
    return AGHEAD(e);
}

#ifdef agopp
#undef agopp
#endif
CGRAPH_API Agedge_t *agopp(Agedge_t *e);
CGRAPH_API Agedge_t *agopp(Agedge_t * e)
{
    return AGOPP(e);
}
