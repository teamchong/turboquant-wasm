/// @file
/// @brief generic manipulations with @ref Agobj_t and derived objects graphs, nodes and edges
/// @ingroup cgraph_object

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

#include <cgraph/cghdr.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/unreachable.h>

int agdelete(Agraph_t * g, void *obj)
{
    if (AGTYPE(obj) == AGRAPH && g != agparent(obj)) {
	agerrorf("agdelete on wrong graph");
	return FAILURE;
    }

    switch (AGTYPE(obj)) {
    case AGNODE:
	return agdelnode(g, obj);
    case AGINEDGE:
    case AGOUTEDGE:
	return agdeledge(g, obj);
    case AGRAPH:
	return agclose(obj);
    default:
	agerrorf("agdelete on bad object");
    }
    return SUCCESS;		/* not reached */
}

int agrename(Agobj_t * obj, char *newname)
{
    Agraph_t *g;
    IDTYPE old_id, new_id;

    switch (AGTYPE(obj)) {
    case AGRAPH:
	old_id = AGID(obj);
	g = agraphof(obj);
	/* can we reserve the id corresponding to newname? */
	if (agmapnametoid(agroot(g), AGTYPE(obj), newname, &new_id, false) == 0)
	    return FAILURE;
	if (new_id == old_id)
	    return SUCCESS;
	if (agmapnametoid(agroot(g), AGTYPE(obj), newname, &new_id, true) == 0)
	    return FAILURE;
        /* obj* is unchanged, so no need to re agregister() */
	if (agparent(g) && agidsubg(agparent(g), new_id))
	    return FAILURE;
	agfreeid(g, AGRAPH, old_id);
	AGID(g) = new_id;
	break;
    case AGNODE:
	return agrelabel_node((Agnode_t *) obj, newname);
    case AGINEDGE:
    case AGOUTEDGE:
	return FAILURE;
    default:
	UNREACHABLE();
    }
    return SUCCESS;
}

/* perform initialization/update/finalization method invocation.
 * skip over nil pointers to next method below.
 */

void agmethod_init(Agraph_t * g, void *obj)
{
  aginitcb(g, obj, g->clos->cb);
}

void aginitcb(Agraph_t * g, void *obj, Agcbstack_t * cbstack)
{
    agobjfn_t fn;

    if (cbstack == NULL)
	return;
    aginitcb(g, obj, cbstack->prev);
    fn = NULL;
    switch (AGTYPE(obj)) {
    case AGRAPH:
	fn = cbstack->f->graph.ins;
	break;
    case AGNODE:
	fn = cbstack->f->node.ins;
	break;
    case AGEDGE:
	fn = cbstack->f->edge.ins;
	break;
    default: // ignore
	break;
    }
    if (fn)
	fn(g, obj, cbstack->state);
}

void agmethod_upd(Agraph_t * g, void *obj, Agsym_t * sym)
{
  agupdcb(g, obj, sym, g->clos->cb);
}

void agupdcb(Agraph_t * g, void *obj, Agsym_t * sym, Agcbstack_t * cbstack)
{
    agobjupdfn_t fn;

    if (cbstack == NULL)
	return;
    agupdcb(g, obj, sym, cbstack->prev);
    fn = NULL;
    switch (AGTYPE(obj)) {
    case AGRAPH:
	fn = cbstack->f->graph.mod;
	break;
    case AGNODE:
	fn = cbstack->f->node.mod;
	break;
    case AGEDGE:
	fn = cbstack->f->edge.mod;
	break;
    default: // ignore
	break;
    }
    if (fn)
	fn(g, obj, cbstack->state, sym);
}

void agmethod_delete(Agraph_t * g, void *obj)
{
  agdelcb(g, obj, g->clos->cb);
}

void agdelcb(Agraph_t * g, void *obj, Agcbstack_t * cbstack)
{
    agobjfn_t fn;

    if (cbstack == NULL)
	return;
    agdelcb(g, obj, cbstack->prev);
    fn = NULL;
    switch (AGTYPE(obj)) {
    case AGRAPH:
	fn = cbstack->f->graph.del;
	break;
    case AGNODE:
	fn = cbstack->f->node.del;
	break;
    case AGEDGE:
	fn = cbstack->f->edge.del;
	break;
    default: // ignore
	break;
    }
    if (fn)
	fn(g, obj, cbstack->state);
}

Agraph_t *agroot(void* obj)
{
    if (obj == 0) return NULL;
    switch (AGTYPE(obj)) {
    case AGINEDGE:
    case AGOUTEDGE:
	return ((Agedge_t *) obj)->node->root;
    case AGNODE:
	return ((Agnode_t *) obj)->root;
    case AGRAPH:
	return ((Agraph_t *) obj)->root;
    default:			/* actually can't occur if only 2 bit tags */
	agerrorf("agroot of a bad object");
	return NULL;
    }
}

Agraph_t *agraphof(void *obj)
{
    switch (AGTYPE(obj)) {
    case AGINEDGE:
    case AGOUTEDGE:
	return ((Agedge_t *) obj)->node->root;
    case AGNODE:
	return ((Agnode_t *) obj)->root;
    case AGRAPH:
	return obj;
    default:			/* actually can't occur if only 2 bit tags */
	agerrorf("agraphof a bad object");
	return NULL;
    }
}

/* to manage disciplines */
void agpushdisc(Agraph_t * g, Agcbdisc_t * cbd, void *state)
{
    Agcbstack_t *stack_ent = gv_alloc(sizeof(Agcbstack_t));
    stack_ent->f = cbd;
    stack_ent->state = state;
    stack_ent->prev = g->clos->cb;
    g->clos->cb = stack_ent;
}

int agpopdisc(Agraph_t * g, Agcbdisc_t * cbd)
{
    Agcbstack_t *stack_ent;

    stack_ent = g->clos->cb;
    if (stack_ent) {
	if (stack_ent->f == cbd)
	    g->clos->cb = stack_ent->prev;
	else {
	    while (stack_ent && stack_ent->prev->f != cbd)
		stack_ent = stack_ent->prev;
	    if (stack_ent && stack_ent->prev)
		stack_ent->prev = stack_ent->prev->prev;
	}
	if (stack_ent) {
	    free(stack_ent);
	    return SUCCESS;
	}
    }
    return FAILURE;
}

int agcontains(Agraph_t* g, void* obj)
{
    Agraph_t* subg;

    if (agroot (g) != agroot (obj)) return 0;
    switch (AGTYPE(obj)) {
    case AGRAPH:
	subg = obj;
	do {
	    if (subg == g) return 1;
	} while ((subg = agparent (subg)));
	return 0;
    case AGNODE: 
        return agidnode(g, AGID(obj), 0) != 0;
    default:
        return agsubedge(g, obj, 0) != 0;
    }
}

int agobjkind(void *arg)
{
	return AGTYPE(arg);
}
