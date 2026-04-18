/**
 * @file
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
#include <stdbool.h>
#include <stdio.h>
#include <cgraph/cghdr.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <util/alloc.h>

/* a default ID allocator that works off the shared string lib */

/// information the ID allocator needs to do its job
typedef struct {
  IDTYPE counter; ///< base to derive next identifier from
  Agraph_t *g;    ///< graph in use
} state_t;

static void *idopen(Agraph_t * g, Agdisc_t* disc)
{
    (void)disc;

  state_t *s = gv_alloc(sizeof(state_t));
  *s = (state_t){.g = g};
  return s;
}

static long idmap(void *state, int objtype, char *str, IDTYPE *id,
		  int createflag)
{
    char *s;
    state_t *st = state;

    (void)objtype;
    if (str) {
        if (createflag)
            s = agstrdup(st->g, str);
        else
            s = agstrbind(st->g, str);
        // The scheme of using pointers as the IDs of named objects and odd
        // numbers as the IDs of unnamed objects relies on heap pointers being
        // even, to avoid collisions. So the low bit had better be unset.
        assert((uintptr_t)s % 2 == 0 &&
               "heap pointer with low bit set will collide with anonymous IDs");
        *id = (IDTYPE)(uintptr_t)s;
    } else {
        *id = st->counter * 2 + 1;
        ++st->counter;
    }
    return 1;
}

static void idfree(void *state, int objtype, IDTYPE id)
{
    (void)objtype;
    state_t *st = state;
    if (id % 2 == 0)
	agstrfree(st->g, (char *)(uintptr_t)id, false);
}

static char *idprint(void *state, int objtype, IDTYPE id)
{
    (void)state;
    (void)objtype;
    if (id % 2 == 0)
	return (char *)(uintptr_t)id;
    else
	return NULL;
}

static void idregister(void *state, int objtype, void *obj)
{
    (void)state;
    (void)objtype;
    (void)obj;
}

Agiddisc_t AgIdDisc = {
    idopen,
    idmap,
    idfree,
    idprint,
    free,
    idregister
};

/* aux functions incl. support for disciplines with anonymous IDs */

int agmapnametoid(Agraph_t * g, int objtype, char *str,
          IDTYPE *result, bool createflag) {
    int rv;

    if (str && str[0] != LOCALNAMEPREFIX) {
	rv = (int) AGDISC(g, id)->map(AGCLOS(g, id), objtype, str, result,
				createflag);
	if (rv)
	    return rv;
    }

    /* either an internal ID, or disc. can't map strings */
    if (str) {
	rv = aginternalmaplookup(g, objtype, str, result);
	if (rv)
	    return rv;
    } else
	rv = 0;

    if (createflag) {
	/* get a new anonymous ID, and store in the internal map */
	rv = (int) AGDISC(g, id)->map(AGCLOS(g, id), objtype, NULL, result,
				createflag);
	if (rv && str)
	    aginternalmapinsert(g, objtype, str, *result);
    }
    return rv;
}

void agfreeid(Agraph_t * g, int objtype, IDTYPE id)
{
    (void) aginternalmapdelete(g, objtype, id);
    (AGDISC(g, id)->free) (AGCLOS(g, id), objtype, id);
}

/**
 * Return string representation of object.
 * In general, returns the name of node or graph,
 * and the key of an edge. If edge is anonymous, returns NULL.
 * Uses static buffer for anonymous graphs.
 */
char *agnameof(void *obj)
{
    Agraph_t *g;
    char *rv;

    /* perform internal lookup first */
    g = agraphof(obj);
    rv = aginternalmapprint(g, AGTYPE(obj), AGID(obj));
    if (rv != NULL)
	return rv;

    if (AGDISC(g, id)->print) {
	rv = AGDISC(g, id)->print(AGCLOS(g, id), AGTYPE(obj), AGID(obj));
	if (rv != NULL)
	    return rv;
    }
    if (AGTYPE(obj) != AGEDGE) {
	static char buf[32];
	snprintf(buf, sizeof(buf), "%c%" PRIu64, LOCALNAMEPREFIX, AGID(obj));
	rv = buf;
    }
    else
	rv = 0;
    return rv;
}

/* register a graph object in an external namespace */
void agregister(Agraph_t * g, int objtype, void *obj)
{
	AGDISC(g, id)->idregister(AGCLOS(g, id), objtype, obj);
}
