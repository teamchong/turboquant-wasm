/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include	<circogen/edgelist.h>
#include	<assert.h>
#include	<util/alloc.h>

static void *mkItem(void *p, Dtdisc_t *disc) {
    (void)disc;
    edgelistitem *obj = p;

    edgelistitem *ap = gv_alloc(sizeof(edgelistitem));

    ap->edge = obj->edge;
    return ap;
}

static int cmpItem(void *k1, void *k2) {
    const Agedge_t **key1 = k1;
    const Agedge_t **key2 = k2;
    if (*key1 > *key2) {
	return 1;
    }
    if (*key1 < *key2) {
	return -1;
    }
    return 0;
}

static Dtdisc_t ELDisc = {
    offsetof(edgelistitem, edge),	/* key */
    sizeof(Agedge_t *),		/* size */
    offsetof(edgelistitem, link),	/* link */
    mkItem,
    free,
    cmpItem,
};

edgelist *init_edgelist(void)
{
    edgelist *list = dtopen(&ELDisc, Dtoset);
    return list;
}

void free_edgelist(edgelist * list)
{
    dtclose(list);
}

void add_edge(edgelist * list, Agedge_t * e)
{
    edgelistitem temp;

    temp.edge = e;
    dtinsert(list, &temp);
}

void remove_edge(edgelist * list, Agedge_t * e)
{
    edgelistitem temp;

    temp.edge = e;
    dtdelete(list, &temp);
}

#ifdef DEBUG
void print_edge(edgelist * list)
{
    edgelistitem *temp;
    Agedge_t *ep;

    for (temp = (edgelistitem *) dtflatten(list); temp;
	 temp = (edgelistitem *)dtlink(list, temp)) {
	ep = temp->edge;
	fprintf(stderr, "%s--", agnameof(agtail(ep)));
	fprintf(stderr, "%s \n", agnameof(aghead(ep)));
    }
    fputs("\n", stderr);
}
#endif
