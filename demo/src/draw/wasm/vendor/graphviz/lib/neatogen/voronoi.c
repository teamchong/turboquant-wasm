/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <neatogen/mem.h>
#include <neatogen/geometry.h>
#include <neatogen/edges.h>
#include <neatogen/hedges.h>
#include <neatogen/heap.h>
#include <neatogen/voronoi.h>

void voronoi(Site *(*nextsite)(void *context), void *context) {
    Site *newsite, *bot, *top, *temp, *p;
    Site *v;
    Point newintstar = {0};
    char pm;
    Halfedge *lbnd, *rbnd, *llbnd, *rrbnd, *bisector;
    Edge *e;

    edgeinit();
    siteinit();
    pq_t *pq = PQinitialize();
    bottomsite = nextsite(context);
    ELinitialize();

    newsite = nextsite(context);
    while (1) {
	if (!PQempty(pq))
	    newintstar = PQ_min(pq);

	if (newsite != NULL &&
      (PQempty(pq) || newsite->coord.y < newintstar.y ||
       (newsite->coord.y ==newintstar.y && newsite->coord.x < newintstar.x))) {
	    /* new site is smallest */
	    lbnd = ELleftbnd(&newsite->coord);
	    rbnd = ELright(lbnd);
	    bot = rightreg(lbnd);
	    e = gvbisect(bot, newsite);
	    bisector = HEcreate(e, le);
	    ELinsert(lbnd, bisector);
	    if ((p = hintersect(lbnd, bisector)) != NULL) {
		PQdelete(pq, lbnd);
		PQinsert(pq, lbnd, p, dist(p, newsite));
	    }
	    lbnd = bisector;
	    bisector = HEcreate(e, re);
	    ELinsert(lbnd, bisector);
	    if ((p = hintersect(bisector, rbnd)) != NULL)
		PQinsert(pq, bisector, p, dist(p, newsite));
	    newsite = nextsite(context);
	} else if (!PQempty(pq)) {
	    /* intersection is smallest */
	    lbnd = PQextractmin(pq);
	    llbnd = ELleft(lbnd);
	    rbnd = ELright(lbnd);
	    rrbnd = ELright(rbnd);
	    bot = leftreg(lbnd);
	    top = rightreg(rbnd);
	    v = lbnd->vertex;
	    makevertex(v);
	    endpoint(lbnd->ELedge, lbnd->ELpm, v);
	    endpoint(rbnd->ELedge, rbnd->ELpm, v);
	    ELdelete(lbnd);
	    PQdelete(pq, rbnd);
	    ELdelete(rbnd);
	    pm = le;
	    if (bot->coord.y > top->coord.y) {
		temp = bot;
		bot = top;
		top = temp;
		pm = re;
	    }
	    e = gvbisect(bot, top);
	    bisector = HEcreate(e, pm);
	    ELinsert(llbnd, bisector);
	    endpoint(e, re - pm, v);
	    deref(v);
	    if ((p = hintersect(llbnd, bisector)) != NULL) {
		PQdelete(pq, llbnd);
		PQinsert(pq, llbnd, p, dist(p, bot));
	    }
	    if ((p = hintersect(bisector, rrbnd)) != NULL) {
		PQinsert(pq, bisector, p, dist(p, bot));
	    }
	} else
	    break;
    }

    for (lbnd = ELright(ELleftend); lbnd != ELrightend; lbnd = ELright(lbnd)) {
	e = lbnd->ELedge;
	clip_line(e);
    }

    // `PQcleanup` relies on the number of sites, so should be discarded and
    // at least every time we use `vAdjust`. See note in adjust.c:cleanup().
    PQcleanup(pq);
}
