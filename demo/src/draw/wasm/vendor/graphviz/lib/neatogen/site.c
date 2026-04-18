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
#include <neatogen/site.h>
#include <math.h>


int siteidx;
Site *bottomsite;

static Freelist sfl;
static size_t nvertices;

void siteinit(void)
{
    freeinit(&sfl, sizeof(Site));
    nvertices = 0;
}


Site *getsite(void)
{
    return getfree(&sfl);
}

double dist(Site * s, Site * t)
{
    double ans;
    double dx, dy;

    dx = s->coord.x - t->coord.x;
    dy = s->coord.y - t->coord.y;
    ans = hypot(dx, dy);
    return ans;
}


void makevertex(Site * v)
{
    v->sitenbr = nvertices;
    ++nvertices;
}

void deref(Site * v)
{
    --v->refcnt;
    if (v->refcnt == 0)
	makefree(v, &sfl);
}

void ref(Site * v)
{
    ++v->refcnt;
}
