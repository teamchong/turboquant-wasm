/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/


/* TODO:
 * In dot, prefer bottom or top routing
 * In general, prefer closest side to closest side routing.
 * Edge labels
 * Ports/compass points
 * ordering attribute
 * Weights on edges in nodes
 * Edge concentrators?
 */

#include "config.h"

#define DEBUG
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <ortho/maze.h>
#include <ortho/fPQ.h>
#include <ortho/ortho.h>
#include <common/geomprocs.h>
#include <common/globals.h>
#include <common/render.h>
#include <common/pointset.h>
#include <util/alloc.h>
#include <util/exit.h>
#include <util/gv_math.h>
#include <util/list.h>
#include <util/unused.h>

typedef struct {
    double d;
    Agedge_t* e;
} epair_t;

static UNUSED void emitSearchGraph(FILE *fp, sgraph *sg);
static UNUSED void emitGraph(FILE *fp, maze *mp, size_t n_edges,
                             route *route_list, epair_t[]);
#ifdef DEBUG
int odb_flags;
#endif

#define CELL(n) ((cell*)ND_alg(n))

static double MID(double a, double b) {
  return (a + b) / 2.0;
}

/* cellOf:
 * Given 2 snodes sharing a cell, return the cell.
 */
static cell*
cellOf (snode* p, snode* q)
{
    cell* cp = p->cells[0];
    if (cp == q->cells[0] || cp == q->cells[1]) return cp;
    return p->cells[1];
}

static pointf midPt(const cell *cp) {
  return mid_pointf(cp->bb.LL, cp->bb.UR);
}

/* sidePt:
 * Given a cell and an snode on one of its sides, return the
 * midpoint of the side.
 */
static pointf sidePt(const snode ptr, const cell* cp) {
    if (cp == ptr.cells[1]) {
	if (ptr.isVert) {
	    return (pointf){.x = cp->bb.LL.x, .y = MID(cp->bb.LL.y, cp->bb.UR.y)};
	}
	return (pointf){.x = MID(cp->bb.LL.x, cp->bb.UR.x), .y = cp->bb.LL.y};
    }
    if (ptr.isVert) {
	return (pointf){.x = cp->bb.UR.x, .y = MID(cp->bb.LL.y, cp->bb.UR.y)};
    }
    return (pointf){.x = MID(cp->bb.LL.x, cp->bb.UR.x), .y = cp->bb.UR.y};
}

/* setSet:
 * Initialize and normalize segments.
 * p1 stores smaller value
 * Assume b1 != b2
 */
static void
setSeg (segment* sp, bool dir, double fix, double b1, double b2, int l1, int l2)
{
    sp->isVert = dir;
    sp->comm_coord = fix;
    if (b1 < b2) {
	sp->p.p1 = b1;
	sp->p.p2 = b2;
	sp->l1 = l1;
	sp->l2 = l2;
    }
    else {
	sp->p.p2 = b1;
	sp->p.p1 = b2;
	sp->l2 = l1;
	sp->l1 = l2;
    }
}

/* Convert route in shortest path graph to route
 * of segments. This records the first and last cells,
 * plus cells where the path bends.
 * Note that the shortest path will always have at least 4 nodes:
 * the two dummy nodes representing the center of the two real nodes,
 * and the two nodes on the boundary of the two real nodes.
 */
static route
convertSPtoRoute (sgraph* g, snode* fst, snode* lst)
{
    snode* ptr;
    snode* next;
    snode* prev;  /* node in shortest path just previous to next */
    cell* cp;
    cell* ncp;
    segment seg;
    double fix, b1, b2;
    int l1, l2;
    pointf bp1, prevbp = {0.0,0.0};  /* bend points */

    LIST(segment) rte = {0};

    seg.prev = seg.next = 0;
    ptr = prev = N_DAD(fst);
    next = N_DAD(ptr);
    if (IsNode(ptr->cells[0]))
	cp = ptr->cells[1];
    else
	cp = ptr->cells[0];
    bp1 = sidePt(*ptr, cp);
    while (N_DAD(next)!=NULL) {
	ncp = cellOf (prev, next);
	updateWts (g, ncp, N_EDGE(ptr));

        /* add seg if path bends or at end */
	if (ptr->isVert != next->isVert || N_DAD(next) == lst) {
	    const pointf bp2 = ptr->isVert != next->isVert ? midPt (ncp) : sidePt(*next, ncp);
	    if (ptr->isVert) {   /* horizontal segment */
		if (ptr == N_DAD(fst)) l1 = B_NODE;
		else if (prevbp.y > bp1.y) l1 = B_UP;
		else l1 = B_DOWN; 
		if (ptr->isVert != next->isVert) {
		    if (next->cells[0] == ncp) l2 = B_UP;
		    else l2 = B_DOWN;
		}
		else l2 = B_NODE;
		fix = cp->bb.LL.y;
		b1 = cp->bb.LL.x;
		b2 = ncp->bb.LL.x;
	    }
	    else {   /* vertical segment */
		if (ptr == N_DAD(fst)) l1 = B_NODE;
		else if (prevbp.x > bp1.x) l1 = B_RIGHT;
		else l1 = B_LEFT; 
		if (ptr->isVert != next->isVert) {
		    if (next->cells[0] == ncp) l2 = B_RIGHT;
		    else l2 = B_LEFT;
		}
		else l2 = B_NODE;
		fix = cp->bb.LL.x;
		b1 = cp->bb.LL.y;
		b2 = ncp->bb.LL.y;
	    }
	    setSeg (&seg, !ptr->isVert, fix, b1, b2, l1, l2);
	    LIST_APPEND(&rte, seg);
	    cp = ncp;
	    prevbp = bp1;
	    bp1 = bp2;
	    if (ptr->isVert != next->isVert && N_DAD(next) == lst) {
		l2 = B_NODE;
		if (next->isVert) {   /* horizontal segment */
		    if (prevbp.y > bp1.y) l1 = B_UP;
		    else l1 = B_DOWN; 
		    fix = cp->bb.LL.y;
		    b1 = cp->bb.LL.x;
		    b2 = ncp->bb.LL.x;
		}
		else {
		    if (prevbp.x > bp1.x) l1 = B_RIGHT;
		    else l1 = B_LEFT; 
		    fix = cp->bb.LL.x;
		    b1 = cp->bb.LL.y;
		    b2 = ncp->bb.LL.y;
		}
		setSeg (&seg, !next->isVert, fix, b1, b2, l1, l2);
		LIST_APPEND(&rte, seg);
	    }
	    ptr = next;
	}
	prev = next;
	next = N_DAD(next);
    }

    route ret = {0};
    LIST_DETACH(&rte, &ret.segs, &ret.n);
    for (size_t i = 0; i < ret.n; i++) {
	if (i > 0)
	    ret.segs[i].prev = ret.segs + (i - 1);
	if (i < ret.n - 1)
	    ret.segs[i].next = ret.segs + (i + 1);
    }

    return ret;
}

typedef struct {
    Dtlink_t  link;
    double    v;
    Dt_t*     chans;
} chanItem;

static void freeChannel(void *chan) {
    channel *cp = chan;
    free_graph (cp->G);
    LIST_FREE(&cp->seg_list);
    free (cp);
}

static void freeChanItem(void *item) {
    chanItem *cp = item;
    dtclose (cp->chans);
    free (cp);
}

/* chancmpid:
 * Compare intervals. Two intervals are equal if one contains
 * the other. Otherwise, the one with the smaller p1 value is
 * less. 
 * This combines two separate functions into one. Channels are
 * disjoint, so we really only need to key on p1.
 * When searching for a channel containing a segment, we rely on
 * interval containment to return the correct channel.
 */
static int chancmpid(void *k1, void *k2) {
  const paird *key1 = k1;
  const paird *key2 = k2;
  if (key1->p1 > key2->p1) {
    if (key1->p2 <= key2->p2) return 0;
    return 1;
  }
  if (key1->p1 < key2->p1) {
    if (key1->p2 >= key2->p2) return 0;
    return -1;
  }
  return 0;
}   

static int dcmpid(void *k1, void *k2) {
  const double *key1 = k1;
  const double *key2 = k2;
  return fcmp(*key1, *key2);
}   

static Dtdisc_t chanDisc = {
    offsetof(channel,p),
    sizeof(paird),
    offsetof(channel,link),
    0,
    freeChannel,
    chancmpid,
};

static Dtdisc_t chanItemDisc = {
    offsetof(chanItem,v),
    sizeof(double),
    offsetof(chanItem,link),
    0,
    freeChanItem,
    dcmpid,
};

static void
addChan (Dt_t* chdict, channel* cp, double j)
{
    chanItem* subd = dtmatch (chdict, &j);

    if (!subd) {
	subd = gv_alloc(sizeof(chanItem));
	subd->v = j;
	subd->chans = dtopen (&chanDisc, Dtoset);
	dtinsert (chdict, subd);
    }
    if (dtinsert(subd->chans, cp) != cp) {
	free(cp);
    }
}

static Dt_t*
extractHChans (maze* mp)
{
    snode* np;
    Dt_t* hchans = dtopen (&chanItemDisc, Dtoset);

    for (size_t i = 0; i < mp->ncells; i++) {
	channel* chp;
	cell* cp = mp->cells+i;
	cell* nextcp;
	if (IsHScan(cp)) continue;

	/* move left */
	while ((np = cp->sides[M_LEFT]) && (nextcp = np->cells[0]) &&
	    !IsNode(nextcp)) {
	    cp = nextcp;
	}

	chp = gv_alloc(sizeof(channel));
	chp->cp = cp;
	chp->p.p1 = cp->bb.LL.x;

	/* move right */
	cp->flags |= MZ_HSCAN;
	while ((np = cp->sides[M_RIGHT]) && (nextcp = np->cells[1]) &&
	    !IsNode(nextcp)) {
	    cp = nextcp;
	    cp->flags |= MZ_HSCAN;
	}

        chp->p.p2 = cp->bb.UR.x;
	addChan (hchans, chp, chp->cp->bb.LL.y);
    }
    return hchans;
}

static Dt_t*
extractVChans (maze* mp)
{
    snode* np;
    Dt_t* vchans = dtopen (&chanItemDisc, Dtoset);

    for (size_t i = 0; i < mp->ncells; i++) {
	channel* chp;
	cell* cp = mp->cells+i;
	cell* nextcp;
	if (IsVScan(cp)) continue;

	/* move down */
	while ((np = cp->sides[M_BOTTOM]) && (nextcp = np->cells[0]) &&
	    !IsNode(nextcp)) {
	    cp = nextcp;
	}

	chp = gv_alloc(sizeof(channel));
	chp->cp = cp;
	chp->p.p1 = cp->bb.LL.y;

	/* move up */
	cp->flags |= MZ_VSCAN;
	while ((np = cp->sides[M_TOP]) && (nextcp = np->cells[1]) &&
	    !IsNode(nextcp)) {
	    cp = nextcp;
	    cp->flags |= MZ_VSCAN;
	}

        chp->p.p2 = cp->bb.UR.y;
	addChan (vchans, chp, chp->cp->bb.LL.x);
    }
    return vchans;
}

static void
insertChan (channel* chan, segment* seg)
{
    seg->ind_no = LIST_SIZE(&chan->seg_list);
    LIST_APPEND(&chan->seg_list, seg);
}

static channel*
chanSearch (Dt_t* chans, segment* seg)
{
  channel* cp;
  chanItem* chani = dtmatch (chans, &seg->comm_coord);
  assert (chani);
  cp = dtmatch (chani->chans, &seg->p);
  assert (cp);
  return cp;
}

static void
assignSegs (size_t nrtes, route* route_list, maze* mp)
{
    channel* chan;

    for (size_t i=0;i<nrtes;i++) {
	route rte = route_list[i];
	for (size_t j=0;j<rte.n;j++) {
	    segment* seg = rte.segs+j;
	    if (seg->isVert)
		chan = chanSearch(mp->vchans, seg);
	    else
		chan = chanSearch(mp->hchans, seg);
	    insertChan (chan, seg);
	}
    }
}

/* addLoop:
 * Add two temporary nodes to sgraph corresponding to two ends of a loop at cell cp, i
 * represented by dp and sp.
 */
static void
addLoop (sgraph* sg, cell* cp, snode* dp, snode* sp)
{
    for (size_t i = 0; i < cp->nsides; i++) {
	snode* onp = cp->sides[i];

	if (onp->isVert) continue;
	const bool onTop = onp->cells[0] == cp;
	if (onTop)
	    createSEdge (sg, sp, onp, 0);  /* FIX weight */
	else
	    createSEdge (sg, dp, onp, 0);  /* FIX weight */
    }
    sg->nnodes += 2;
}

/* addNodeEdges:
 * Add temporary node to sgraph corresponding to cell cp, represented
 * by np.
 */
static void
addNodeEdges (sgraph* sg, cell* cp, snode* np)
{
    for (size_t i = 0; i < cp->nsides; i++) {
	snode* onp = cp->sides[i];

	createSEdge (sg, np, onp, 0);  /* FIX weight */
    }
    sg->nnodes++;
#ifdef DEBUG
    np->cells[0] = np->cells[1] = cp;
#endif
}

static char* bendToStr (bend b)
{
  char* s = NULL;
  switch (b) {
  case B_NODE :
    s = "B_NODE";
    break;
  case B_UP :
    s = "B_UP";
    break;
  case B_LEFT :
    s = "B_LEFT";
    break;
  case B_DOWN :
    s = "B_DOWN";
    break;
  default:
    assert(b == B_RIGHT);
    s = "B_RIGHT";
    break;
  }
  return s;
}

static void putSeg (FILE* fp, segment* seg)
{
  if (seg->isVert)
    fprintf (fp, "((%f,%f),(%f,%f)) %s %s", seg->comm_coord, seg->p.p1,
      seg->comm_coord, seg->p.p2, bendToStr (seg->l1), bendToStr (seg->l2));
  else
    fprintf (fp, "((%f,%f),(%f,%f)) %s %s", seg->p.p1,seg->comm_coord, 
      seg->p.p2, seg->comm_coord, bendToStr (seg->l1), bendToStr (seg->l2));
}

static UNUSED void dumpChanG(channel *cp, double v) {
  if (LIST_SIZE(&cp->seg_list) < 2) return;
  fprintf (stderr, "channel %.0f (%f,%f)\n", v, cp->p.p1, cp->p.p2);
  for (size_t k = 0; k < LIST_SIZE(&cp->seg_list); ++k) {
    const adj_list_t adj = cp->G->vertices[k].adj_list;
    if (LIST_IS_EMPTY(&adj)) continue;
    putSeg(stderr, LIST_GET(&cp->seg_list, k));
    fputs (" ->\n", stderr);
    for (size_t i = 0; i < LIST_SIZE(&adj); ++i) {
      fputs ("     ", stderr);
      putSeg(stderr, LIST_GET(&cp->seg_list, LIST_GET(&adj, i)));
      fputs ("\n", stderr);
    }
  }
}

static void
assignTrackNo (Dt_t* chans)
{
    Dt_t* lp;
    Dtlink_t* l1;
    Dtlink_t* l2;
    channel* cp;

    for (l1 = dtflatten (chans); l1; l1 = dtlink(chans,l1)) {
	lp = ((chanItem*)l1)->chans;
	for (l2 = dtflatten (lp); l2; l2 = dtlink(lp,l2)) {
	    cp = (channel*)l2;
	    if (!LIST_IS_EMPTY(&cp->seg_list)) {
#ifdef DEBUG
    if (odb_flags & ODB_CHANG) dumpChanG (cp, ((chanItem*)l1)->v);
#endif
		top_sort (cp->G);
		for (size_t k = 0; k < LIST_SIZE(&cp->seg_list); ++k)
		    LIST_GET(&cp->seg_list, k)->track_no = cp->G->vertices[k].topsort_order+1;
	    }
   	}
    }
}

static void
create_graphs(Dt_t* chans)
{
    Dt_t* lp;
    Dtlink_t* l1;
    Dtlink_t* l2;
    channel* cp;

    for (l1 = dtflatten (chans); l1; l1 = dtlink(chans,l1)) {
	lp = ((chanItem*)l1)->chans;
	for (l2 = dtflatten (lp); l2; l2 = dtlink(lp,l2)) {
	    cp = (channel*)l2;
	    cp->G = make_graph(LIST_SIZE(&cp->seg_list));
   	}
    }
}

static int
eqEndSeg (bend S1l2, bend S2l2, bend T1, bend T2)
{
    if ((S1l2==T2 && S2l2!=T2) || (S1l2==B_NODE && S2l2==T1))
	return 0;
    else
	return -1;
}

static int
overlapSeg (segment* S1, segment* S2, bend T1, bend T2)
{
	if(S1->p.p2<S2->p.p2) {
		if (S1->l2 == T1 && S2->l1 == T2) return -1;
		if (S1->l2 == T2 && S2->l1 == T1) return 1;
		return 0;
	}
	if (S1->p.p2 > S2->p.p2) {
		if (S2->l1 == T2 && S2->l2 == T2) return -1;
		if (S2->l1 == T1 && S2->l2 == T1) return 1;
		return 0;
	}
	if (S2->l1 == T2) return eqEndSeg (S1->l2, S2->l2, T1, T2);
	return -1 * eqEndSeg(S2->l2, S1->l2, T1, T2);
}

static int
ellSeg (bend S1l1, bend S1l2, bend T)
{
    if (S1l1 == T) {
	if (S1l2== T) return -1;
	return 0;
    }
    return 1;
}

static int
segCmp (segment* S1, segment* S2, bend T1, bend T2)
{
	/* no overlap */
    if (S1->p.p2 < S2->p.p1 || S1->p.p1 > S2->p.p2) return 0;
	/* left endpoint of S2 inside S1 */
    if(S1->p.p1<S2->p.p1&&S2->p.p1<S1->p.p2)
	return overlapSeg (S1, S2, T1, T2);
	/* left endpoint of S1 inside S2 */
    if (S2->p.p1 < S1->p.p1 && S1->p.p1 < S2->p.p2)
	return -1*overlapSeg (S2, S1, T1, T2);
    if (S1->p.p1 == S2->p.p1) {
	if (S1->p.p2 < S2->p.p2) {
	    if(S1->l2==T1)
		return eqEndSeg (S2->l1, S1->l1, T1, T2);
	    return -1 * eqEndSeg(S2->l1, S1->l1, T1, T2);
	}
	if (S1->p.p2 > S2->p.p2) {
	    if (S2->l2 == T2)
		return eqEndSeg(S1->l1, S2->l1, T1, T2);
	    return -1 * eqEndSeg(S1->l1, S2->l1, T1, T2);
	}
	if (S1->l1 == S2->l1 && S1->l2 == S2->l2)
	    return 0;
	if (S2->l1 == S2->l2) {
	    if (S2->l1 == T1) return 1;
	    if (S2->l1 == T2) return -1;
	    if (S1->l1 != T1 && S1->l2 != T1) return 1;
	    if (S1->l1 != T2 && S1->l2 != T2) return -1;
	    return 0;
	}
	if (S2->l1 == T1 && S2->l2 == T2) {
	    if (S1->l1 != T1 && S1->l2 == T2) return 1;
	    if (S1->l1 == T1 && S1->l2 != T2) return -1;
	    return 0;
	}
	if (S2->l2 == T1 && S2->l1 == T2) {
	    if (S1->l2 != T1 && S1->l1 == T2) return 1;
	    if (S1->l2 == T1 && S1->l1 != T2) return -1;
	    return 0;
	}
	if (S2->l1 == B_NODE && S2->l2 == T1) {
	    return ellSeg (S1->l1, S1->l2, T1);
	}
	if (S2->l1 == B_NODE && S2->l2 == T2) {
	    return -1*ellSeg (S1->l1, S1->l2, T2);
	}
	if (S2->l1 == T1 && S2->l2 == B_NODE) {
	    return ellSeg (S1->l2, S1->l1, T1);
	}
	/* ((S2->l1==T2)&&(S2->l2==B_NODE)) */
	return -1 * ellSeg(S1->l2, S1->l1, T2);
    }
    if (S1->p.p2 == S2->p.p1) {
	if (S1->l2 == S2->l1) return 0;
	if (S1->l2 == T2) return 1;
	return -1;
    }
    /* S1->p.p1==S2->p.p2 */
    if (S1->l1 == S2->l2) return 0;
    if (S1->l1 == T2) return 1;
    return -1;
}

/* Function seg_cmp returns
 *  -1 if S1 HAS TO BE to the right/below S2 to avoid a crossing, 
 *   0 if a crossing is unavoidable or there is no crossing at all or 
 *     the segments are parallel,
 *   1 if S1 HAS TO BE to the left/above S2 to avoid a crossing
 *  -2 if S1 and S2 are incomparable
 *
 * Note: This definition means horizontal segments have track numbers
 * increasing as y decreases, while vertical segments have track numbers
 * increasing as x increases. It would be good to make this consistent,
 * with horizontal track numbers increasing with y. This can be done by
 * switching B_DOWN and B_UP in the first call to segCmp. At present,
 * though, I'm not sure what assumptions are made in handling parallel
 * segments, so we leave the code alone for the time being.
 */
static int
seg_cmp(segment* S1, segment* S2)		
{
    if(S1->isVert!=S2->isVert||S1->comm_coord!=S2->comm_coord) {
	agerrorf("incomparable segments !! -- Aborting\n");
	return -2;
    }
    if(S1->isVert)
	return segCmp (S1, S2, B_RIGHT, B_LEFT);
    else
	return segCmp (S1, S2, B_DOWN, B_UP);
}

static int
add_edges_in_G(channel* cp)
{
    seg_list_t *seg_list = &cp->seg_list;
    const size_t size = LIST_SIZE(&cp->seg_list);
    rawgraph* G = cp->G;

    for (size_t x = 0; x + 1 < size; ++x) {
	for (size_t y = x + 1; y < size; ++y) {
	    const int cmp = seg_cmp(LIST_GET(seg_list, x), LIST_GET(seg_list, y));
	    if (cmp == -2) {
		return -1;
	    } else if (cmp > 0) {
		insert_edge(G,x,y);
	    } else if (cmp == -1) {
		insert_edge(G,y,x);
	    }
	}
    }

    return 0;
}

static int
add_np_edges (Dt_t* chans)
{
    for (Dtlink_t *l1 = dtflatten(chans); l1; l1 = dtlink(chans, l1)) {
	Dt_t *const lp = ((chanItem*)l1)->chans;
	for (Dtlink_t *l2 = dtflatten(lp); l2; l2 = dtlink(lp, l2)) {
	    channel *const cp = (channel*)l2;
	    if (!LIST_IS_EMPTY(&cp->seg_list))
		if (add_edges_in_G(cp)) {
		  return -1;
		}
   	}
    }

    return 0;
}

static segment*
next_seg(segment* seg, int dir)
{
    assert(seg);
    if (!dir)
        return seg->prev;
    else
        return seg->next;
}

/* propagate_prec propagates the precedence relationship along 
 * a series of parallel segments on 2 edges
 */
static int
propagate_prec(segment* seg, int prec, int hops, int dir)
{
    int x;
    int ans=prec;
    segment* next;
    segment* current;

    current = seg;
    for(x=1;x<=hops;x++) {
	next = next_seg(current, dir);
	if(!current->isVert) {
	    if(next->comm_coord==current->p.p1) {
		if(current->l1==B_UP) ans *= -1;
	    }
	    else {
		if(current->l2==B_DOWN) ans *= -1;
	    }
	}
	else {
	    if(next->comm_coord==current->p.p1) {
		if(current->l1==B_RIGHT) ans *= -1;
	    }
	    else {
		if(current->l2==B_LEFT) ans *= -1;
	    }
	}
	current = next;
    }
    return ans;
}

static bool
is_parallel(segment* s1, segment* s2)
{
    assert (s1->comm_coord==s2->comm_coord);
    return s1->p.p1 == s2->p.p1 &&
           s1->p.p2 == s2->p.p2 &&
           s1->l1 == s2->l1 &&
           s1->l2 == s2->l2;
}

/* decide_point returns (through ret) the number of hops needed in the given
 * directions along the 2 edges to get to a deciding point (or NODES) and also
 * puts into prec the appropriate dependency (follows same convention as
 * seg_cmp)
 */
static int
decide_point(pair *ret, segment* si, segment* sj, int dir1, int dir2)
{
    int prec = 0, ans = 0, temp;
    segment* np1;
    segment *np2 = NULL;
    
    while ((np1 = next_seg(si,dir1)) && (np2 = next_seg(sj,dir2)) &&
	is_parallel(np1, np2)) {
	ans++;
	si = np1;
	sj = np2;
    }
    if (!np1)
	prec = 0;
    else if (!np2)
	assert(0); /* FIXME */
    else {
	temp = seg_cmp(np1, np2);
	if (temp == -2) {
	    return -1;
	}
	prec = propagate_prec(np1, temp, ans+1, 1-dir1);
    }
		
    ret->a = ans;
    ret->b = prec;
    return 0;
}

/* sets the edges for a series of parallel segments along two edges starting 
 * from segment i, segment j. It is assumed that the edge should be from 
 * segment i to segment j - the dependency is appropriately propagated
 */
static void
set_parallel_edges (segment* seg1, segment* seg2, int dir1, int dir2, int hops,
    maze* mp)
{
    int x;
    channel* chan;
    channel* nchan;
    segment* prev1;
    segment* prev2;

    if (seg1->isVert)
	chan = chanSearch(mp->vchans, seg1);
    else
	chan = chanSearch(mp->hchans, seg1);
    insert_edge(chan->G, seg1->ind_no, seg2->ind_no);

    for (x=1;x<=hops;x++) {
	prev1 = next_seg(seg1, dir1);
	prev2 = next_seg(seg2, dir2);
	if(!seg1->isVert) {
	    nchan = chanSearch(mp->vchans, prev1);
	    if(prev1->comm_coord==seg1->p.p1) {
		if(seg1->l1==B_UP) {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		    else
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		}
		else {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		    else
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		}
	    }
	    else {
		if(seg1->l2==B_UP) {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G,prev1->ind_no, prev2->ind_no);
		    else
			insert_edge(nchan->G,prev2->ind_no, prev1->ind_no);
		}
		else {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		    else
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		}
	    }
	}
	else {
	    nchan = chanSearch(mp->hchans, prev1);
	    if(prev1->comm_coord==seg1->p.p1) {
		if(seg1->l1==B_LEFT) {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		    else
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		}
		else {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		    else
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		}
	    }
	    else {
		if(seg1->l2==B_LEFT) {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		    else
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		}
		else {
		    if(edge_exists(chan->G, seg1->ind_no, seg2->ind_no))
			insert_edge(nchan->G, prev1->ind_no, prev2->ind_no);
		    else
			insert_edge(nchan->G, prev2->ind_no, prev1->ind_no);
		}
	    }
	}
	chan = nchan;
	seg1 = prev1;
	seg2 = prev2;
    }
}

/* removes the edge between segments after the resolution of a conflict
 */
static void
removeEdge(segment* seg1, segment* seg2, int dir, maze* mp)
{
    segment* ptr1;
    segment* ptr2;
    channel* chan;

    ptr1 = seg1;
    ptr2 = seg2;
    while(is_parallel(ptr1, ptr2)) {
	ptr1 = next_seg(ptr1, 1);
	ptr2 = next_seg(ptr2, dir);
    }
    if(ptr1->isVert)
	chan = chanSearch(mp->vchans, ptr1);
    else
	chan = chanSearch(mp->hchans, ptr1);
    remove_redge (chan->G, ptr1->ind_no, ptr2->ind_no);
}

static int
addPEdges (channel* cp, maze* mp)
{
    /* dir[1,2] are used to figure out whether we should use prev 
     * pointers or next pointers -- 0 : decrease, 1 : increase
     */
    int dir;
    /* number of hops along the route to get to the deciding points */
    pair hops;
    /* precedences of the deciding points : same convention as 
     * seg_cmp function 
     */
    int prec1, prec2;
    pair p;
    rawgraph* G = cp->G;
    seg_list_t *segs = &cp->seg_list;

    for(size_t i = 0; i + 1 < LIST_SIZE(&cp->seg_list); ++i) {
	for(size_t j = i + 1; j < LIST_SIZE(&cp->seg_list); ++j) {
	    if (!edge_exists(G,i,j) && !edge_exists(G,j,i)) {
		if (is_parallel(LIST_GET(segs, i), LIST_GET(segs, j))) {
		/* get_directions */
		    if (LIST_GET(segs, i)->prev == 0) {
			if (LIST_GET(segs, j)->prev == 0)
			    dir = 0;
			else
			    dir = 1;
		    }
		    else if (LIST_GET(segs, j)->prev == 0) {
			dir = 1;
		    }
		    else {
			if (LIST_GET(segs, i)->prev->comm_coord ==
			    LIST_GET(segs, j)->prev->comm_coord)
			    dir = 0;
			else
			    dir = 1;
		    }

		    if (decide_point(&p, LIST_GET(segs, i), LIST_GET(segs, j), 0, dir)
		        != 0) {
			return -1;
		    }
		    hops.a = p.a;
		    prec1 = p.b;
		    if (decide_point(&p, LIST_GET(segs, i), LIST_GET(segs, j), 1,
		                     1 - dir) != 0) {
			return -1;
		    }
		    hops.b = p.a;
		    prec2 = p.b;

		    if (prec1 == -1) {
			set_parallel_edges(LIST_GET(segs, j), LIST_GET(segs, i), dir, 0,
			                   hops.a, mp);
			set_parallel_edges(LIST_GET(segs, j), LIST_GET(segs, i), 1 - dir, 1,
			                   hops.b, mp);
			if(prec2==1)
			    removeEdge(LIST_GET(segs, i), LIST_GET(segs, j), 1 - dir, mp);
		    } else if (prec1 == 0) {
			if (prec2 == -1) {
			    set_parallel_edges(LIST_GET(segs, j), LIST_GET(segs, i), dir, 0,
			                       hops.a, mp);
			    set_parallel_edges(LIST_GET(segs, j), LIST_GET(segs, i), 1 - dir,
			                       1, hops.b, mp);
			} else if (prec2 == 0) {
			    set_parallel_edges(LIST_GET(segs, i), LIST_GET(segs, j), 0, dir,
			                       hops.a, mp);
			    set_parallel_edges(LIST_GET(segs, i), LIST_GET(segs, j), 1,
			                       1 - dir, hops.b, mp);
			} else if (prec2 == 1) {
			    set_parallel_edges(LIST_GET(segs, i), LIST_GET(segs, j), 0, dir,
			                       hops.a, mp);
			    set_parallel_edges(LIST_GET(segs, i), LIST_GET(segs, j), 1,
			                       1 - dir, hops.b, mp);
			}
		    } else if (prec1 == 1) {
			set_parallel_edges(LIST_GET(segs, i), LIST_GET(segs, j), 0, dir,
			                   hops.a, mp);
			set_parallel_edges(LIST_GET(segs, i), LIST_GET(segs, j), 1, 1 - dir,
			                   hops.b, mp);
			if(prec2==-1)
			    removeEdge(LIST_GET(segs, i), LIST_GET(segs, j), 1 - dir, mp);
		    }
		}
	    }
	}
    }

    return 0;
}

static int
add_p_edges (Dt_t* chans, maze* mp)
{
    for (Dtlink_t *l1 = dtflatten(chans); l1; l1 = dtlink(chans, l1)) {
	Dt_t *const lp = ((chanItem*)l1)->chans;
	for (Dtlink_t *l2 = dtflatten(lp); l2; l2 = dtlink(lp, l2)) {
	    if (addPEdges((channel*)l2, mp) != 0) {
	        return -1;
	    }
   	}
    }

    return 0;
}

static int
assignTracks (maze* mp)
{
    /* Create the graphs for each channel */
    create_graphs(mp->hchans);
    create_graphs(mp->vchans);

    /* add edges between non-parallel segments */
    if (add_np_edges(mp->hchans) != 0) {
	return -1;
    }
    if (add_np_edges(mp->vchans) != 0) {
	return -1;
    }

    /* add edges between parallel segments + remove appropriate edges */
    if (add_p_edges(mp->hchans, mp) != 0) {
	return -1;
    }
    if (add_p_edges(mp->vchans, mp) != 0) {
	return -1;
    }

    /* Assign the tracks after a top sort */
    assignTrackNo (mp->hchans);
    assignTrackNo (mp->vchans);

    return 0;
}

static double
vtrack (segment* seg, maze* m)
{
  channel* chp = chanSearch(m->vchans, seg);
  const double f = seg->track_no / ((double)LIST_SIZE(&chp->seg_list) + 1);
  const pointf interp = interpolate_pointf(f, chp->cp->bb.LL, chp->cp->bb.UR);
  return interp.x;
}

static double htrack(segment *seg, maze *m) {
  channel* chp = chanSearch(m->hchans, seg);
  double f = 1.0 - seg->track_no / ((double)LIST_SIZE(&chp->seg_list) + 1);
  double lo = chp->cp->bb.LL.y;
  double hi = chp->cp->bb.UR.y;
  return round(lo + f * (hi - lo));
}

static void attachOrthoEdges(maze *mp, size_t n_edges, route* route_list,
                             splineInfo *sinfo, epair_t es[], bool doLbls) {
    LIST(pointf) ispline = {0};
    textlabel_t* lbl;

    for (size_t irte = 0; irte < n_edges; irte++) {
	Agedge_t *const e = es[irte].e;
	const pointf p1 = add_pointf(ND_coord(agtail(e)), ED_tail_port(e).p);
	const pointf q1 = add_pointf(ND_coord(aghead(e)), ED_head_port(e).p);

	route rte = route_list[irte];
	size_t npts = 1 + 3*rte.n;
	LIST_RESERVE(&ispline, npts);
	    
	segment *seg = rte.segs;
	if (seg == NULL) {
		continue;
	}
	pointf p;
	if (seg->isVert) {
		p = (pointf){.x = vtrack(seg, mp), .y = p1.y};
	}
	else {
		p = (pointf){.x = p1.x, .y = htrack(seg, mp)};
	}
	LIST_APPEND(&ispline, p);
	LIST_APPEND(&ispline, p);

	for (size_t i = 1;i<rte.n;i++) {
		seg = rte.segs+i;
		if (seg->isVert)
		    p.x = vtrack(seg, mp);
		else
		    p.y = htrack(seg, mp);
		LIST_APPEND(&ispline, p);
		LIST_APPEND(&ispline, p);
		LIST_APPEND(&ispline, p);
	}

	if (seg->isVert) {
		p = (pointf){.x = vtrack(seg, mp), .y = q1.y};
	}
	else {
		p = (pointf){.x = q1.x, .y = htrack(seg, mp)};
	}
	LIST_APPEND(&ispline, p);
	LIST_APPEND(&ispline, p);
	if (Verbose > 1)
	    fprintf(stderr, "ortho %s %s\n", agnameof(agtail(e)),agnameof(aghead(e)));
	clip_and_install(e, aghead(e), LIST_FRONT(&ispline), LIST_SIZE(&ispline),
	                 sinfo);
	if (doLbls && (lbl = ED_label(e)) && !lbl->set)
	    addEdgeLabels(e);
	LIST_CLEAR(&ispline);
    }
    LIST_FREE(&ispline);
}

static double edgeLen(Agedge_t *e) {
    pointf p = ND_coord(agtail(e));
    pointf q = ND_coord(aghead(e));
    return DIST2(p, q);
}

static int edgecmp(const void *x, const void *y) {
  const epair_t *e0 = x;
  const epair_t *e1 = y;
  if (e0->d > e1->d) {
    return 1;
  }
  if (e0->d < e1->d) {
    return -1;
  }
  return 0;
}

static bool spline_merge(node_t * n)
{
    (void)n;
    return false;
}

static bool swap_ends_p(edge_t * e)
{
    (void)e;
    return false;
}

/* orthoEdges:
 * For edges without position information, construct an orthogonal routing.
 * If useLbls is true, use edge label info when available to guide routing, 
 * and set label pos for those edges for which this info is not available.
 */
void orthoEdges(Agraph_t *g, bool useLbls) {
    epair_t* es = gv_calloc(agnedges(g), sizeof(epair_t));
    PointSet* ps = NULL;
    textlabel_t* lbl;

    if (Concentrate) 
	ps = newPS();

#ifdef DEBUG
    {
	char* s = agget(g, "odb");
        char c;
	odb_flags = 0;
	if (s && *s != '\0') {
	    while ((c = *s++)) {
		switch (c) {
		case 'c' :
		    odb_flags |= ODB_CHANG;     // emit channel graph 
		    break;
		case 'i' :
		    odb_flags |= (ODB_SGRAPH|ODB_IGRAPH);  // emit search graphs
		    break;
		case 'm' :
		    odb_flags |= ODB_MAZE;      // emit maze
		    break;
		case 'r' :
		    odb_flags |= ODB_ROUTE;     // emit routes in maze
		    break;
		case 's' :
		    odb_flags |= ODB_SGRAPH;    // emit search graph 
		    break;
		default:
		    break;
		}
	    }
	}
    }
#endif
    if (useLbls) {
	agwarningf("Orthogonal edges do not currently handle edge labels. Try using xlabels.\n");
	useLbls = false;
    }
    maze *const mp = mkMaze(g);
    sgraph *const sg = mp->sg;
#ifdef DEBUG
    if (odb_flags & ODB_SGRAPH) emitSearchGraph (stderr, sg);
#endif

    /* store edges to be routed in es, along with their lengths */
    size_t n_edges = 0;
    for (Agnode_t *n = agfstnode (g); n; n = agnxtnode(g, n)) {
        for (Agedge_t *e = agfstout(g, n); e; e = agnxtout(g,e)) {
	    if (Nop == 2 && ED_spl(e)) continue;
	    if (Concentrate) {
		int ti = AGSEQ(agtail(e));
		int hi = AGSEQ(aghead(e));
		if (ti <= hi) {
		    if (isInPS (ps,ti,hi)) continue;
		    addPS(ps,ti,hi);
		}
		else {
		    if (isInPS (ps,hi,ti)) continue;
		    addPS(ps,hi,ti);
		}
	    }
	    es[n_edges].e = e;
	    es[n_edges].d = edgeLen (e);
	    n_edges++;
	}
    }

    route *const route_list = gv_calloc(n_edges, sizeof(route));

    qsort(es, n_edges, sizeof(epair_t), edgecmp);

    const int gstart = sg->nnodes;
    pq_t *const pq = PQgen(sg->nnodes + 2);
    snode *const sn = &sg->nodes[gstart];
    snode *const dn = &sg->nodes[gstart+1];
    for (size_t i = 0; i < n_edges; i++) {
#ifdef DEBUG
	if (i > 0 && (odb_flags & ODB_IGRAPH)) emitSearchGraph (stderr, sg);
#endif
	Agedge_t *const e = es[i].e;
        cell *const start = CELL(agtail(e));
        cell *const dest = CELL(aghead(e));

	if (useLbls && (lbl = ED_label(e)) && lbl->set) {
	}
	else {
	    if (start == dest)
		addLoop (sg, start, dn, sn);
	    else {
       		addNodeEdges (sg, dest, dn);
		addNodeEdges (sg, start, sn);
	    }
       	    if (shortPath(pq, sg, dn, sn)) {
		PQfree(pq);
		goto orthofinish;
       	    }
	}
	    
       	route_list[i] = convertSPtoRoute(sg, sn, dn);
       	reset (sg);
    }
    PQfree(pq);

    mp->hchans = extractHChans (mp);
    mp->vchans = extractVChans (mp);
    assignSegs (n_edges, route_list, mp);
    if (assignTracks(mp) != 0)
	goto orthofinish;
#ifdef DEBUG
    if (odb_flags & ODB_ROUTE) emitGraph (stderr, mp, n_edges, route_list, es);
#endif
    splineInfo sinfo = {swap_ends_p, spline_merge, true, true};
    attachOrthoEdges(mp, n_edges, route_list, &sinfo, es, useLbls);

orthofinish:
    if (Concentrate)
	freePS (ps);

    for (size_t i=0; i < n_edges; i++)
	free (route_list[i].segs);
    free (route_list);
    freeMaze (mp);
    free (es);
}

#include <common/arith.h>
#define TRANS 10

static const char prolog2[] =
"%%!PS-Adobe-2.0\n\
%%%%BoundingBox: (atend)\n\
/point {\n\
  /Y exch def\n\
  /X exch def\n\
  newpath\n\
  X Y 3 0 360 arc fill\n\
} def\n\
/cell {\n\
  /Y exch def\n\
  /X exch def\n\
  /y exch def\n\
  /x exch def\n\
  newpath\n\
  x y moveto\n\
  x Y lineto\n\
  X Y lineto\n\
  X y lineto\n\
  closepath stroke\n\
} def\n\
/node {\n\
 /u exch def\n\
 /r exch def\n\
 /d exch def\n\
 /l exch def\n\
 newpath l d moveto\n\
 r d lineto r u lineto l u lineto\n\
 closepath fill\n\
} def\n\
\n";

static pointf coordOf(cell *cp, snode *np) {
    if (cp->sides[M_TOP] == np) {
	return (pointf){.x = (cp->bb.LL.x + cp->bb.UR.x) / 2, .y = cp->bb.UR.y};
    }
    if (cp->sides[M_BOTTOM] == np) {
	return (pointf){.x = (cp->bb.LL.x + cp->bb.UR.x) / 2, .y = cp->bb.LL.y};
    }
    if (cp->sides[M_LEFT] == np) {
	return (pointf){.x = cp->bb.LL.x, .y = (cp->bb.LL.y + cp->bb.UR.y) / 2};
    }
    if (cp->sides[M_RIGHT] == np) {
	return (pointf){.x = cp->bb.UR.x, .y = (cp->bb.LL.y + cp->bb.UR.y) / 2};
    }
    agerrorf("Node not adjacent to cell -- Aborting\n");
    graphviz_exit(EXIT_FAILURE);
}

static boxf
emitEdge (FILE* fp, Agedge_t* e, route rte, maze* m, boxf bb)
{
    double x, y;
    boxf n = CELL(agtail(e))->bb;
    segment* seg = rte.segs;
    if (seg->isVert) {
	x = vtrack(seg, m);
	y = (n.UR.y + n.LL.y)/2;
    }
    else {
	y = htrack(seg, m);
	x = (n.UR.x + n.LL.x)/2;
    }
    bb.LL.x = fmin(bb.LL.x, x);
    bb.LL.y = fmin(bb.LL.y, y);
    bb.UR.x = fmax(bb.UR.x, x);
    bb.UR.y = fmax(bb.UR.y, y);
    fprintf(fp, "newpath %.0f %.0f moveto\n", x, y);

    for (size_t i = 1;i<rte.n;i++) {
	seg = rte.segs+i;
	if (seg->isVert) {
	    x = vtrack(seg, m);
	}
	else {
	    y = htrack(seg, m);
	}
	bb.LL.x = fmin(bb.LL.x, x);
	bb.LL.y = fmin(bb.LL.y, y);
	bb.UR.x = fmax(bb.UR.x, x);
	bb.UR.y = fmax(bb.UR.y, y);
	fprintf(fp, "%.0f %.0f lineto\n", x, y);
    }

    n = CELL(aghead(e))->bb;
    if (seg->isVert) {
	x = vtrack(seg, m);
	y = (n.UR.y + n.LL.y)/2;
    }
    else {
	y = htrack(seg, m);
	x = (n.LL.x + n.UR.x)/2;
    }
    bb.LL.x = fmin(bb.LL.x, x);
    bb.LL.y = fmin(bb.LL.y, y);
    bb.UR.x = fmax(bb.UR.x, x);
    bb.UR.y = fmax(bb.UR.y, y);
    fprintf(fp, "%.0f %.0f lineto stroke\n", x, y);

    return bb;
}

/**
 * @brief dumps in dot format @ref snode::cells and @ref edges of
 * @ref sgraph for debugging
 *
 * The routine uses coordinates of @ref cells calculated
 * from @ref gcells.
 * Coordinates of @ref gcellg are calculated by original
 * specified graph layout engine.
 */

static UNUSED void emitSearchGraph(FILE *fp, sgraph *sg) {
    pointf p;
    fputs ("graph G {\n", fp);
    fputs (" node[shape=point]\n", fp);
    fputs (" layout=neato\n", fp);
    for (int i = 0; i < sg->nnodes; i++) {
	snode *const np = sg->nodes+i;
	cell *cp = np->cells[0];
	if (cp == np->cells[1]) {
	    p = midPt(cp);
	}
	else {
	    if (IsNode(cp)) cp = np->cells[1];
	    p = coordOf (cp, np);
	}
	fprintf (fp, "  %d [pos=\"%.0f,%.0f!\"]\n", i, p.x, p.y);
    }
    for (int i = 0; i < sg->nedges; i++) {
	sedge *const ep = sg->edges+i;
	fprintf (fp, "  %d -- %d[label=\"%f\"]\n", ep->v1, ep->v2, ep->weight);
    }
    fputs ("}\n", fp);
}

static UNUSED void emitGraph(FILE *fp, maze *mp, size_t n_edges,
                             route *route_list, epair_t es[]) {
    boxf absbb = {.LL = {.x = DBL_MAX, .y = DBL_MAX},
                  .UR = {.x = -DBL_MAX, .y = -DBL_MAX}};

    fputs(prolog2, fp);
    fprintf (fp, "%d %d translate\n", TRANS, TRANS);

    fputs ("0 0 1 setrgbcolor\n", fp);
    for (size_t i = 0; i < mp->ngcells; i++) {
      const boxf bb = mp->gcells[i].bb;
      fprintf (fp, "%f %f %f %f node\n", bb.LL.x, bb.LL.y, bb.UR.x, bb.UR.y);
    }

    for (size_t i = 0; i < n_edges; i++) {
	absbb = emitEdge (fp, es[i].e, route_list[i], mp, absbb);
    }
    
    fputs ("0.8 0.8 0.8 setrgbcolor\n", fp);
    for (size_t i = 0; i < mp->ncells; i++) {
      const boxf bb = mp->cells[i].bb;
      fprintf (fp, "%f %f %f %f cell\n", bb.LL.x, bb.LL.y, bb.UR.x, bb.UR.y);
      absbb.LL.x = fmin(absbb.LL.x, bb.LL.x);
      absbb.LL.y = fmin(absbb.LL.y, bb.LL.y);
      absbb.UR.x = fmax(absbb.UR.x, bb.UR.x);
      absbb.UR.y = fmax(absbb.UR.y, bb.UR.y);
    }

    const boxf bbox = {
      .LL = {.x = absbb.LL.x + TRANS,
             .y = absbb.LL.y + TRANS},
      .UR = {.x = absbb.UR.x + TRANS,
             .y = absbb.UR.y + TRANS}};
    fprintf(fp, "showpage\n%%%%Trailer\n%%%%BoundingBox: %.f %.f %.f %.f\n",
            bbox.LL.x, bbox.LL.y,  bbox.UR.x, bbox.UR.y);
}
