#pragma once

/*	Internal definitions for libcdt.
**	Written by Kiem-Phong Vo (5/25/96)
*/

#include <stdlib.h>

#include	<cdt.h>

/* short-hand notations */
#define left	hl._left
#define hash	hl._hash
#define htab	hh._htab
#define head	hh._head

/* this must be disjoint from DT_METHODS */
#define DT_FLATTEN	010000	/* dictionary already flattened	*/
#define DT_WALK		020000	/* hash table being walked	*/

/* hash start size and load factor */
#define HSLOT		(256)
#define HRESIZE(n)	((n) << 1)
#define HLOAD(s)	((s) << 1)
#define HINDEX(n,h)	((h)&((n)-1))

#define UNFLATTEN(dt) ((dt->data.type & DT_FLATTEN) ? dtrestore(dt, NULL) : 0)

/* tree rotation/linking functions */
#define rrotate(x,y)	((x)->left  = (y)->right, (y)->right = (x))
#define lrotate(x,y)	((x)->right = (y)->left,  (y)->left  = (x))
#define rlink(r,x)	((r) = (r)->left   = (x) )
#define llink(l,x)	((l) = (l)->right  = (x) )

#define RROTATE(x,y)	(rrotate(x,y), (x) = (y))
#define LROTATE(x,y)	(lrotate(x,y), (x) = (y))
