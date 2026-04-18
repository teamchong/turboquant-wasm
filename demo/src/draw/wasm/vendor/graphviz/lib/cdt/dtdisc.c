#include "config.h"

#include	<cdt/dthdr.h>
#include	<stddef.h>

/*	Change discipline.
**	dt :	dictionary
**	disc :	discipline
**
**	Written by Kiem-Phong Vo (5/26/96)
*/

Dtdisc_t *dtdisc(Dt_t *dt, Dtdisc_t *disc) {
	Dtsearch_f	searchf;
	Dtlink_t	*r, *t;
	char*	k;
	Dtdisc_t*	old;

	if(!(old = dt->disc) )	/* initialization call from dtopen() */
	{	dt->disc = disc;
		return disc;
	}

	if(!disc)	/* only want to know current discipline */
		return old;

	searchf = dt->meth->searchf;

	UNFLATTEN(dt);

	dt->disc = disc;

	r = dtflatten(dt);
	dt->data.type &= ~DT_FLATTEN;
	dt->data.here = NULL;
	dt->data.size = 0;

	if (dt->data.type & DT_SET)
	{	Dtlink_t	**s, **ends;
		ends = (s = dt->data.htab) + dt->data.ntab;
		while(s < ends)
			*s++ = NULL;
	}

	/* reinsert them */
	while(r)
	{	t = r->right;
		k = _DTOBJ(r,disc->link);
		k = _DTKEY(k, disc->key, disc->size);
		r->hash = dtstrhash(k, disc->size);
		(void)searchf(dt, r, DT_RENEW);
		r = t;
	}

	return old;
}
