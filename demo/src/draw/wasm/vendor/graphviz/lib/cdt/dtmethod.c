#include "config.h"

#include	<cdt/dthdr.h>
#include	<stdlib.h>

/*	Change search method.
**
**	Written by Kiem-Phong Vo (05/25/96)
*/

Dtmethod_t* dtmethod(Dt_t* dt, Dtmethod_t* meth)
{
	Dtlink_t	*list, *r;
	Dtdisc_t*	disc = dt->disc;
	Dtmethod_t*	oldmeth = dt->meth;

	if(!meth || meth->type == oldmeth->type)
		return oldmeth;

	/* get the list of elements */
	list = dtflatten(dt);

	if (dt->data.type & DT_SET)
	{	if (dt->data.ntab > 0)
			free(dt->data.htab);
		dt->data.ntab = 0;
		dt->data.htab = NULL;
	}

	dt->data.here = NULL;
	dt->data.type = (dt->data.type & ~(DT_METHODS|DT_FLATTEN)) | meth->type;
	dt->meth = meth;
	if(dt->searchf == oldmeth->searchf)
		dt->searchf = meth->searchf;

	if(meth->type&(DT_OSET|DT_OBAG))
	{	dt->data.size = 0;
		while(list)
		{	r = list->right;
			meth->searchf(dt, list, DT_RENEW);
			list = r;
		}
	}
	else if(oldmeth->type&DT_SET)
	{	int	rehash;
		if((meth->type&DT_SET) && !(oldmeth->type&DT_SET))
			rehash = 1;
		else	rehash = 0;

		dt->data.size = 0;
		dt->data.loop = 0;
		while(list)
		{	r = list->right;
			if(rehash)
			{	void* key = _DTOBJ(list,disc->link);
				key = _DTKEY(key,disc->key,disc->size);
				list->hash = dtstrhash(key, disc->size);
			}
			(void)meth->searchf(dt, list, DT_RENEW);
			list = r;
		}
	}

	return oldmeth;
}
