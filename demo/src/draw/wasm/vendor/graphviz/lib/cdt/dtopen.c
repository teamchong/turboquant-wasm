#include "config.h"

#include	<cdt/dthdr.h>
#include	<stdlib.h>

/* 	Make a new dictionary
**
**	Written by Kiem-Phong Vo (5/25/96)
*/

Dt_t* dtopen(Dtdisc_t* disc, Dtmethod_t* meth)
{
	Dt_t*		dt;

	if(!disc || !meth)
		return NULL;

	/* allocate space for dictionary */
	if(!(dt = malloc(sizeof(Dt_t))))
		return NULL;

	/* initialize all absolutely private data */
	dt->searchf = NULL;
	dt->meth = NULL;
	dt->disc = NULL;
	dtdisc(dt, disc);
	dt->nview = 0;
	dt->view = dt->walk = NULL;
	dt->user = NULL;

	dt->data = (Dtdata_t){.type = meth->type};

	dt->searchf = meth->searchf;
	dt->meth = meth;

	return dt;
}
