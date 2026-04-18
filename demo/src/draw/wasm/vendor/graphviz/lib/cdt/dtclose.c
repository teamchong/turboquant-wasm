#include "config.h"

#include	<cdt/dthdr.h>
#include	<stdlib.h>

/*	Close a dictionary
**
**	Written by Kiem-Phong Vo (05/25/96)
*/
int dtclose(Dt_t* dt)
{
	if(!dt || dt->nview > 0 ) /* can't close if being viewed */
		return -1;

	if(dt->view)	/* turn off viewing */
		dtview(dt,NULL);

	/* release all allocated data */
	(void)dt->meth->searchf(dt, NULL, DT_CLEAR);
	if(dtsize(dt) > 0)
		return -1;

	if (dt->data.ntab > 0)
		free(dt->data.htab);

	free(dt);

	return 0;
}
