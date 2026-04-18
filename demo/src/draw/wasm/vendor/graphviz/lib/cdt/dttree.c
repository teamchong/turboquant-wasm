#include "config.h"

#include	<cdt/dthdr.h>
#include	<stdlib.h>

/*	Ordered set/multiset
**	dt:	dictionary being searched
**	obj:	the object to look for.
**	type:	search type.
**
**      Written by Kiem-Phong Vo (5/25/96)
*/

static void* dttree(Dt_t* dt, void* obj, int type)
{
	Dtlink_t	*root, *t;
	int		cmp, lk, sz, ky;
	void		*o, *k, *key;
	Dtlink_t	*l, *r, *me = NULL, link;
	Dtcompar_f	cmpf;
	Dtdisc_t*	disc;

	UNFLATTEN(dt);
	disc = dt->disc; _DTDSC(disc,ky,sz,lk,cmpf);

	root = dt->data.here;
	if(!obj)
	{	if(!root || !(type&(DT_CLEAR|DT_FIRST|DT_LAST)) )
			return NULL;

		if(type&DT_CLEAR) /* delete all objects */
		{	if(disc->freef || disc->link < 0)
			{	do
				{	while((t = root->left) )
						RROTATE(root,t);
					t = root->right;
					if(disc->freef)
						disc->freef(_DTOBJ(root, lk));
					if(disc->link < 0)
						free(root);
				} while((root = t) );
			}

			dt->data.size = 0;
			dt->data.here = NULL;
			return NULL;
		}
		else /* computing largest/smallest element */
		{	if(type&DT_LAST)
			{	while((t = root->right) )
					LROTATE(root,t);
			}
			else /* type&DT_FIRST */
			{	while((t = root->left) )
					RROTATE(root,t);
			}

			dt->data.here = root;
			return _DTOBJ(root,lk);
		}
	}

	/* note that link.right is LEFT tree and link.left is RIGHT tree */
	l = r = &link;

	/* allow apps to delete an object "actually" in the dictionary */
	if(dt->meth->type == DT_OBAG && (type&(DT_DELETE|DT_DETACH)) )
	{	key = _DTKEY(obj,ky,sz);
		for(o = dtsearch(dt,obj); o; o = dtnext(dt,o) )
		{	k = _DTKEY(o,ky,sz);
			if(_DTCMP(key, k, cmpf, sz) != 0)
				break;
			if(o == obj)
			{	root = dt->data.here;
				l->right = root->left;
				r->left  = root->right;
				goto dt_delete;
			}
		}
	}

	if(type&(DT_MATCH|DT_SEARCH|DT_INSERT))
	{	key = (type&DT_MATCH) ? obj : _DTKEY(obj,ky,sz);
		if(root)
			goto do_search;
	}
	else if(type&DT_RENEW)
	{	me = obj;
		obj = _DTOBJ(me,lk);
		key = _DTKEY(obj,ky,sz);
		if(root)
			goto do_search;
	}
	else if(root && _DTOBJ(root,lk) != obj)
	{	key = _DTKEY(obj,ky,sz);
	do_search:
		while(1)
		{	k = _DTOBJ(root,lk); k = _DTKEY(k,ky,sz);
			if((cmp = _DTCMP(key, k, cmpf, sz)) == 0)
				break;
			else if(cmp < 0)
			{	if((t = root->left) )
				{	k = _DTOBJ(t,lk); k = _DTKEY(k,ky,sz);
					if((cmp = _DTCMP(key, k, cmpf, sz)) < 0)
					{	rrotate(root,t);
						rlink(r,t);
						if(!(root = t->left) )
							break;
					}
					else if(cmp == 0)
					{	rlink(r,root);
						root = t;
						break;
					}
					else /* if(cmp > 0) */
					{	llink(l,t);
						rlink(r,root);
						if(!(root = t->right) )
							break;
					}
				}
				else
				{	rlink(r,root);
					root = NULL;
					break;
				}
			}
			else /* if(cmp > 0) */
			{	if((t = root->right) )
				{	k = _DTOBJ(t,lk); k = _DTKEY(k,ky,sz);
					if((cmp = _DTCMP(key, k, cmpf, sz)) > 0)
					{	lrotate(root,t);
						llink(l,t);
						if(!(root = t->right) )
							break;
					}
					else if(cmp == 0)
					{	llink(l,root);
						root = t;
						break;
					}
					else /* if(cmp < 0) */
					{	rlink(r,t);
						llink(l,root);
						if(!(root = t->left) )
							break;
					}
				}
				else
				{	llink(l,root);
					root = NULL;
					break;
				}
			}
		}
	}

	if(root)
	{	/* found it, now isolate it */
		l->right = root->left;
		r->left = root->right;

		if(type&(DT_SEARCH|DT_MATCH))
		{ has_root:
			root->left = link.right;
			root->right = link.left;
			if((dt->meth->type&DT_OBAG) && (type&(DT_SEARCH|DT_MATCH)) )
			{	key = _DTOBJ(root,lk); key = _DTKEY(key,ky,sz);
				while((t = root->left) )
				{	/* find max of left subtree */
					while((r = t->right) )
						LROTATE(t,r);
					root->left = t;

					/* now see if it's in the same group */
					k = _DTOBJ(t,lk); k = _DTKEY(k,ky,sz);
					if(_DTCMP(key, k, cmpf, sz) != 0)
						break;
					RROTATE(root,t);
				}
			}
			dt->data.here = root;
			return _DTOBJ(root,lk);
		}
		else if(type&DT_NEXT)
		{	root->left = link.right;
			root->right = NULL;
			link.right = root;
		dt_next:
			if((root = link.left) )	
			{	while((t = root->left) )
					RROTATE(root,t);
				link.left = root->right;
				goto has_root;
			}
			else	goto no_root;
		}
		else if(type&DT_PREV)
		{	root->right = link.left;
			root->left = NULL;
			link.left = root;
		dt_prev:
			if((root = link.right) )
			{	while((t = root->right) )
					LROTATE(root,t);
				link.right = root->left;
				goto has_root;
			}
			else	goto no_root;
		}
		else if(type&(DT_DELETE|DT_DETACH))
		{	/* taking an object out of the dictionary */
		dt_delete:
			obj = _DTOBJ(root,lk);
			if(disc->freef && (type&DT_DELETE))
				disc->freef(obj);
			if(disc->link < 0)
				free(root);
			if (--dt->data.size < 0)
				--dt->data.size;
			goto no_root;
		}
		else if(type&DT_INSERT)
		{	if(dt->meth->type&DT_OSET)
				goto has_root;
			else
			{	root->left = NULL;
				root->right = link.left;
				link.left = root;
				goto dt_insert;
			}
		}
		else if(type&DT_RENEW) /* a duplicate */
		{	if(dt->meth->type&DT_OSET)
			{	if(disc->freef)
					disc->freef(obj);
				if(disc->link < 0)
					free(me);
			}
			else
			{	me->left = NULL;
				me->right = link.left;
				link.left = me;
				++dt->data.size;
			}
			goto has_root;
		}
	}
	else
	{	/* not found, finish up LEFT and RIGHT trees */
		r->left = NULL;
		l->right = NULL;

		if(type&DT_NEXT)
			goto dt_next;
		else if(type&DT_PREV)
			goto dt_prev;
		else if(type&(DT_SEARCH|DT_MATCH))
		{ no_root:
			while((t = r->left) )
				r = t;
			r->left = link.right;
			dt->data.here = link.left;
			return (type&DT_DELETE) ? obj : NULL;
		}
		else if(type&DT_INSERT)
		{ dt_insert:
			if(disc->makef && (type&DT_INSERT))
				obj = disc->makef(obj, disc);
			if(obj)
			{	if(lk >= 0)
					root = _DTLNK(obj,lk);
				else
				{	root = malloc(sizeof(Dthold_t));
					if(root)
						((Dthold_t*)root)->obj = obj;
					else if(disc->makef && disc->freef &&
						(type&DT_INSERT))
						disc->freef(obj);
				}
			}
			if(root)
			{	if (dt->data.size >= 0)
					++dt->data.size;
				goto has_root;
			}
			else	goto no_root;
		}
		else if(type&DT_RENEW)
		{	root = me;
			++dt->data.size;
			goto has_root;
		}
		else /*if(type&DT_DELETE)*/
		{	obj = NULL;
			goto no_root;
		}
	}

	return NULL;
}

/* make this method available */
static Dtmethod_t	Dtoset_ =  { dttree, DT_OSET };
static Dtmethod_t	Dtobag_ =  { dttree, DT_OBAG };
Dtmethod_t* Dtoset = &Dtoset_;
Dtmethod_t* Dtobag = &Dtobag_;

static Dtmethod_t Dttree_ = { dttree, DT_OSET };
Dtmethod_t* Dttree = &Dttree_;
