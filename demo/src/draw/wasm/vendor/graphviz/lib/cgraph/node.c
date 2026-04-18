/// @file
/// @ingroup cgraph_core
/// @ingroup cgraph_node
/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include "config.h"

#include <assert.h>
#include <cgraph/cghdr.h>
#include <cgraph/node_set.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <util/alloc.h>
#include <util/unreachable.h>

Agnode_t *agfindnode_by_id(Agraph_t * g, IDTYPE id)
{
    Agsubnode_t *sn;

    sn = node_set_find(g->n_id, id);
    return sn ? sn->node : NULL;
}

static Agnode_t *agfindnode_by_name(Agraph_t * g, char *name)
{
    IDTYPE id;

    if (agmapnametoid(g, AGNODE, name, &id, false))
	return agfindnode_by_id(g, id);
    else
	return NULL;
}

Agnode_t *agfstnode(Agraph_t * g)
{
    Agsubnode_t *sn;
    sn = dtfirst(g->n_seq);
    return sn ? sn->node : NULL;
}

Agnode_t *agnxtnode(Agraph_t * g, Agnode_t * n)
{
    Agsubnode_t *sn;
    sn = agsubrep(g, n);
    if (sn) sn = dtnext(g->n_seq, sn);
    return sn ? sn->node : NULL;
}

Agnode_t *aglstnode(Agraph_t * g)
{
    Agsubnode_t *sn;
    sn = dtlast(g->n_seq);
    return sn ? sn->node : NULL;
}

Agnode_t *agprvnode(Agraph_t * g, Agnode_t * n)
{
    Agsubnode_t *sn;
    sn = agsubrep(g, n);
    if (sn) sn = dtprev(g->n_seq, sn);
    return sn ? sn->node : NULL;
}


/* internal node constructor */
static Agnode_t *newnode(Agraph_t * g, IDTYPE id, uint64_t seq)
{
    assert((seq & SEQ_MASK) == seq && "sequence ID overflow");

    Agnode_t *n = gv_alloc(sizeof(Agnode_t));
    AGTYPE(n) = AGNODE;
    AGID(n) = id;
    AGSEQ(n) = seq & SEQ_MASK;
    n->root = agroot(g);
    if (agroot(g)->desc.has_attrs)
	(void)agbindrec(n, AgDataRecName, sizeof(Agattr_t), false);
    /* nodeattr_init and method_init will be called later, from the
     * subgraph where the node was actually created, but first it has
     * to be installed in all the (sub)graphs up to root. */
    return n;
}

static void installnode(Agraph_t * g, Agnode_t * n)
{
    Agsubnode_t *sn;
    size_t osize;
    (void)osize;

    assert(node_set_size(g->n_id) == (size_t)dtsize(g->n_seq));
    osize = node_set_size(g->n_id);
    if (g == agroot(g)) sn = &(n->mainsub);
    else sn = gv_alloc(sizeof(Agsubnode_t));
    sn->node = n;
    node_set_add(g->n_id, sn);
    dtinsert(g->n_seq, sn);
    assert(node_set_size(g->n_id) == (size_t)dtsize(g->n_seq));
    assert(node_set_size(g->n_id) == osize + 1);
}

static void installnodetoroot(Agraph_t * g, Agnode_t * n)
{
    Agraph_t *par;
    installnode(g, n);
    if ((par = agparent(g)))
	installnodetoroot(par, n);
}

static void initnode(Agraph_t * g, Agnode_t * n)
{
    if (agroot(g)->desc.has_attrs)
	agnodeattr_init(g,n);
    agmethod_init(g, n);
}

/* external node constructor - create by id */
Agnode_t *agidnode(Agraph_t * g, IDTYPE id, int cflag)
{
    Agraph_t *root;
    Agnode_t *n;

    n = agfindnode_by_id(g, id);
    if (n == NULL && cflag) {
	root = agroot(g);
	if (g != root && (n = agfindnode_by_id(root, id))) // old
	    agsubnode(g, n, 1);	/* insert locally */
	else {
	    n = NULL;
	}
    }
    /* else return probe result */
    return n;
}

Agnode_t *agnode(Agraph_t * g, char *name, int cflag)
{
    Agraph_t *root;
    Agnode_t *n;
    IDTYPE id;

    root = agroot(g);
    /* probe for existing node */
    if (agmapnametoid(g, AGNODE, name, &id, false)) {
	if ((n = agfindnode_by_id(g, id)))
	    return n;

	/* might already exist globally, but need to insert locally */
	if (cflag && g != root && (n = agfindnode_by_id(root, id))) {
	    return agsubnode(g, n, 1);
	}
    }

    if (cflag && agmapnametoid(g, AGNODE, name, &id, true)) {	/* reserve id */
	n = newnode(g, id, agnextseq(g, AGNODE));
	installnodetoroot(g, n);
	initnode(g, n);
	assert(agsubrep(g,n));
	agregister(g, AGNODE, n); /* register in external namespace */
	return n;
    }

    return NULL;
}

/* removes image of node and its edges from graph.
   caller must ensure n belongs to g. */
void agdelnodeimage(Agraph_t *g, Agobj_t *node, void *ignored) {
    Agnode_t *const n = (Agnode_t *)((char *)node - offsetof(Agnode_t, base));
    Agedge_t *e, *f;
    Agsubnode_t template = {.node = n};

    (void)ignored;
    for (e = agfstedge(g, n); e; e = f) {
	f = agnxtedge(g, e, n);
	agdeledgeimage(g, &e->base, 0);
    }
    /* If the following lines are switched, switch the discipline using
     * free_subnode below.
     */ 
    node_set_remove(g->n_id, n->base.tag.id);
    dtdelete(g->n_seq, &template);
}

int agdelnode(Agraph_t * g, Agnode_t * n)
{
    Agedge_t *e, *f;

    if (!agfindnode_by_id(g, AGID(n)))
	return FAILURE;		/* bad arg */
    if (g == agroot(g)) {
	for (e = agfstedge(g, n); e; e = f) {
	    f = agnxtedge(g, e, n);
	    agdeledge(g, e);
	}
	if (g->desc.has_attrs)
	    agnodeattr_delete(n);
	agmethod_delete(g, n);
	agrecclose(&n->base);
	agfreeid(g, AGNODE, AGID(n));
    }
    if (agapply(g, &n->base, agdelnodeimage, NULL, false) == SUCCESS) {
	if (g == agroot(g))
	    free(n);
	return SUCCESS;
    } else
	return FAILURE;
}

static void dict_relabel(Agraph_t *ignored, Agobj_t *node, void *arg) {
    (void)ignored;
    Agnode_t *const n = (Agnode_t *)((char *)node - offsetof(Agnode_t, base));

    Agraph_t *const g = agraphof(n);
    const IDTYPE new_id = *(IDTYPE *)arg;
    Agsubnode_t *key = agsubrep(g, n);
    assert(key != NULL && "node being renamed does not exist");
    node_set_remove(g->n_id, key->node->base.tag.id);
    AGID(key->node) = new_id;
    node_set_add(g->n_id, key);
    /* because all the subgraphs share the same node now, this
       now requires a separate deletion and insertion phase */
}

int agrelabel_node(Agnode_t * n, char *newname)
{
    Agraph_t *g;
    IDTYPE new_id;

    g = agroot(agraphof(n));
    if (agfindnode_by_name(g, newname))
	return FAILURE;
    if (agmapnametoid(g, AGNODE, newname, &new_id, true)) {
	if (agfindnode_by_id(agroot(g), new_id) == NULL) {
	    agfreeid(g, AGNODE, AGID(n));
	    agapply(g, &n->base, dict_relabel, &new_id, false);
	    return SUCCESS;
	} else {
	    agfreeid(g, AGNODE, new_id);	/* couldn't use it after all */
	}
        /* obj* is unchanged, so no need to re agregister() */
    }
    return FAILURE;
}

/* lookup or insert <n> in <g> */
Agnode_t *agsubnode(Agraph_t * g, Agnode_t * n0, int cflag)
{
    Agraph_t *par;
    Agnode_t *n;

    if (agroot(g) != n0->root)
	return NULL;
    n = agfindnode_by_id(g, AGID(n0));
    if (n == NULL && cflag) {
	if ((par = agparent(g))) {
	    n = agsubnode(par, n0, cflag);
	    installnode(g, n);
	    /* no callback for existing node insertion in subgraph (?) */
	}
	/* else impossible that <n> doesn't belong to <g> */
    }
    /* else lookup succeeded */
    return n;
}

/// compare a subnode to an identifier for equality
///
/// @param sn0 Operand 1
/// @param sn1 Operand 2
/// @return True if nodes are equal
static bool agsubnodeideq(const Agsubnode_t *sn0, IDTYPE id) {
  return AGID(sn0->node) == id;
}

static int agsubnodeseqcmpf(void *arg0, void *arg1) {
    Agsubnode_t *sn0 = arg0;
    Agsubnode_t *sn1 = arg1;

    if (AGSEQ(sn0->node) < AGSEQ(sn1->node)) return -1;
    if (AGSEQ(sn0->node) > AGSEQ(sn1->node)) return 1;
    return 0; 
}

/* free_subnode:
 * Free Agsubnode_t allocated in installnode. This should
 * only be done for subgraphs, as the root graph uses the
 * subnode structure built into the node. This explains the
 * AGSNMAIN test. Also, note that both the id and the seq
 * dictionaries use the same subnode object, so only one
 * should do the deletion.
 */
static void free_subnode(void *subnode) {
   Agsubnode_t *sn = subnode;
   if (!AGSNMAIN(sn)) 
	free(sn);
}

Dtdisc_t Ag_subnode_seq_disc = {
    .link = offsetof(Agsubnode_t, seq_link), // link offset
    .freef = free_subnode,
    .comparf = agsubnodeseqcmpf,
};

static void agnodesetfinger(Agraph_t *g, Agobj_t *node, void *ignored) {
    Agnode_t *const n = (Agnode_t *)((char *)node - offsetof(Agnode_t, base));
    Agsubnode_t template = {.node = n};
	dtsearch(g->n_seq,&template);
    (void)ignored;
}

static void agnoderenew(Agraph_t *g, Agobj_t *n, void *ignored) {
    dtrenew(g->n_seq, dtfinger(g->n_seq));
    (void)n;
    (void)ignored;
}

int agnodebefore(Agnode_t *fst, Agnode_t *snd)
{
	Agraph_t *g;
	Agnode_t *n, *nxt;
	

	g = agroot(fst);
	if (AGSEQ(fst) > AGSEQ(snd)) return SUCCESS;

	/* move snd out of the way somewhere */
	n = snd;
	if (agapply(g, &n->base, agnodesetfinger, n, false) != SUCCESS) {
		return FAILURE;
	}
	{
		uint64_t seq = g->clos->seq[AGNODE] + 2;
		assert((seq & SEQ_MASK) == seq && "sequence ID overflow");
		AGSEQ(snd) = seq & SEQ_MASK;
	}
	if (agapply(g, &n->base, agnoderenew, n, false) != SUCCESS) {
		return FAILURE;
	}
	n = agprvnode(g,snd);
	do {
		nxt = agprvnode(g,n);
		if (agapply(g, &n->base, agnodesetfinger, n, false) != SUCCESS) {
		  return FAILURE;
		}
		uint64_t seq = AGSEQ(n) + 1;
		assert((seq & SEQ_MASK) == seq && "sequence ID overflow");
		AGSEQ(n) = seq & SEQ_MASK;
		if (agapply(g, &n->base, agnoderenew, n, false) != SUCCESS) {
		  return FAILURE;
		}
		if (n == fst) break;
		n = nxt;
	} while (n);
	if (agapply(g, &snd->base, agnodesetfinger, n, false) != SUCCESS) {
		return FAILURE;
	}
	assert(AGSEQ(fst) != 0 && "sequence ID overflow");
	AGSEQ(snd) = (AGSEQ(fst) - 1) & SEQ_MASK;
	if (agapply(g, &snd->base, agnoderenew, snd, false) != SUCCESS) {
		return FAILURE;
	}
	return SUCCESS;
} 

struct graphviz_node_set {
  Agsubnode_t **slots; ///< backing store for elements
  size_t size;         ///< number of elements in the set
  size_t capacity_exp; ///< log₂ size of `slots`

  /// the minimum and maximum ID of nodes that have been inserted into the set
  ///
  /// These fields are used to optimize existence checks. `node_set_find` can
  /// quickly return `NULL` if the target node is outside the known range to
  /// have been inserted into the set. This seems niche, but negative queries
  /// like this are common enough that this is a measurable performance
  /// improvement.
  bool min_initialized;
  IDTYPE min;
  IDTYPE max;
};

/// a sentinel, marking a set slot from which an element has been deleted
static Agsubnode_t *const TOMBSTONE = (Agsubnode_t *)-1;

/// get the allocated size of the backing storage of a node set
///
/// The capacity of a set is represented as its base-2 exponent, to make clearer
/// to the compiler that it can implement `% capacity` as a mask, avoiding the
/// expense of a modulo operation.
///
/// @param self Set to inspect
/// @return Capacity of the given set
static size_t node_set_get_capacity(const node_set_t *self) {
  assert(self != NULL);
  return self->slots == NULL ? 0 : (size_t)1 << self->capacity_exp;
}

node_set_t *node_set_new(void) { return gv_alloc(sizeof(node_set_t)); }

/// compute a hash of a node
///
/// If the suboptimal choice of using the ID here turns out to be bad for
/// performance, this could be converted to a more sophisticated hashing
/// algorithm. None of the callers depend on the exact implementation.
///
/// @param id Identifier of element being sought/added
/// @return Hash digest of the target node
static size_t node_set_hash(IDTYPE id) { return (size_t)id; }

void node_set_add(node_set_t *self, Agsubnode_t *item) {
  assert(self != NULL);
  assert(item != NULL);

  // a watermark ratio at which the set capacity should be expanded
  static const size_t OCCUPANCY_THRESHOLD_PERCENT = 70;

  // do we need to expand the backing store?
  size_t capacity = node_set_get_capacity(self);
  const bool grow = 100 * self->size >= OCCUPANCY_THRESHOLD_PERCENT * capacity;

  if (grow) {
    const size_t new_c = capacity == 0 ? 10 : self->capacity_exp + 1;
    Agsubnode_t **new_slots = gv_calloc((size_t)1 << new_c,
                                        sizeof(Agsubnode_t *));

    // Construct a new set and copy everything into it. Note we need to rehash
    // because capacity (and hence modulo wraparound behavior) has changed. This
    // conveniently flushes out the tombstones too.
    node_set_t new_self = {.slots = new_slots, .capacity_exp = new_c};
    for (size_t i = 0; i < capacity; ++i) {
      // skip empty slots
      if (self->slots[i] == NULL) {
        continue;
      }
      // skip deleted slots
      if (self->slots[i] == TOMBSTONE) {
        continue;
      }
      node_set_add(&new_self, self->slots[i]);
    }

    // replace ourselves with this new set
    free(self->slots);
    *self = new_self;
  }

  // update bounds of what we have seen
  if (!self->min_initialized || item->node->base.tag.id < self->min) {
    self->min_initialized = true;
    self->min = item->node->base.tag.id;
  }
  if (item->node->base.tag.id > self->max) {
    self->max = item->node->base.tag.id;
  }

  capacity = node_set_get_capacity(self);
  assert(capacity > self->size);

  const size_t hash = node_set_hash(item->node->base.tag.id);

  for (size_t i = 0; i < capacity; ++i) {
    const size_t candidate = (hash + i) % capacity;

    // if we found an empty slot or a previously deleted slot, we can insert
    if (self->slots[candidate] == NULL || self->slots[candidate] == TOMBSTONE) {
      self->slots[candidate] = item;
      ++self->size;
      return;
    }
  }

  UNREACHABLE();
}

Agsubnode_t *node_set_find(node_set_t *self, IDTYPE key) {
  assert(self != NULL);

  // do we know immediately a node of this key has never been inserted?
  if (self->min_initialized && key < self->min) {
    return NULL;
  }
  if (key > self->max) {
    return NULL;
  }

  const size_t hash = node_set_hash(key);
  const size_t capacity = node_set_get_capacity(self);

  for (size_t i = 0; i < capacity; ++i) {
    const size_t candidate = (hash + i) % capacity;

    // if we found an empty slot, the sought item does not exist
    if (self->slots[candidate] == NULL) {
      return NULL;
    }

    // if we found a previously deleted slot, skip over it
    if (self->slots[candidate] == TOMBSTONE) {
      continue;
    }

    if (agsubnodeideq(self->slots[candidate], key)) {
      return self->slots[candidate];
    }
  }

  return NULL;
}

void node_set_remove(node_set_t *self, IDTYPE item) {
  assert(self != NULL);

  const size_t hash = node_set_hash(item);
  const size_t capacity = node_set_get_capacity(self);

  for (size_t i = 0; i < capacity; ++i) {
    const size_t candidate = (hash + i) % capacity;

    // if we found an empty slot, the sought item does not exist
    if (self->slots[candidate] == NULL) {
      return;
    }

    // if we found a previously deleted slot, skip over it
    if (self->slots[candidate] == TOMBSTONE) {
      continue;
    }

    if (agsubnodeideq(self->slots[candidate], item)) {
      assert(self->size > 0);
      self->slots[candidate] = TOMBSTONE;
      --self->size;
      return;
    }
  }
}

size_t node_set_size(const node_set_t *self) {
  assert(self != NULL);
  return self->size;
}

void node_set_free(node_set_t **self) {
  assert(self != NULL);

  if (*self != NULL) {
    free((*self)->slots);
  }

  free(*self);
  *self = NULL;
}
