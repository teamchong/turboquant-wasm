/**
 * @file
 * @brief API: cgraph.h, cghdr.h
 * @ingroup cgraph_utils
 */
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
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util/alloc.h>
#include <util/unreachable.h>

/*
 * reference counted strings.
 */

typedef struct {
    uint64_t refcnt: sizeof(uint64_t) * 8 - 1;
    uint64_t is_html: 1;
    char s[];
} refstr_t;

_Static_assert(
    offsetof(refstr_t, s) % 2 == 0,
    "refstr_t.s is not at an even offset, breaking lib/cgraph/id.c code");

/// compare a string to a reference-counted string for equality
///
/// @param a Content of the first string
/// @param is_html Whether the first string was an HTML-like string
/// @param b The second reference-counted string
/// @return True if the two were equal
static bool refstr_eq(const char *a, bool is_html, const refstr_t *b) {
  if (is_html != b->is_html) {
    return false;
  }
  return strcmp(a, b->s) == 0;
}

/// a string dictionary
typedef struct {
  refstr_t **buckets;  ///< backing store of elements
  size_t size;         ///< number of elements in the dictionary
  size_t capacity_exp; ///< log₂ size of `buckets`
} strdict_t;

static strdict_t *Refdict_default;

/// derive a hash value from the given data
///
/// @param key Start of data to read
/// @param len Number of bytes to read
/// @param extra An extra byte to include in the hash
/// @return A hash digest suitable for dictionary indexing
static uint64_t hash(const void *key, size_t len, uint8_t extra) {
  assert(key != NULL || len == 0);

  // The following implementation is based on the `MurmurHash64A` variant of the
  // public domain MurmurHash by Austin Appleby. More information on this at
  // https://github.com/aappleby/smhasher/. Relevant changes made to Austin’s
  // original implementation:
  //   • Our implementation is alignment-agnostic. No assumption is made about
  //     the initial alignment of `key`.
  //   • Our implementation uses `unsigned char` pointers, avoiding Undefined
  //     Behavior when the input pointer originated from a non-`uint64_t`
  //     object. This is written in a style that allows contemporary compilers
  //     to optimize code back into wider 8-byte accesses where possible.
  //   • Our implementation supports an extra byte to be considered to have
  //     followed the main data. See calls to this function for why this `extra`
  //     parameter exists.

  static const uint64_t seed = 0;

  const uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
  const unsigned r = 47;

  uint64_t h = seed ^ (len * m);

  const unsigned char *data = key;
  const unsigned char *end = data + len / sizeof(uint64_t) * sizeof(uint64_t);

  while (data != end) {

    uint64_t k;
    memcpy(&k, data, sizeof(k));
    data += sizeof(k);

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const unsigned char *data2 = data;

  // accumulate extra byte
  h ^= (uint64_t)extra << 56;

  switch (len & 7) {
  case 7:
    h ^= (uint64_t)data2[6] << 48; // fall through
  case 6:
    h ^= (uint64_t)data2[5] << 40; // fall through
  case 5:
    h ^= (uint64_t)data2[4] << 32; // fall through
  case 4:
    h ^= (uint64_t)data2[3] << 24; // fall through
  case 3:
    h ^= (uint64_t)data2[2] << 16; // fall through
  case 2:
    h ^= (uint64_t)data2[1] << 8; // fall through
  case 1:
    h ^= (uint64_t)data2[0];
    break;
  default:
    // nothing required
    break;
  }
  h *= m;

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

/// create a new string dictionary
static strdict_t *strdict_new(void) { return gv_alloc(sizeof(strdict_t)); }

/// derive hash for a given reference-counted string
///
/// @param s The reference-counted string’s `s` member
/// @param is_html Is this an HTML-like string?
/// @return A hash digest suitable for dictionary indexing
static size_t strdict_hash(const char *s, bool is_html) {
  assert(s != NULL);
  return (size_t)hash(s, strlen(s), is_html);
}

/// a sentinel, marking a dictionary bucket from which an element has been
/// deleted
static refstr_t *const TOMBSTONE = (refstr_t *)-1;

/// add a reference-counted string to a dictionary
static void strdict_add(strdict_t *dict, refstr_t *r) {
  assert(dict != NULL);
  assert(r != NULL);
  assert(r != TOMBSTONE);

  // a watermark ratio at which the set capacity should be expanded
  static const size_t OCCUPANCY_THRESHOLD_PERCENT = 70;

  // do we need to expand the backing store?
  size_t capacity = dict->buckets == NULL ? 0 : (size_t)1 << dict->capacity_exp;
  const bool grow = 100 * dict->size >= OCCUPANCY_THRESHOLD_PERCENT * capacity;

  if (grow) {
    const size_t new_c = capacity == 0 ? 10 : dict->capacity_exp + 1;
    refstr_t **new_b = gv_calloc((size_t)1 << new_c, sizeof(refstr_t *));

    // Construct a new dictionary and copy everything into it. Note we need to
    // rehash because capacity (and hence modulo wraparound behavior) has
    // changed. This conveniently flushes out the tombstones too.
    strdict_t new_d = {.buckets = new_b, .capacity_exp = new_c};
    for (size_t i = 0; i < capacity; ++i) {
      // skip empty buckets
      if (dict->buckets[i] == NULL) {
        continue;
      }
      // skip deleted buckets
      if (dict->buckets[i] == TOMBSTONE) {
        continue;
      }
      strdict_add(&new_d, dict->buckets[i]);
    }

    // replace ourselves with this new dictionary
    free(dict->buckets);
    *dict = new_d;
  }

  assert(dict->buckets != NULL);
  capacity = (size_t)1 << dict->capacity_exp;
  assert(capacity > dict->size);

  const size_t h = strdict_hash(r->s, r->is_html != 0);

  for (size_t i = 0; i < capacity; ++i) {
    const size_t candidate = (h + i) % capacity;

    // if we found an empty bucket or a previously deleted bucket, we can insert
    if (dict->buckets[candidate] == NULL ||
        dict->buckets[candidate] == TOMBSTONE) {
      dict->buckets[candidate] = r;
      ++dict->size;
      return;
    }
  }

  UNREACHABLE();
}

/// lookup a reference-counted string in a dictionary
///
/// @param dict String dictionary to search
/// @param s String content to search for
/// @param is_html Is this an HTML-like string?
/// @return Found reference-counted string, or null if not found
static refstr_t *strdict_find(strdict_t *dict, const char *s, bool is_html) {
  assert(dict != NULL);
  assert(s != NULL);

  const size_t h = strdict_hash(s, is_html);
  const size_t capacity = dict->buckets == NULL
                        ? 0 : (size_t)1 << dict->capacity_exp;

  for (size_t i = 0; i < capacity; ++i) {
    const size_t candidate = (h + i) % capacity;

    // if we found an empty bucket, the sought item does not exist
    if (dict->buckets[candidate] == NULL) {
      return NULL;
    }

    // if we found a previously deleted slot, skip over it
    if (dict->buckets[candidate] == TOMBSTONE) {
      continue;
    }

    // is this the string we are searching for?
    if (refstr_eq(s, is_html, dict->buckets[candidate])) {
      return dict->buckets[candidate];
    }
  }

  // not found
  return NULL;
}

/// remove a reference-counted string from a dictionary
static void strdict_remove(strdict_t *dict, const refstr_t *key) {
  assert(dict != NULL);
  assert(key != NULL);
  assert(key != TOMBSTONE);

  const size_t h = strdict_hash(key->s, key->is_html != 0);
  const size_t capacity = dict->buckets == NULL
                        ? 0 : (size_t)1 << dict->capacity_exp;

  for (size_t i = 0; i < capacity; ++i) {
    const size_t candidate = (h + i) % capacity;

    // if we found an empty bucket, the sought item does not exist
    if (dict->buckets[candidate] == NULL) {
      return;
    }

    // if we found a previously deleted bucket, skip over it
    if (dict->buckets[candidate] == TOMBSTONE) {
      continue;
    }

    // is this the string we are searching for?
    if (refstr_eq(key->s, key->is_html != 0, dict->buckets[candidate])) {
      assert(dict->size > 0);
      free(dict->buckets[candidate]);
      dict->buckets[candidate] = TOMBSTONE;
      --dict->size;
      return;
    }
  }
}

/// destroy a string dictionary
static void strdict_free(strdict_t **dict) {
  assert(dict != NULL);

  if (*dict != NULL && (*dict)->buckets != NULL) {
    for (size_t i = 0; i < (size_t)1 << (*dict)->capacity_exp; ++i) {
      if ((*dict)->buckets[i] != TOMBSTONE) {
        free((*dict)->buckets[i]);
      }
    }
    free((*dict)->buckets);
  }

  free(*dict);
  *dict = NULL;
}

/* refdict:
 * Return a pointer to the string dictionary associated with g.
 * If necessary, create it.
 */
static strdict_t **refdict(Agraph_t *g) {
    strdict_t **dictref;

    if (g)
	dictref = (strdict_t **)&g->clos->strdict;
    else
	dictref = &Refdict_default;
    if (*dictref == NULL) {
	*dictref = strdict_new();
    }
    return dictref;
}

int agstrclose(Agraph_t * g)
{
    strdict_free(refdict(g));
    return 0;
}

static char *refstrbind(strdict_t *strdict, const char *s, bool is_html) {
    refstr_t *r;
    r = strdict_find(strdict, s, is_html);
    if (r)
	return r->s;
    else
	return NULL;
}

char *agstrbind(Agraph_t *g, const char *s) {

  // did this string originate from `agstrdup_html(g, …)`?
  if (s != NULL) {
    strdict_t *const strdict = *refdict(g);
    refstr_t *const ref = strdict_find(strdict, s, true);
    if (ref != NULL && ref->s == s) {
      // create this copy as HTML-like
      return agstrbind_html(g, s);
    }
  }

  return agstrbind_text(g, s);
}

char *agstrbind_html(Agraph_t *g, const char *s) {
  return refstrbind(*refdict(g), s, true);
}

char *agstrbind_text(Agraph_t * g, const char *s)
{
    return refstrbind(*refdict(g), s, false);
}

static char *agstrdup_internal(Agraph_t *g, const char *s, bool is_html) {
    refstr_t *r;

    if (s == NULL)
	 return NULL;
    strdict_t *strdict = *refdict(g);
    r = strdict_find(strdict, s, is_html);
    if (r)
	r->refcnt++;
    else {
	const size_t s_size = strlen(s) + 1;
	const size_t sz = sizeof(refstr_t) + s_size;
	if (g)
	    r = gv_calloc(sz, sizeof(char));
	else {
	    r = malloc(sz);
	    if (r == NULL) {
	        return NULL;
	    }
	}
	r->refcnt = 1;
	r->is_html = is_html;
	memcpy(r->s, s, s_size);
	strdict_add(strdict, r);
    }
    return r->s;
}

char *agstrdup_text(Agraph_t *g, const char *s) {
  return agstrdup_internal(g, s, false);
}

char *agstrdup_html(Agraph_t *g, const char *s) {
  return agstrdup_internal(g, s, true);
}

char *agstrdup(Agraph_t *g, const char *s) {

  // did this string originate from `agstrdup_html(g, …)`?
  if (s != NULL) {
    strdict_t *const strdict = *refdict(g);
    refstr_t *const ref = strdict_find(strdict, s, true);
    if (ref != NULL && ref->s == s) {
      // create this copy as HTML-like
      return agstrdup_html(g, s);
    }
  }

  // otherwise, create the copy as regular text
  return agstrdup_text(g, s);
}

int agstrfree(Agraph_t *g, const char *s, bool is_html) {
    refstr_t *r;

    if (s == NULL)
	 return FAILURE;

    strdict_t *strdict = *refdict(g);
    r = strdict_find(strdict, s, is_html);
    if (r && r->s == s) {
	r->refcnt--;
	if (r->refcnt == 0) {
	    strdict_remove(strdict, r);
	}
    }
    if (r == NULL)
	return FAILURE;
    return SUCCESS;
}

/* aghtmlstr:
 * Return true if s is an HTML string.
 * We assume s is within a refstr.
 */
int aghtmlstr(const char *s)
{
    const refstr_t *key;

    if (s == NULL)
	return 0;
    key = (const refstr_t *)(s - offsetof(refstr_t, s));
    return key->is_html != 0;
}

#ifdef DEBUG
static int refstrprint(const refstr_t *r) {
    fprintf(stderr, "%s\n", r->s);
    return 0;
}

void agrefstrdump(Agraph_t * g)
{
    const strdict_t *d = *refdict(g);
    for (size_t i = 0;
         d != NULL && d->buckets != NULL && i < (size_t)1 << d->capacity_exp;
         ++i) {
	refstrprint(d->buckets[i]);
    }
}
#endif
