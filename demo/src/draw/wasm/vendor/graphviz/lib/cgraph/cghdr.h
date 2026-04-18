/**
 * @file
 * @brief cgraph.h additions
 * @ingroup core
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

#pragma once

#include "config.h"

#ifdef GVDLL
#ifdef EXPORT_CGHDR
#define CGHDR_API __declspec(dllexport)
#else
#define CGHDR_API __declspec(dllimport)
#endif
#endif

#ifndef CGHDR_API
#define CGHDR_API /* nothing */
#endif

#include <cgraph.h>

#include	 	<ctype.h>
#include		<sys/types.h>
#include		<stdarg.h>
#include		<stdbool.h>
#include		<stdlib.h>
#include		<string.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define	SUCCESS				0
#define FAILURE				-1
#define LOCALNAMEPREFIX		'%'

#define AGDISC(g,d)			((g)->clos->disc.d)
#define AGCLOS(g,d)			((g)->clos->state.d)

	/* functional definitions */
typedef Agobj_t *(*agobjsearchfn_t) (Agraph_t * g, Agobj_t * obj);
CGHDR_API int agapply(Agraph_t * g, Agobj_t * obj, agobjfn_t fn, void *arg,
	    int preorder);

	/* global variables */
extern const char AgDataRecName[];

	/* set ordering disciplines */
extern Dtdisc_t Ag_subnode_seq_disc;
extern Dtdisc_t Ag_mainedge_id_disc;
extern Dtdisc_t Ag_subedge_id_disc;
extern Dtdisc_t Ag_mainedge_seq_disc;
extern Dtdisc_t Ag_subedge_seq_disc;
extern Dtdisc_t Ag_subgraph_id_disc;
extern Dtdisc_t Ag_subgraph_seq_disc;

	/* internal constructor of graphs and subgraphs */
Agraph_t *agopen1(Agraph_t * g);
int agstrclose(Agraph_t * g);

/// Mask of `Agtag_s.seq` width
enum { SEQ_MASK = (1 << (sizeof(unsigned) * 8 - 4)) - 1 };

	/* object set management */
Agnode_t *agfindnode_by_id(Agraph_t * g, IDTYPE id);
uint64_t agnextseq(Agraph_t * g, int objtype);

/* dict helper functions */
Dict_t *agdtopen(Dtdisc_t *disc, Dtmethod_t *method);
void agdtdisc(Agraph_t * g, Dict_t * dict, Dtdisc_t * disc);
int agdtdelete(Agraph_t * g, Dict_t * dict, void *obj);
int agdtclose(Agraph_t * g, Dict_t * dict);

/** @addtogroup cgraph_attr
 *  @{
 */
	/* name-value pair operations */
CGHDR_API Agdatadict_t *agdatadict(Agraph_t *g, bool cflag);
CGHDR_API Agattr_t *agattrrec(void *obj);

void agraphattr_init(Agraph_t * g);
int agraphattr_delete(Agraph_t * g);
void agnodeattr_init(Agraph_t *g, Agnode_t * n);
void agnodeattr_delete(Agnode_t * n);
void agedgeattr_init(Agraph_t *g, Agedge_t * e);
void agedgeattr_delete(Agedge_t * e);
/// @}

	/* parsing and lexing graph files
	 *
	 * aagscan_t is opaque, because it is generated
	 * by flex based on scan.l.
	 * See https://westes.github.io/flex/manual/About-yyscan_005ft.html
	 */
typedef void *aagscan_t;
typedef struct aagextra_s aagextra_t;

int aaglex_init_extra(aagextra_t* user_defined, aagscan_t* scanner);
int aaglex_destroy(aagscan_t);
aagextra_t *aagget_extra(aagscan_t yyscanner);
void aagset_in(FILE *, aagscan_t);

int aagparse(aagscan_t scanner);
void aglexeof(aagscan_t yyscanner);
void aglexbad(aagscan_t yyscanner);

	/* ID management */
int agmapnametoid(Agraph_t *g, int objtype, char *str, IDTYPE *result,
                  bool createflag);
void agfreeid(Agraph_t * g, int objtype, IDTYPE id);
char *agprintid(Agobj_t * obj);
bool aginternalmaplookup(Agraph_t *g, int objtype, char *str, IDTYPE *result);
void aginternalmapinsert(Agraph_t * g, int objtype, char *str,
             IDTYPE result);
char *aginternalmapprint(Agraph_t * g, int objtype, IDTYPE id);
int aginternalmapdelete(Agraph_t * g, int objtype, IDTYPE id);
void aginternalmapclose(Agraph_t * g);
void agregister(Agraph_t * g, int objtype, void *obj);

	/* internal set operations */
void agedgesetop(Agraph_t * g, Agedge_t * e, int insertion);
void agdelnodeimage(Agraph_t *g, Agobj_t *node, void *ignored);
void agdeledgeimage(Agraph_t *g, Agobj_t *edge, void *ignored);
/// rename an object
///
/// @param obj Target to rename
/// @param newname Name to assign
/// @return 0 on success
CGHDR_API int agrename(Agobj_t * obj, char *newname);
void agrecclose(Agobj_t * obj);

/// @addtogroup cgraph_callback
/// @{
void agmethod_init(Agraph_t * g, void *obj);
void agmethod_upd(Agraph_t * g, void *obj, Agsym_t * sym);
void agmethod_delete(Agraph_t * g, void *obj);

void aginitcb(Agraph_t * g, void *obj, Agcbstack_t * disc);
void agupdcb(Agraph_t * g, void *obj, Agsym_t * sym, Agcbstack_t * disc);
void agdelcb(Agraph_t * g, void *obj, Agcbstack_t * disc);
/// @}

/// @defgroup cgmem memory
/// @{
void aginternalmapclearlocalnames(Agraph_t *g);
/// @}
