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

#include <dotgen/dot.h>
#include <stdio.h>

/*
 * Author: Mohammad T. Irfan
 *   Summer, 2008
 */

/* TODO:
 *   - Support clusters
 *   - Support disconnected graphs
 *   - Provide algorithms for aspect ratios < 1
 */

void setAspect(Agraph_t *g) {
  const char *const p = agget(g, "aspect");

  if (!p || sscanf(p, "%lf,%d", &(double){0}, &(int){0}) <= 0) {
    return;
  }
  agwarningf("the aspect attribute has been disabled due to implementation "
             "flaws - attribute ignored.\n");
}
