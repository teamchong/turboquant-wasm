/// @file
/// @ingroup common_utils
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

#include <common/types.h>
#include <common/utils.h>
#include <time.h>

static clock_t T;

void start_timer(void) { T = clock(); }

double elapsed_sec(void) {
  const clock_t S = clock();
  return (S - T) / (double)CLOCKS_PER_SEC;
}
