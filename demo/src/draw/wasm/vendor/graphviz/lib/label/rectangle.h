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

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Rect {
  double boundary[NUMSIDES];
} Rect_t;

void InitRect(Rect_t *r);
#ifdef RTDEBUG
void PrintRect(Rect_t);
#endif
uint64_t RectArea(const Rect_t);
bool Overlap(const Rect_t, const Rect_t);
Rect_t CombineRect(const Rect_t, const Rect_t);
Rect_t NullRect(void);

#ifdef __cplusplus
}
#endif
