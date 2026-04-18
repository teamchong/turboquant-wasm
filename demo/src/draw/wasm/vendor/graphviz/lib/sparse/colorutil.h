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

#include <color.h>
#include <common/colorprocs.h>
#include <util/agxbuf.h>

#ifdef __cplusplus
extern "C" {
#endif

void rgb2hex(float r, float g, float b, agxbuf *cstring, const char *opacity);

#ifdef __cplusplus
}
#endif
