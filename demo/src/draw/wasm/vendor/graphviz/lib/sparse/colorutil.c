/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <sparse/colorutil.h>
#include <sparse/general.h>
#include <stdio.h>
#include <util/agxbuf.h>

static int r2i(float r) {
  /* convert a number in [0,1] to 0 to 255 */
  return (int)(255 * r + 0.5);
}

void rgb2hex(float r, float g, float b, agxbuf *cstring, const char *opacity) {
  agxbprint(cstring, "#%02x%02x%02x", r2i(r), r2i(g), r2i(b));
  // set to semitransparent for multiple sets vis
  if (opacity && strlen(opacity) >= 2) {
    agxbput_n(cstring, opacity, 2);
  }
}
