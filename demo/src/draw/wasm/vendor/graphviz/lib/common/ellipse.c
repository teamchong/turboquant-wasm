/// @file
/// @ingroup common_render
/*************************************************************************
 * Copyright (c) 2012 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

/* This code is derived from the Java implementation by Luc Maisonobe */
/* Copyright (c) 2003-2004, Luc Maisonobe
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with
 * or without modification, are permitted provided that
 * the following conditions are met:
 *
 *    Redistributions of source code must retain the
 *    above copyright notice, this list of conditions and
 *    the following disclaimer.
 *    Redistributions in binary form must reproduce the
 *    above copyright notice, this list of conditions and
 *    the following disclaimer in the documentation
 *    and/or other materials provided with the
 *    distribution.
 *    Neither the names of spaceroots.org, spaceroots.com
 *    nor the names of their contributors may be used to
 *    endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 * THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <assert.h>
#include <common/render.h>
#include <limits.h>
#include <math.h>
#include <pathplan/pathplan.h>
#include <stdbool.h>
#include <util/alloc.h>
#include <util/list.h>

#define TWOPI (2 * M_PI)

typedef struct {
  double cx, cy; /* center */
  double a, b;   /* semi-major and -minor axes */

  /* Start and end angles of the arc. */
  double eta1, eta2;
} ellipse_t;

static void initEllipse(ellipse_t *ep, double cx, double cy, double a, double b,
                        double lambda1, double lambda2) {
  ep->cx = cx;
  ep->cy = cy;
  ep->a = a;
  ep->b = b;

  ep->eta1 = atan2(sin(lambda1) / b, cos(lambda1) / a);
  ep->eta2 = atan2(sin(lambda2) / b, cos(lambda2) / a);

  // make sure we have eta1 <= eta2 <= eta1 + 2*PI
  ep->eta2 -= TWOPI * floor((ep->eta2 - ep->eta1) / TWOPI);

  // the preceding correction fails if we have exactly eta2 - eta1 = 2*PI
  // it reduces the interval to zero length
  if (lambda2 - lambda1 > M_PI && ep->eta2 - ep->eta1 < M_PI) {
    ep->eta2 += TWOPI;
  }
}

typedef double erray_t[2][4][4];

// coefficients for error estimation
// while using cubic Bézier curves for approximation
// 0 < b/a < 1/4
static const erray_t coeffs3Low = {
    {{3.85268, -21.229, -0.330434, 0.0127842},
     {-1.61486, 0.706564, 0.225945, 0.263682},
     {-0.910164, 0.388383, 0.00551445, 0.00671814},
     {-0.630184, 0.192402, 0.0098871, 0.0102527}},
    {{-0.162211, 9.94329, 0.13723, 0.0124084},
     {-0.253135, 0.00187735, 0.0230286, 0.01264},
     {-0.0695069, -0.0437594, 0.0120636, 0.0163087},
     {-0.0328856, -0.00926032, -0.00173573, 0.00527385}}};

// coefficients for error estimation
// while using cubic Bézier curves for approximation
// 1/4 <= b/a <= 1
static const erray_t coeffs3High = {
    {{0.0899116, -19.2349, -4.11711, 0.183362},
     {0.138148, -1.45804, 1.32044, 1.38474},
     {0.230903, -0.450262, 0.219963, 0.414038},
     {0.0590565, -0.101062, 0.0430592, 0.0204699}},
    {{0.0164649, 9.89394, 0.0919496, 0.00760802},
     {0.0191603, -0.0322058, 0.0134667, -0.0825018},
     {0.0156192, -0.017535, 0.00326508, -0.228157},
     {-0.0236752, 0.0405821, -0.0173086, 0.176187}}};

// safety factor to convert the "best" error approximation
// into a "max bound" error
static const double safety3[] = {0.001, 4.98, 0.207, 0.0067};

/* Compute the value of a rational function.
 * This method handles rational functions where the numerator is
 * quadratic and the denominator is linear
 */
static double RationalFunction(double x, const double *c) {
  return (x * (x * c[0] + c[1]) + c[2]) / (x + c[3]);
}

/* Estimate the approximation error for a sub-arc of the instance.
 * tA and tB give the start and end angle of the subarc
 * Returns upper bound of the approximation error between the Bézier
 * curve and the real ellipse
 */
static double estimateError(ellipse_t *ep, double etaA, double etaB) {
  double c0, c1, eta = 0.5 * (etaA + etaB);

  double x = ep->b / ep->a;
  double dEta = etaB - etaA;
  double cos2 = cos(2 * eta);
  double cos4 = cos(4 * eta);
  double cos6 = cos(6 * eta);

  // select the right coefficient's set according to b/a
  const double(*coeffs)[4][4] = x < 0.25 ? coeffs3Low : coeffs3High;

  c0 = RationalFunction(x, coeffs[0][0]) +
       cos2 * RationalFunction(x, coeffs[0][1]) +
       cos4 * RationalFunction(x, coeffs[0][2]) +
       cos6 * RationalFunction(x, coeffs[0][3]);

  c1 = RationalFunction(x, coeffs[1][0]) +
       cos2 * RationalFunction(x, coeffs[1][1]) +
       cos4 * RationalFunction(x, coeffs[1][2]) +
       cos6 * RationalFunction(x, coeffs[1][3]);

  return RationalFunction(x, safety3) * ep->a * exp(c0 + c1 * dEta);
}

typedef LIST(pointf) bezier_path_t;

/* append points to a Bézier path
 * Assume initial call to moveTo to initialize, followed by
 * calls to curveTo and lineTo, and finished with endPath.
 */

static void moveTo(bezier_path_t *polypath, double x, double y) {
  LIST_APPEND(polypath, ((pointf){.x = x, .y = y}));
}

static void curveTo(bezier_path_t *polypath, double x1, double y1, double x2,
                    double y2, double x3, double y3) {
  LIST_APPEND(polypath, ((pointf){.x = x1, .y = y1}));
  LIST_APPEND(polypath, ((pointf){.x = x2, .y = y2}));
  LIST_APPEND(polypath, ((pointf){.x = x3, .y = y3}));
}

static void lineTo(bezier_path_t *polypath, double x, double y) {
  const pointf curp = LIST_GET(polypath, LIST_SIZE(polypath) - 1);
  curveTo(polypath, curp.x, curp.y, x, y, x, y);
}

static void endPath(bezier_path_t *polypath) {
  const pointf p0 = LIST_GET(polypath, 0);
  lineTo(polypath, p0.x, p0.y);
}

/* genEllipticPath:
 * Approximate an elliptical arc via Béziers of degree 3
 * The path begins and ends with line segments to the center of the ellipse.
 * Returned path must be freed by the caller.
 */
static Ppolyline_t *genEllipticPath(ellipse_t *ep) {
  Ppolyline_t *polypath = gv_alloc(sizeof(Ppolyline_t));

  static const double THRESHOLD = 0.00001; // quality of approximation

  // find the number of Bézier curves needed
  bool found = false;
  int i, n = 1;
  while (!found && n < 1024) {
    double diffEta = (ep->eta2 - ep->eta1) / n;
    if (diffEta <= 0.5 * M_PI) {
      double etaOne = ep->eta1;
      found = true;
      for (i = 0; found && i < n; ++i) {
        double etaA = etaOne;
        etaOne += diffEta;
        found = estimateError(ep, etaA, etaOne) <= THRESHOLD;
      }
    }
    n = n << 1;
  }

  const double dEta = (ep->eta2 - ep->eta1) / n;
  double etaB = ep->eta1;

  double cosEtaB = cos(etaB);
  double sinEtaB = sin(etaB);
  double aCosEtaB = ep->a * cosEtaB;
  double bSinEtaB = ep->b * sinEtaB;
  double aSinEtaB = ep->a * sinEtaB;
  double bCosEtaB = ep->b * cosEtaB;
  double xB = ep->cx + aCosEtaB;
  double yB = ep->cy + bSinEtaB;
  double xBDot = -aSinEtaB;
  double yBDot = bCosEtaB;

  bezier_path_t bezier_path = {0};
  moveTo(&bezier_path, ep->cx, ep->cy);
  lineTo(&bezier_path, xB, yB);

  const double t = tan(0.5 * dEta);
  const double alpha = sin(dEta) * (sqrt(4 + 3 * t * t) - 1) / 3;

  for (i = 0; i < n; ++i) {

    double xA = xB;
    double yA = yB;
    double xADot = xBDot;
    double yADot = yBDot;

    etaB += dEta;
    cosEtaB = cos(etaB);
    sinEtaB = sin(etaB);
    aCosEtaB = ep->a * cosEtaB;
    bSinEtaB = ep->b * sinEtaB;
    aSinEtaB = ep->a * sinEtaB;
    bCosEtaB = ep->b * cosEtaB;
    xB = ep->cx + aCosEtaB;
    yB = ep->cy + bSinEtaB;
    xBDot = -aSinEtaB;
    yBDot = bCosEtaB;

    curveTo(&bezier_path, xA + alpha * xADot, yA + alpha * yADot,
            xB - alpha * xBDot, yB - alpha * yBDot, xB, yB);
  }

  endPath(&bezier_path);

  LIST_DETACH(&bezier_path, &polypath->ps, &polypath->pn);

  return polypath;
}

/* ellipticWedge:
 * Return a cubic Bézier for an elliptical wedge, with center ctr, x and y
 * semi-axes xsemi and ysemi, start angle angle0 and end angle angle1.
 * This includes beginning and ending line segments to the ellipse center.
 * Calling function must free storage of returned path.
 */
Ppolyline_t *ellipticWedge(pointf ctr, double xsemi, double ysemi,
                           double angle0, double angle1) {
  ellipse_t ell;

  initEllipse(&ell, ctr.x, ctr.y, xsemi, ysemi, angle0, angle1);
  return genEllipticPath(&ell);
}
