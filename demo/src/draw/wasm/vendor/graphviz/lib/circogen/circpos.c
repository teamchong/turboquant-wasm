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
/* TODO:
 * If cut point is in exactly 2 blocks, expand block circles to overlap
 * especially in the case where one block is the sole child of the other.
 */

#include	<circogen/blockpath.h>
#include	<circogen/circpos.h>
#include	<circogen/nodelist.h>
#include	<math.h>
#include	<stddef.h>
#include	<util/alloc.h>

/* The function determines how much the block should be rotated
 * for best positioning with parent, assuming its center is at x and y
 * relative to the parent.
 * angle gives the angle of the new position, i.e., tan(angle) = y/x.
 * If sn has 2 nodes, we arrange the line of the 2 normal to angle.
 * If sn has 1 node, parent_pos has already been set to the 
 * correct angle assuming no rotation.
 * Otherwise, we find the node in sn connected to the parent and rotate
 * the block so that it is closer or at least visible to its node in the
 * parent.
 *
 * For COALESCED blocks, if neighbor is in left half plane, 
 * use unCOALESCED case.
 * Else let theta be angle, R = LEN(x,y), pho the radius of actual 
 * child block, phi be angle of neighbor in actual child block,
 * and r the distance from center of coalesced block to center of 
 * actual block. Then, the angle to rotate the coalesced block to
 * that the edge from the parent is tangent to the neighbor on the
 * actual child block circle is
 *    alpha = theta + M_PI/2 - phi - arcsin((l/R)*(sin B))
 * where l = r - rho/(cos phi) and beta = M_PI/2 + phi.
 * Thus, 
 *    alpha = theta + M_PI/2 - phi - arcsin((l/R)*(cos phi))
 */
static double getRotation(block_t *sn, double x, double y, double theta) {
    double mindist2;
    Agraph_t *subg;
    Agnode_t *n, *closest_node, *neighbor;
    double len2, newX, newY;

    subg = sn->sub_graph;

    nodelist_t *list = &sn->circle_list;

    if (sn->parent_pos >= 0) {
	theta += M_PI - sn->parent_pos;
	if (theta < 0)
	    theta += 2 * M_PI;

	return theta;
    }

    size_t count = nodelist_size(list);
    if (count == 2) {
	return theta - M_PI / 2.0;
    }

    /* Find node in block connected to block's parent */
    neighbor = CHILD(sn);
    newX = ND_pos(neighbor)[0] + x;
    newY = ND_pos(neighbor)[1] + y;
    mindist2 = LEN2(newX, newY);    /* save sqrts by using sqr of dist to find min */
    closest_node = neighbor;

    for (n = agfstnode(subg); n; n = agnxtnode(subg, n)) {
	if (n == neighbor)
	    continue;

	newX = ND_pos(n)[0] + x;
	newY = ND_pos(n)[1] + y;

	len2 = LEN2(newX, newY);
	if (len2 < mindist2) {
	    mindist2 = len2;
	    closest_node = n;
	}
    }

    if (neighbor != closest_node) {
	double rho = sn->rad0;
	double r = sn->radius - rho;
	double n_x = ND_pos(neighbor)[0];
	if (COALESCED(sn) && -r < n_x) {
	    const double R = hypot(x, y);
	    double n_y = ND_pos(neighbor)[1];
	    double phi = atan2(n_y, n_x + r);
	    double l = r - rho / cos(phi);

	    theta += M_PI / 2.0 - phi - asin(l / R * cos(phi));
	} else {		/* Origin still at center of this block */
	    double phi = atan2(ND_pos(neighbor)[1], ND_pos(neighbor)[0]);
	    theta += M_PI - phi - PSI(neighbor);
	    if (theta > 2 * M_PI)
		theta -= 2 * M_PI;
	}
    } else
	theta = 0;
    return theta;
}

/* Recursively apply rotation rotate followed by translation (x,y)
 * to block sn and its children.
 */
static void applyDelta(block_t * sn, double x, double y, double rotate)
{
    block_t *child;
    Agraph_t *subg;
    Agnode_t *n;

    subg = sn->sub_graph;

    for (n = agfstnode(subg); n; n = agnxtnode(subg, n)) {

	const double tmpX = ND_pos(n)[0];
	const double tmpY = ND_pos(n)[1];
	const double cosR = cos(rotate);
	const double sinR = sin(rotate);

	const double X = tmpX * cosR - tmpY * sinR;
	const double Y = tmpX * sinR + tmpY * cosR;

	/* translate */
	ND_pos(n)[0] = X + x;
	ND_pos(n)[1] = Y + y;
    }

    for (child = sn->children.first; child; child = child->next)
	applyDelta(child, x, y, rotate);
}

/* firstangle and lastangle give the range of child angles.
 * These are set and used only when a block has just 1 node.
 * And are used to give the center angle between the two extremes.
 * The parent will then be attached at PI - center angle (parent_pos).
 * If this block has no children, this is PI. Otherwise, positionChildren will
 * be called once with the blocks node. firstangle will be 0, with
 * succeeding angles increasing. 
 * position can always return the center angle - PI, since the block
 * must have children and if the block has 1 node, the limits will be
 * correctly set. If the block has more than 1 node, the value is
 * unused.
 */
typedef struct {
    double radius;		/* Basic radius of block */
    double subtreeR;		/* Max of subtree radii */
    double nodeAngle;		/* Angle allocated to each node in block */
    double firstAngle;		/* Smallest child angle when block has 1 node */
    double lastAngle;		/* Largest child angle when block has 1 node */
    block_t *cp;		/* Children of block */
    node_t *neighbor;		/* Node connected to parent block, if any */
} posstate;

typedef struct {
    Agnode_t* n;
    double theta;        /* angle of node */
    double minRadius;    /* minimum radius for child circle */
    double maxRadius;    /* maximum radius of child blocks */
    double diameter;     /* length of arc needed for child blocks */
    double scale;        /* scale factor to increase minRadius to parents' children don't overlap */
    int childCount;      /* no. of child blocks attached at n */
} posinfo_t;

/// get size info for blocks attached to the given node.
static double
getInfo (posinfo_t* pi, posstate * stp, double min_dist)
{
    block_t *child;
    double maxRadius = 0;	/* Max. radius of children */
    double diameter = 0;	/* sum of child diameters */
    int childCount = 0;

    for (child = stp->cp; child; child = child->next) {
	if (BLK_PARENT(child) == pi->n) {
	    childCount++;
	    if (maxRadius < child->radius) {
		maxRadius = child->radius;
	    }
	    diameter += 2 * child->radius + min_dist;
	}
    }

    pi->diameter = diameter;
    pi->childCount = childCount;
    pi->minRadius = stp->radius + min_dist + maxRadius;
    pi->maxRadius = maxRadius;
    return maxRadius;
}

static void
setInfo (posinfo_t* p0, posinfo_t* p1, double delta)
{
    double t = p0->diameter * p1->minRadius + p1->diameter * p0->minRadius;

    t /= 2*delta*p0->minRadius*p1->minRadius;

    t = fmax(t, 1);

    p0->scale = fmax(p0->scale, t);
    p1->scale = fmax(p1->scale, t);
}

static void positionChildren(posinfo_t *info, posstate *stp, size_t length,
                             double min_dist) {
    block_t *child;
    double childAngle, childRadius, incidentAngle;
    double mindistAngle, rotateAngle, midAngle = 0.0;
    int midChild, cnt = 0;
    double snRadius = stp->subtreeR;	/* max subtree radius */
    double firstAngle = stp->firstAngle;
    double lastAngle = stp->lastAngle;
    double d, deltaX, deltaY;

    childRadius = info->scale * info->minRadius;
    if (length == 1) {
	childAngle = 0;
	d = info->diameter / (2 * M_PI);
	childRadius = fmax(childRadius, d);
	d = 2 * M_PI * childRadius - info->diameter;
	if (d > 0)
	    min_dist += d / info->childCount;
    }
    else
	childAngle = info->theta - info->diameter / (2 * childRadius);

    if ((childRadius + info->maxRadius) > snRadius)
	snRadius = childRadius + info->maxRadius;

    mindistAngle = min_dist / childRadius;

    midChild = (info->childCount + 1) / 2;
    for (child = stp->cp; child; child = child->next) {
	if (BLK_PARENT(child) != info->n)
	    continue;
	if (nodelist_is_empty(&child->circle_list))
	    continue;

	incidentAngle = child->radius / childRadius;
	if (length == 1) {
	    if (childAngle != 0) {
		if (info->childCount == 2)
		    childAngle = M_PI;
		else
		    childAngle += incidentAngle;
	    }

	    if (firstAngle < 0)
		firstAngle = childAngle;

	    lastAngle = childAngle;
	} else {
	    if (info->childCount == 1) {
		childAngle = info->theta;
	    } else {
		childAngle += incidentAngle + mindistAngle / 2;
	    }
	}

	deltaX = childRadius * cos(childAngle);
	deltaY = childRadius * sin(childAngle);

	/* first apply the delta to the immediate child and see if we need
	 * to rotate it for better edge link                                            
	 * should return the theta value if there was a rotation else zero
	 */

	rotateAngle = getRotation(child, deltaX, deltaY, childAngle);
	applyDelta(child, deltaX, deltaY, rotateAngle);

	if (length == 1) {
	    childAngle += incidentAngle + mindistAngle;
	} else {
	    childAngle += incidentAngle + mindistAngle / 2;
	}
	cnt++;
	if (cnt == midChild)
	    midAngle = childAngle;
    }

    if (length > 1 && info->n == stp->neighbor) {
	PSI(info->n) = midAngle;
    }

    stp->subtreeR = snRadius;
    stp->firstAngle = firstAngle;
    stp->lastAngle = lastAngle;
}

/* Assume childCount > 0
 * For each node in the block with children, getInfo is called, with the
 * information stored in the parents array.
 * This information is used by setInfo to compute the amount of space allocated
 * to each parent and the radius at which to place its children.
 * Finally, positionChildren is called to do the actual positioning.
 * If length is 1, keeps track of minimum and maximum child angle.
 */
static double position(size_t childCount, size_t length, nodelist_t *nodepath,
	 block_t * sn, double min_dist)
{
    posstate state;
    int i, counter = 0;
    double maxRadius = 0.0;
    double angle;
    double theta = 0.0;
    posinfo_t* parents = gv_calloc(childCount, sizeof(posinfo_t));
    int num_parents = 0;
    posinfo_t* next;
    posinfo_t* curr;
    double delta;

    state.cp = sn->children.first;
    state.subtreeR = sn->radius;
    state.radius = sn->radius;
    state.neighbor = CHILD(sn);
    state.nodeAngle = 2 * M_PI / (double)length;
    state.firstAngle = -1;
    state.lastAngle = -1;

    for (size_t item = 0; item < nodelist_size(nodepath); ++item) {
	Agnode_t *n = nodelist_get(nodepath, item);

	theta = counter * state.nodeAngle;
	counter++;

	if (ISPARENT(n)) {
	    parents[num_parents].n = n;
	    parents[num_parents].theta = theta;
	    maxRadius = getInfo (parents+num_parents, &state, min_dist);
	    num_parents++;
	}
    }

    if (num_parents == 1)
	parents->scale = 1.0;
    else if (num_parents == 2) {
	curr = parents;
	next = parents+1;
	delta = next->theta - curr->theta;
        if (delta > M_PI)
	    delta = 2*M_PI - delta;
	setInfo (curr, next, delta);
    }
    else {
	curr = parents;
	for (i = 0; i < num_parents; i++) {
	    if (i+1 == num_parents) {
		next = parents;
		delta = next->theta - curr->theta + 2*M_PI; 
	    }
	    else {
		next = curr+1;
		delta = next->theta - curr->theta; 
	    }
	    setInfo (curr, next, delta);
	    curr++;
	}
    }

    for (i = 0; i < num_parents; i++) {
	positionChildren(parents + i, &state, length, min_dist);
    }

    free (parents);

    /* If block has only 1 child, to save space, we coalesce it with the
     * child. Instead of having final radius sn->radius + max child radius,
     * we have half that. However, the origin of the block is no longer in
     * the center of the block, so we cannot do a simple rotation to get
     * the neighbor node next to the parent block in getRotate.
     */
    if (childCount == 1) {
	applyDelta(sn, -(maxRadius + min_dist / 2), 0, 0);
	sn->radius += min_dist / 2 + maxRadius;
	SET_COALESCED(sn);
    } else
	sn->radius = state.subtreeR;

    angle = (state.firstAngle + state.lastAngle) / 2.0 - M_PI;
    return angle;
}

/// Set positions of block sn and its child blocks.
///
/// @param state Context containing a counter to use for graph copy naming
static void doBlock(Agraph_t *g, block_t *sn, double min_dist,
                    circ_state *state) {
    block_t *child;
    double centerAngle = M_PI;

    /* layout child subtrees */
    size_t childCount = 0;
    for (child = sn->children.first; child; child = child->next) {
	doBlock(g, child, min_dist, state);
	childCount++;
    }

    /* layout this block */
    nodelist_t longest_path = layout_block(g, sn, min_dist, state);
    sn->circle_list = longest_path;
    size_t length = nodelist_size(&longest_path); // path contains everything in block

    /* attach children */
    if (childCount > 0)
	centerAngle = position(childCount, length, &longest_path, sn, min_dist);

    if (length == 1 && BLK_PARENT(sn)) {
	sn->parent_pos = centerAngle;
	if (sn->parent_pos < 0)
	    sn->parent_pos += 2 * M_PI;
    }
}

void circPos(Agraph_t * g, block_t * sn, circ_state * state)
{
  doBlock(g, sn, state->min_dist, state);
}
