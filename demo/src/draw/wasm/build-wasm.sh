#!/bin/bash
# Direct `zig build-exe` invocation — bypasses Zig's build.zig runner because
# the build-runner itself can't link on macOS Sequoia with Zig 0.15.2. We call
# the compiler directly with every graphviz C file and the right include paths.
#
# Output: zig-out/bin/drawmode.wasm

set -e
cd "$(dirname "$0")"

mkdir -p zig-out/bin

# Collect every graphviz .c file the build.zig listed. Using find+grep to
# generate the list at runtime so adding/removing a .c doesn't require
# touching this script.
C_SOURCES=(
  vendor/graphviz/lib/cdt/dtclose.c
  vendor/graphviz/lib/cdt/dtdisc.c
  vendor/graphviz/lib/cdt/dtextract.c
  vendor/graphviz/lib/cdt/dtflatten.c
  vendor/graphviz/lib/cdt/dthash.c
  vendor/graphviz/lib/cdt/dtmethod.c
  vendor/graphviz/lib/cdt/dtopen.c
  vendor/graphviz/lib/cdt/dtrenew.c
  vendor/graphviz/lib/cdt/dtrestore.c
  vendor/graphviz/lib/cdt/dtsize.c
  vendor/graphviz/lib/cdt/dtstat.c
  vendor/graphviz/lib/cdt/dtstrhash.c
  vendor/graphviz/lib/cdt/dttree.c
  vendor/graphviz/lib/cdt/dtview.c
  vendor/graphviz/lib/cdt/dtwalk.c
  vendor/graphviz/lib/cgraph/acyclic.c
  vendor/graphviz/lib/cgraph/agerror.c
  vendor/graphviz/lib/cgraph/apply.c
  vendor/graphviz/lib/cgraph/attr.c
  vendor/graphviz/lib/cgraph/edge.c
  vendor/graphviz/lib/cgraph/graph.c
  vendor/graphviz/lib/cgraph/id.c
  vendor/graphviz/lib/cgraph/imap.c
  vendor/graphviz/lib/cgraph/io.c
  vendor/graphviz/lib/cgraph/node_induce.c
  vendor/graphviz/lib/cgraph/node.c
  vendor/graphviz/lib/cgraph/obj.c
  vendor/graphviz/lib/cgraph/rec.c
  vendor/graphviz/lib/cgraph/refstr.c
  vendor/graphviz/lib/cgraph/subg.c
  vendor/graphviz/lib/cgraph/tred.c
  vendor/graphviz/lib/cgraph/unflatten.c
  vendor/graphviz/lib/cgraph/utils.c
  vendor/graphviz/lib/cgraph/write.c
  vendor/graphviz/lib/common/arrows.c
  vendor/graphviz/lib/common/colxlate.c
  vendor/graphviz/lib/common/ellipse.c
  vendor/graphviz/lib/common/geom.c
  vendor/graphviz/lib/common/globals.c
  vendor/graphviz/lib/common/input.c
  vendor/graphviz/lib/common/labels.c
  vendor/graphviz/lib/common/ns.c
  vendor/graphviz/lib/common/pointset.c
  vendor/graphviz/lib/common/postproc.c
  vendor/graphviz/lib/common/routespl.c
  vendor/graphviz/lib/common/shapes.c
  vendor/graphviz/lib/common/splines.c
  vendor/graphviz/lib/common/taper.c
  vendor/graphviz/lib/common/textspan_lut.c
  vendor/graphviz/lib/common/textspan.c
  vendor/graphviz/lib/common/timing.c
  vendor/graphviz/lib/common/utils.c
  vendor/graphviz/lib/dotgen/acyclic.c
  vendor/graphviz/lib/dotgen/aspect.c
  vendor/graphviz/lib/dotgen/class1.c
  vendor/graphviz/lib/dotgen/class2.c
  vendor/graphviz/lib/dotgen/cluster.c
  vendor/graphviz/lib/dotgen/compound.c
  vendor/graphviz/lib/dotgen/conc.c
  vendor/graphviz/lib/dotgen/decomp.c
  vendor/graphviz/lib/dotgen/dotinit.c
  vendor/graphviz/lib/dotgen/dotsplines.c
  vendor/graphviz/lib/dotgen/fastgr.c
  vendor/graphviz/lib/dotgen/flat.c
  vendor/graphviz/lib/dotgen/mincross.c
  vendor/graphviz/lib/dotgen/position.c
  vendor/graphviz/lib/dotgen/rank.c
  vendor/graphviz/lib/dotgen/sameport.c
  vendor/graphviz/lib/gvc/gvc.c
  vendor/graphviz/lib/gvc/gvcontext.c
  vendor/graphviz/lib/gvc/gvjobs.c
  vendor/graphviz/lib/gvc/gvlayout.c
  vendor/graphviz/lib/gvc/gvplugin.c
  vendor/graphviz/lib/label/index.c
  vendor/graphviz/lib/label/node.c
  vendor/graphviz/lib/label/rectangle.c
  vendor/graphviz/lib/label/split.q.c
  vendor/graphviz/lib/label/xlabels.c
  vendor/graphviz/lib/ortho/fPQ.c
  vendor/graphviz/lib/ortho/maze.c
  vendor/graphviz/lib/ortho/ortho.c
  vendor/graphviz/lib/ortho/partition.c
  vendor/graphviz/lib/ortho/rawgraph.c
  vendor/graphviz/lib/ortho/sgraph.c
  vendor/graphviz/lib/ortho/trapezoid.c
  vendor/graphviz/lib/pack/ccomps.c
  vendor/graphviz/lib/pack/pack.c
  vendor/graphviz/lib/pathplan/cvt.c
  vendor/graphviz/lib/pathplan/inpoly.c
  vendor/graphviz/lib/pathplan/route.c
  vendor/graphviz/lib/pathplan/shortest.c
  vendor/graphviz/lib/pathplan/shortestpth.c
  vendor/graphviz/lib/pathplan/solvers.c
  vendor/graphviz/lib/pathplan/triang.c
  vendor/graphviz/lib/pathplan/util.c
  vendor/graphviz/lib/pathplan/visibility.c
  vendor/graphviz/lib/util/arena.c
  vendor/graphviz/lib/util/base64.c
  vendor/graphviz/lib/util/list.c
  vendor/graphviz/lib/util/random.c
  vendor/graphviz/lib/util/xml.c
  vendor/graphviz/lib/xdot/xdot.c
  vendor/graphviz/lib/neatogen/adjust.c
  vendor/graphviz/lib/neatogen/bfs.c
  vendor/graphviz/lib/neatogen/call_tri.c
  vendor/graphviz/lib/neatogen/circuit.c
  vendor/graphviz/lib/neatogen/compute_hierarchy.c
  vendor/graphviz/lib/neatogen/conjgrad.c
  vendor/graphviz/lib/neatogen/constrained_majorization_ipsep.c
  vendor/graphviz/lib/neatogen/constrained_majorization.c
  vendor/graphviz/lib/neatogen/delaunay.c
  vendor/graphviz/lib/neatogen/dijkstra.c
  vendor/graphviz/lib/neatogen/edges.c
  vendor/graphviz/lib/neatogen/embed_graph.c
  vendor/graphviz/lib/neatogen/geometry.c
  vendor/graphviz/lib/neatogen/heap.c
  vendor/graphviz/lib/neatogen/hedges.c
  vendor/graphviz/lib/neatogen/info.c
  vendor/graphviz/lib/neatogen/kkutils.c
  vendor/graphviz/lib/neatogen/legal.c
  vendor/graphviz/lib/neatogen/lu.c
  vendor/graphviz/lib/neatogen/matinv.c
  vendor/graphviz/lib/neatogen/matrix_ops.c
  vendor/graphviz/lib/neatogen/memory.c
  vendor/graphviz/lib/neatogen/multispline.c
  vendor/graphviz/lib/neatogen/neatoinit.c
  vendor/graphviz/lib/neatogen/neatosplines.c
  vendor/graphviz/lib/neatogen/opt_arrangement.c
  vendor/graphviz/lib/neatogen/overlap.c
  vendor/graphviz/lib/neatogen/pca.c
  vendor/graphviz/lib/neatogen/poly.c
  vendor/graphviz/lib/neatogen/quad_prog_vpsc.c
  vendor/graphviz/lib/neatogen/randomkit.c
  vendor/graphviz/lib/neatogen/sgd.c
  vendor/graphviz/lib/neatogen/site.c
  vendor/graphviz/lib/neatogen/smart_ini_x.c
  vendor/graphviz/lib/neatogen/solve.c
  vendor/graphviz/lib/neatogen/stress.c
  vendor/graphviz/lib/neatogen/stuff.c
  vendor/graphviz/lib/neatogen/voronoi.c
  vendor/graphviz/lib/sparse/general.c
  vendor/graphviz/lib/sparse/SparseMatrix.c
  vendor/graphviz/lib/rbtree/red_black_tree.c
  vendor/graphviz/plugin/neato_layout/gvlayout_neato_layout.c
  vendor/graphviz/plugin/neato_layout/gvplugin_neato_layout.c
  vendor/graphviz/plugin/dot_layout/gvlayout_dot_layout.c
  vendor/graphviz/plugin/dot_layout/gvplugin_dot_layout.c
  vendor/graphviz/lib/circogen/block.c
  vendor/graphviz/lib/circogen/blockpath.c
  vendor/graphviz/lib/circogen/blocktree.c
  vendor/graphviz/lib/circogen/circpos.c
  vendor/graphviz/lib/circogen/circular.c
  vendor/graphviz/lib/circogen/circularinit.c
  vendor/graphviz/lib/circogen/edgelist.c
  vendor/graphviz/lib/circogen/nodelist.c
  vendor/graphviz/plugin/circo_layout/gvlayout_circo_layout.c
  vendor/graphviz/plugin/circo_layout/gvplugin_circo_layout.c
  vendor/graphviz/wasm_platform/gviz_bridge.c
  vendor/graphviz/wasm_platform/neato_compat.c
)

INCLUDES=(
  -Ivendor/graphviz
  -Ivendor/graphviz/lib
  -Ivendor/graphviz/plugin
  -Ivendor/graphviz/lib/cdt
  -Ivendor/graphviz/lib/cgraph
  -Ivendor/graphviz/lib/common
  -Ivendor/graphviz/lib/dotgen
  -Ivendor/graphviz/lib/gvc
  -Ivendor/graphviz/lib/label
  -Ivendor/graphviz/lib/ortho
  -Ivendor/graphviz/lib/pack
  -Ivendor/graphviz/lib/pathplan
  -Ivendor/graphviz/lib/util
  -Ivendor/graphviz/lib/xdot
  -Ivendor/graphviz/lib/fdpgen
  -Ivendor/graphviz/lib/neatogen
  -Ivendor/graphviz/lib/sparse
  -Ivendor/graphviz/lib/rbtree
  -Ivendor/graphviz/lib/circogen
  -Ivendor/graphviz/lib/twopigen
  -Ivendor/graphviz/lib/sfdpgen
)

CFLAGS=(
  -DHAVE_CONFIG_H
  -DNONDLL
  -D_XOPEN_SOURCE=700
  -std=c11
  -O2
  -fno-strict-aliasing
  -Wno-unused-parameter
  -Wno-sign-compare
  -Wno-implicit-function-declaration
  -Wno-incompatible-pointer-types
  -Wno-pointer-sign
  -Wno-unused-variable
  -Wno-missing-field-initializers
  -Wno-int-conversion
  -Wno-unused-but-set-variable
)

echo "build-wasm: compiling ${#C_SOURCES[@]} C sources + Zig entry..."

zig build-exe \
  -target wasm32-wasi \
  -O ReleaseSmall \
  -fstrip \
  -fno-entry \
  -rdynamic \
  -lc \
  --name drawmode \
  -femit-bin=zig-out/bin/drawmode.wasm \
  "${INCLUDES[@]}" \
  -cflags "${CFLAGS[@]}" -- \
  "${C_SOURCES[@]}" \
  src/main.zig

echo "build-wasm: $(ls -la zig-out/bin/drawmode.wasm)"
