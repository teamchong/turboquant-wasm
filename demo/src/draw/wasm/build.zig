const std = @import("std");

const graphviz_c_sources = [_][]const u8{
    "lib/cdt/dtclose.c",
    "lib/cdt/dtdisc.c",
    "lib/cdt/dtextract.c",
    "lib/cdt/dtflatten.c",
    "lib/cdt/dthash.c",
    "lib/cdt/dtmethod.c",
    "lib/cdt/dtopen.c",
    "lib/cdt/dtrenew.c",
    "lib/cdt/dtrestore.c",
    "lib/cdt/dtsize.c",
    "lib/cdt/dtstat.c",
    "lib/cdt/dtstrhash.c",
    "lib/cdt/dttree.c",
    "lib/cdt/dtview.c",
    "lib/cdt/dtwalk.c",
    "lib/cgraph/acyclic.c",
    "lib/cgraph/agerror.c",
    "lib/cgraph/apply.c",
    "lib/cgraph/attr.c",
    "lib/cgraph/edge.c",
    "lib/cgraph/graph.c",
    "lib/cgraph/id.c",
    "lib/cgraph/imap.c",
    "lib/cgraph/io.c",
    "lib/cgraph/node_induce.c",
    "lib/cgraph/node.c",
    "lib/cgraph/obj.c",
    "lib/cgraph/rec.c",
    "lib/cgraph/refstr.c",
    "lib/cgraph/subg.c",
    "lib/cgraph/tred.c",
    "lib/cgraph/unflatten.c",
    "lib/cgraph/utils.c",
    "lib/cgraph/write.c",
    "lib/common/arrows.c",
    "lib/common/colxlate.c",
    "lib/common/ellipse.c",
    "lib/common/geom.c",
    "lib/common/globals.c",
    "lib/common/input.c",
    "lib/common/labels.c",
    "lib/common/ns.c",
    "lib/common/pointset.c",
    "lib/common/postproc.c",
    "lib/common/routespl.c",
    "lib/common/shapes.c",
    "lib/common/splines.c",
    "lib/common/taper.c",
    "lib/common/textspan_lut.c",
    "lib/common/textspan.c",
    "lib/common/timing.c",
    "lib/common/utils.c",
    "lib/dotgen/acyclic.c",
    "lib/dotgen/aspect.c",
    "lib/dotgen/class1.c",
    "lib/dotgen/class2.c",
    "lib/dotgen/cluster.c",
    "lib/dotgen/compound.c",
    "lib/dotgen/conc.c",
    "lib/dotgen/decomp.c",
    "lib/dotgen/dotinit.c",
    "lib/dotgen/dotsplines.c",
    "lib/dotgen/fastgr.c",
    "lib/dotgen/flat.c",
    "lib/dotgen/mincross.c",
    "lib/dotgen/position.c",
    "lib/dotgen/rank.c",
    "lib/dotgen/sameport.c",
    "lib/gvc/gvc.c",
    "lib/gvc/gvcontext.c",
    "lib/gvc/gvjobs.c",
    "lib/gvc/gvlayout.c",
    "lib/gvc/gvplugin.c",
    "lib/label/index.c",
    "lib/label/node.c",
    "lib/label/rectangle.c",
    "lib/label/split.q.c",
    "lib/label/xlabels.c",
    "lib/ortho/fPQ.c",
    "lib/ortho/maze.c",
    "lib/ortho/ortho.c",
    "lib/ortho/partition.c",
    "lib/ortho/rawgraph.c",
    "lib/ortho/sgraph.c",
    "lib/ortho/trapezoid.c",
    "lib/pack/ccomps.c",
    "lib/pack/pack.c",
    "lib/pathplan/cvt.c",
    "lib/pathplan/inpoly.c",
    "lib/pathplan/route.c",
    "lib/pathplan/shortest.c",
    "lib/pathplan/shortestpth.c",
    "lib/pathplan/solvers.c",
    "lib/pathplan/triang.c",
    "lib/pathplan/util.c",
    "lib/pathplan/visibility.c",
    "lib/util/arena.c",
    "lib/util/base64.c",
    "lib/util/list.c",
    "lib/util/random.c",
    "lib/util/xml.c",
    "lib/xdot/xdot.c",
    // neato layout engine (needed for nop2 mode — pre-positioned edge routing)
    "lib/neatogen/adjust.c",
    "lib/neatogen/bfs.c",
    "lib/neatogen/call_tri.c",
    "lib/neatogen/circuit.c",
    "lib/neatogen/compute_hierarchy.c",
    "lib/neatogen/conjgrad.c",
    "lib/neatogen/constrained_majorization_ipsep.c",
    "lib/neatogen/constrained_majorization.c",
    "lib/neatogen/delaunay.c",
    "lib/neatogen/dijkstra.c",
    "lib/neatogen/edges.c",
    "lib/neatogen/embed_graph.c",
    "lib/neatogen/geometry.c",
    "lib/neatogen/heap.c",
    "lib/neatogen/hedges.c",
    "lib/neatogen/info.c",
    "lib/neatogen/kkutils.c",
    "lib/neatogen/legal.c",
    "lib/neatogen/lu.c",
    "lib/neatogen/matinv.c",
    "lib/neatogen/matrix_ops.c",
    "lib/neatogen/memory.c",
    "lib/neatogen/multispline.c",
    "lib/neatogen/neatoinit.c",
    "lib/neatogen/neatosplines.c",
    "lib/neatogen/opt_arrangement.c",
    "lib/neatogen/overlap.c",
    "lib/neatogen/pca.c",
    "lib/neatogen/poly.c",
    "lib/neatogen/quad_prog_vpsc.c",
    "lib/neatogen/randomkit.c",
    "lib/neatogen/sgd.c",
    "lib/neatogen/site.c",
    "lib/neatogen/smart_ini_x.c",
    "lib/neatogen/solve.c",
    "lib/neatogen/stress.c",
    "lib/neatogen/stuff.c",
    "lib/neatogen/voronoi.c",
    // neato deps (minimal — only what neato core needs)
    "lib/sparse/general.c",
    "lib/sparse/SparseMatrix.c",
    "lib/rbtree/red_black_tree.c",
    // neato layout plugin
    "plugin/neato_layout/gvlayout_neato_layout.c",
    "plugin/neato_layout/gvplugin_neato_layout.c",
    "plugin/dot_layout/gvlayout_dot_layout.c",
    "plugin/dot_layout/gvplugin_dot_layout.c",
    // circo layout engine (radial layout for cyclic graphs / state machines)
    "lib/circogen/block.c",
    "lib/circogen/blockpath.c",
    "lib/circogen/blocktree.c",
    "lib/circogen/circpos.c",
    "lib/circogen/circular.c",
    "lib/circogen/circularinit.c",
    "lib/circogen/edgelist.c",
    "lib/circogen/nodelist.c",
    "plugin/circo_layout/gvlayout_circo_layout.c",
    "plugin/circo_layout/gvplugin_circo_layout.c",
    "wasm_platform/gviz_bridge.c",
    "wasm_platform/neato_compat.c",
};

const graphviz_c_flags = [_][]const u8{
    "-DHAVE_CONFIG_H",
    "-DNONDLL",
    "-D_XOPEN_SOURCE=700",
    "-std=c11",
    "-O2",
    "-fno-strict-aliasing",
    // Suppress warnings in vendored code
    "-Wno-unused-parameter",
    "-Wno-sign-compare",
    "-Wno-implicit-function-declaration",
    "-Wno-incompatible-pointer-types",
    "-Wno-pointer-sign",
    "-Wno-unused-variable",
    "-Wno-missing-field-initializers",
    "-Wno-int-conversion",
    "-Wno-unused-but-set-variable",
};

pub fn build(b: *std.Build) void {
    const enable_validation = b.option(bool, "enable_validation", "Include element validation export (default: true)") orelse true;
    const enable_compression = b.option(bool, "enable_compression", "Include zlib compression export (default: true)") orelse true;

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .wasi,
    });

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_validation", enable_validation);
    build_options.addOption(bool, "enable_compression", enable_compression);

    const wasm = b.addExecutable(.{
        .name = "drawmode",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = wasm_target,
            .optimize = .ReleaseSmall,
            .link_libc = true,
        }),
    });
    wasm.root_module.addOptions("build_options", build_options);
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.export_memory = true;

    // Add Graphviz C source files
    const graphviz_root = b.path("vendor/graphviz");
    wasm.addCSourceFiles(.{
        .root = graphviz_root,
        .files = &graphviz_c_sources,
        .flags = &graphviz_c_flags,
    });

    // Include paths for Graphviz headers:
    // - vendor/graphviz/ for "config.h"
    // - vendor/graphviz/lib/ for <cgraph/cgraph.h>, <gvc/gvc.h>, etc.
    // - per-library dirs for internal includes like <cgraph.h>, <cdt.h>, <pathgeom.h>
    wasm.addIncludePath(b.path("vendor/graphviz"));
    wasm.addIncludePath(b.path("vendor/graphviz/lib"));
    wasm.addIncludePath(b.path("vendor/graphviz/plugin"));
    // Internal headers reference siblings without directory prefix
    const lib_subdirs = [_][]const u8{
        "vendor/graphviz/lib/cdt",
        "vendor/graphviz/lib/cgraph",
        "vendor/graphviz/lib/common",
        "vendor/graphviz/lib/dotgen",
        "vendor/graphviz/lib/gvc",
        "vendor/graphviz/lib/label",
        "vendor/graphviz/lib/ortho",
        "vendor/graphviz/lib/pack",
        "vendor/graphviz/lib/pathplan",
        "vendor/graphviz/lib/util",
        "vendor/graphviz/lib/xdot",
        "vendor/graphviz/lib/fdpgen",
        "vendor/graphviz/lib/neatogen",
        "vendor/graphviz/lib/sparse",
        "vendor/graphviz/lib/rbtree",
        "vendor/graphviz/lib/circogen",
        "vendor/graphviz/lib/twopigen",
        "vendor/graphviz/lib/sfdpgen",
    };
    for (lib_subdirs) |subdir| {
        wasm.addIncludePath(b.path(subdir));
    }


    const install_wasm = b.addInstallArtifact(wasm, .{});
    b.getInstallStep().dependOn(&install_wasm.step);

    const wasm_step = b.step("wasm", "Build WASM module");
    wasm_step.dependOn(&install_wasm.step);

    // Native tests (Zig only, no Graphviz C for native tests)
    const target = b.standardTargetOptions(.{});
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
        }),
    });
    tests.root_module.addOptions("build_options", build_options);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
