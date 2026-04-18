const std = @import("std");
const util = @import("util.zig");

/// Graphviz C FFI — calls the statically-linked Graphviz dot layout engine
/// via gviz_bridge.c (which wraps the Graphviz C API in bitfield-free types).
///
/// Input JSON (nodes):
///   [{"id":"box_1","width":180,"height":80,"row":0,"col":1}, ...]
/// Input JSON (edges):
///   [{"from":"box_1","to":"box_2"}, ...]
/// Output JSON:
///   {"nodes":[{"id":"box_1","x":340,"y":100},...],
///    "edges":[{"from":"box_1","to":"box_2","points":[[x,y],[x,y],...]},...]}
pub fn layoutGraph(nodes_json: []const u8, edges_json: []const u8, groups_json: []const u8, opts_json: []const u8, out: []u8) !usize {
    var nodes: [MAX_NODES]Node = undefined;
    var node_count: usize = 0;
    node_count = parseNodes(nodes_json, &nodes) catch return 0;

    var edges: [MAX_EDGES]Edge = undefined;
    var edge_count: usize = 0;
    edge_count = parseEdges(edges_json, &edges) catch return 0;

    var groups: [MAX_GROUPS]Group = undefined;
    var group_count: usize = 0;
    if (groups_json.len > 2) { // not empty "[]"
        // Groups are optional — skip silently if JSON is malformed
        group_count = parseGroups(groups_json, &groups) catch 0;
    }

    // Build graph programmatically via cgraph API (no DOT parser needed)
    const graph = c.gviz_graph_new("G") orelse return 0;

    // Set graph attributes — parse rankdir from options
    var rankdir_buf: [8]u8 = undefined;
    const rankdir_slice = extractStringField(opts_json, "rankdir") orelse "TB";
    const rd_len = @min(rankdir_slice.len, rankdir_buf.len - 1);
    @memcpy(rankdir_buf[0..rd_len], rankdir_slice[0..rd_len]);
    rankdir_buf[rd_len] = 0;
    c.gviz_set_graph_attr(graph, "rankdir", @ptrCast(&rankdir_buf));

    // Read engine from options: "dot" (default), "nop2" (pre-positioned edge
    // routing), or "circo" (radial layout for cyclic graphs).
    const engine_slice = extractStringField(opts_json, "engine") orelse "dot";
    const use_nop2 = std.mem.eql(u8, engine_slice, "nop2");
    const use_circo = std.mem.eql(u8, engine_slice, "circo");

    // Spline routing:
    //   - circo: `true` gives curved splines that route around obstacles; the
    //     default would be straight lines center-to-center, which pass through
    //     other circle nodes. `mindist` controls the minimum radius.
    //   - dot LR/RL: `true` (curved splines, avoids long backward ortho curls)
    //   - dot TB/BT: `ortho` (crisp right-angle routing)
    const is_horizontal = std.mem.eql(u8, rankdir_slice, "LR") or std.mem.eql(u8, rankdir_slice, "RL");
    if (use_circo) {
        c.gviz_set_graph_attr(graph, "splines", "true");
        c.gviz_set_graph_attr(graph, "mindist", "1.5");
        c.gviz_set_graph_attr(graph, "overlap", "false");
    } else if (is_horizontal) {
        c.gviz_set_graph_attr(graph, "splines", "true");
    } else {
        c.gviz_set_graph_attr(graph, "splines", "ortho");
    }
    c.gviz_set_graph_attr(graph, "nodesep", "1.5");
    c.gviz_set_graph_attr(graph, "ranksep", "1.2");

    // Set default node attributes so per-node agsafeset works
    c.gviz_set_default_node_attr(graph, "width", "");
    c.gviz_set_default_node_attr(graph, "height", "");
    c.gviz_set_default_node_attr(graph, "fixedsize", "");
    if (use_nop2) {
        c.gviz_set_default_node_attr(graph, "pos", "");
    }

    // Set default edge attributes so per-edge agsafeset works
    c.gviz_set_default_edge_attr(graph, "label", "");
    c.gviz_set_default_edge_attr(graph, "constraint", "");
    c.gviz_set_default_edge_attr(graph, "style", "");

    // Create nodes with size attributes
    var node_ptrs: [MAX_NODES]?*anyopaque = undefined;
    for (nodes[0..node_count], 0..) |n, i| {
        const name_z = nullTerminate(n.id_slice) orelse {
            c.gviz_graph_close(graph);
            return 0;
        };

        const node_ptr = c.gviz_add_node(graph, name_z) orelse {
            c.gviz_graph_close(graph);
            return 0;
        };
        node_ptrs[i] = node_ptr;

        // Set width/height in inches (72 points per inch)
        var width_buf: [32]u8 = undefined;
        const width_len = writeFloat(&width_buf, @as(f64, @floatFromInt(n.width)) / 72.0);
        width_buf[width_len] = 0;
        c.gviz_set_node_attr(graph, node_ptr, "width", @ptrCast(&width_buf));

        var height_buf: [32]u8 = undefined;
        const height_len = writeFloat(&height_buf, @as(f64, @floatFromInt(n.height)) / 72.0);
        height_buf[height_len] = 0;
        c.gviz_set_node_attr(graph, node_ptr, "height", @ptrCast(&height_buf));

        c.gviz_set_node_attr(graph, node_ptr, "fixedsize", "true");
    }

    // Create rank=same subgraphs for nodes with explicit row values.
    // This ensures nodes at the same row share a Graphviz rank (same vertical position in TB).
    // Within each row, invisible edges enforce left-to-right column ordering.
    if (!use_nop2) {
        // Collect distinct row values
        var row_vals: [MAX_NODES]i32 = undefined;
        var row_val_count: usize = 0;
        for (nodes[0..node_count]) |n| {
            if (n.row) |r| {
                var found = false;
                for (row_vals[0..row_val_count]) |rv| {
                    if (rv == r) {
                        found = true;
                        break;
                    }
                }
                if (!found and row_val_count < MAX_NODES) {
                    row_vals[row_val_count] = r;
                    row_val_count += 1;
                }
            }
        }

        // Sort row values ascending (insertion sort)
        if (row_val_count > 1) {
            for (1..row_val_count) |ii| {
                const key = row_vals[ii];
                var jj: usize = ii;
                while (jj > 0 and row_vals[jj - 1] > key) : (jj -= 1) {
                    row_vals[jj] = row_vals[jj - 1];
                }
                row_vals[jj] = key;
            }
        }

        // For each distinct row, create a rank=same subgraph and ordering edges
        for (row_vals[0..row_val_count], 0..) |row_val, ri| {
            // Subgraph name must NOT start with "cluster" (that would make it a box)
            var rank_name: [32]u8 = undefined;
            const rn_prefix = "rank_";
            @memcpy(rank_name[0..rn_prefix.len], rn_prefix);
            const ri_len = writeInt(rank_name[rn_prefix.len..], @intCast(ri));
            rank_name[rn_prefix.len + ri_len] = 0;

            const rank_sg = c.gviz_add_subgraph(graph, @ptrCast(&rank_name)) orelse continue;
            c.gviz_set_subgraph_attr(rank_sg, "rank", "same");

            // Collect nodes in this row with their col values
            const RowNode = struct { idx: usize, col_val: i32 };
            var row_members: [MAX_NODES]RowNode = undefined;
            var rm_count: usize = 0;
            for (nodes[0..node_count], 0..) |n, ni| {
                if (n.row != null and n.row.? == row_val) {
                    row_members[rm_count] = .{ .idx = ni, .col_val = n.col orelse 0 };
                    rm_count += 1;
                }
            }

            // Sort by col (insertion sort)
            if (rm_count > 1) {
                for (1..rm_count) |ii| {
                    const key = row_members[ii];
                    var jj: usize = ii;
                    while (jj > 0 and row_members[jj - 1].col_val > key.col_val) : (jj -= 1) {
                        row_members[jj] = row_members[jj - 1];
                    }
                    row_members[jj] = key;
                }
            }

            // Add nodes to rank subgraph
            for (row_members[0..rm_count]) |rm| {
                if (node_ptrs[rm.idx]) |np| {
                    _ = c.gviz_subgraph_add_node(rank_sg, np);
                }
            }

            // Add invisible edges between adjacent col-sorted nodes for left-to-right ordering
            if (rm_count > 1) {
                for (0..rm_count - 1) |ii| {
                    const from_p = node_ptrs[row_members[ii].idx];
                    const to_p = node_ptrs[row_members[ii + 1].idx];
                    if (from_p != null and to_p != null) {
                        var invis_name: [32]u8 = undefined;
                        const ip = "iv_";
                        @memcpy(invis_name[0..ip.len], ip);
                        var il: usize = ip.len;
                        il += writeInt(invis_name[il..], @intCast(ri));
                        invis_name[il] = '_';
                        il += 1;
                        il += writeInt(invis_name[il..], @intCast(ii));
                        invis_name[il] = 0;

                        const invis_edge = c.gviz_add_edge(graph, from_p.?, to_p.?, @ptrCast(&invis_name));
                        if (invis_edge) |ie| {
                            c.gviz_set_edge_attr(graph, ie, "style", "invis");
                            c.gviz_set_edge_attr(graph, ie, "weight", "100");
                            // constraint=false: don't use this edge for rank assignment
                            // (rank=same handles vertical placement; this edge only orders left-to-right)
                            c.gviz_set_edge_attr(graph, ie, "constraint", "false");
                        }
                    }
                }
            }
        }

        // Add invisible edges between rows to enforce vertical ordering.
        // Connect one node from row[i] to one node from row[i+1].
        if (row_val_count > 1) {
            for (0..row_val_count - 1) |ri| {
                // Find first node in row ri and first node in row ri+1
                var from_idx: ?usize = null;
                var to_idx: ?usize = null;
                for (nodes[0..node_count], 0..) |n, ni| {
                    if (n.row != null and n.row.? == row_vals[ri] and from_idx == null) from_idx = ni;
                    if (n.row != null and n.row.? == row_vals[ri + 1] and to_idx == null) to_idx = ni;
                }
                if (from_idx != null and to_idx != null) {
                    const from_p = node_ptrs[from_idx.?];
                    const to_p = node_ptrs[to_idx.?];
                    if (from_p != null and to_p != null) {
                        var chain_name: [32]u8 = undefined;
                        const cp = "rc_";
                        @memcpy(chain_name[0..cp.len], cp);
                        const cl = writeInt(chain_name[cp.len..], @intCast(ri));
                        chain_name[cp.len + cl] = 0;

                        const chain_edge = c.gviz_add_edge(graph, from_p.?, to_p.?, @ptrCast(&chain_name));
                        if (chain_edge) |ce| {
                            c.gviz_set_edge_attr(graph, ce, "style", "invis");
                            c.gviz_set_edge_attr(graph, ce, "weight", "1");
                        }
                    }
                }
            }
        }
    }

    // Create cluster subgraphs for groups (Graphviz keeps clusters non-overlapping).
    // Two passes: first create top-level clusters, then nested ones (parent must exist first).
    var cluster_ptrs: [MAX_GROUPS]?*anyopaque = undefined;
    // Pass 1: top-level groups (no parent)
    for (groups[0..group_count], 0..) |g, gi| {
        if (g.parent_slice.len > 0) {
            cluster_ptrs[gi] = null;
            continue;
        }
        cluster_ptrs[gi] = createCluster(graph, gi, g, &nodes, node_count, &node_ptrs);
    }
    // Pass 2: nested groups (have parent — create under parent cluster)
    for (groups[0..group_count], 0..) |g, gi| {
        if (g.parent_slice.len == 0) continue;
        // Find parent cluster pointer
        var parent_ptr: ?*anyopaque = graph;
        for (groups[0..group_count], 0..) |pg, pgi| {
            if (std.mem.eql(u8, pg.id_slice, g.parent_slice)) {
                parent_ptr = cluster_ptrs[pgi] orelse graph;
                break;
            }
        }
        cluster_ptrs[gi] = createCluster(parent_ptr.?, gi, g, &nodes, node_count, &node_ptrs);
    }

    // Create edges (store pointers for reading label positions back)
    var edge_ptrs: [MAX_EDGES]?*anyopaque = undefined;
    for (edges[0..edge_count], 0..) |e, ei| {
        // Find source and target node pointers
        var from_ptr: ?*anyopaque = null;
        var to_ptr: ?*anyopaque = null;
        for (nodes[0..node_count], 0..) |n, ni| {
            if (std.mem.eql(u8, n.id_slice, e.from_slice)) from_ptr = node_ptrs[ni];
            if (std.mem.eql(u8, n.id_slice, e.to_slice)) to_ptr = node_ptrs[ni];
        }
        if (from_ptr == null or to_ptr == null) {
            edge_ptrs[ei] = null;
            continue;
        }

        // Edge name for uniqueness
        var edge_name: [32]u8 = undefined;
        const elen = writeInt(&edge_name, @intCast(ei));
        edge_name[elen] = 0;
        const edge_ptr = c.gviz_add_edge(graph, from_ptr.?, to_ptr.?, @ptrCast(&edge_name));
        edge_ptrs[ei] = edge_ptr;

        // Pass label to Graphviz so it computes label placement natively
        if (e.label_slice.len > 0) {
            if (edge_ptr) |ep| {
                const label_z = nullTerminate(e.label_slice) orelse continue;
                c.gviz_set_edge_attr(graph, ep, "label", label_z);
            }
        }

        // If both endpoints have the same row value, set constraint=false
        // so this edge doesn't override rank=same placement.
        if (edge_ptr) |ep| {
            var from_row: ?i32 = null;
            var to_row: ?i32 = null;
            for (nodes[0..node_count]) |n| {
                if (std.mem.eql(u8, n.id_slice, e.from_slice)) from_row = n.row;
                if (std.mem.eql(u8, n.id_slice, e.to_slice)) to_row = n.row;
            }
            if (from_row != null and to_row != null and from_row.? == to_row.?) {
                c.gviz_set_edge_attr(graph, ep, "constraint", "false");
            }
        }
    }

    // For nop2 mode: set pos="x,y!" on each node so neato reads fixed positions
    if (use_nop2) {
        for (nodes[0..node_count], 0..) |n, i| {
            if (n.abs_x != null and n.abs_y != null) {
                const np = node_ptrs[i] orelse continue;
                // Convert Excalidraw coords (top-left, Y-down) to Graphviz (center, Y-up, 72dpi)
                // absX/absY are top-left corner; add half width/height for center
                const cx = @as(f64, @floatFromInt(n.abs_x.?)) + @as(f64, @floatFromInt(n.width)) / 2.0;
                const cy_excali = @as(f64, @floatFromInt(n.abs_y.?)) + @as(f64, @floatFromInt(n.height)) / 2.0;
                // Graphviz Y-up: we use raw value, writeGraphvizOutput will flip back
                var pos_buf: [64]u8 = undefined;
                var pb: usize = 0;
                pb += writeFloat(pos_buf[pb..], cx);
                pos_buf[pb] = ',';
                pb += 1;
                pb += writeFloat(pos_buf[pb..], cy_excali);
                pos_buf[pb] = '!';
                pb += 1;
                pos_buf[pb] = 0;
                c.gviz_set_node_attr(graph, np, "pos", @ptrCast(&pos_buf));
            }
        }
        // For nop2: positions are already set, skip overlap adjustment and edge routing.
        // Edge routing for pre-positioned graphs is handled by preserving original arrow
        // paths from the source file (SDK passes them through unchanged).
        c.gviz_set_graph_attr(graph, "overlap", "true");
        c.gviz_set_graph_attr(graph, "splines", "none");
    }

    // Create GVC context
    const gvc = c.gviz_context_new() orelse {
        c.gviz_graph_close(graph);
        return 0;
    };

    // Run layout: circo for radial (state-machine-style cycles),
    // nop2 for pre-positioned, dot for everything else.
    const layout_rc = if (use_nop2)
        c.gviz_layout_nop2(gvc, graph)
    else if (use_circo)
        c.gviz_layout_circo(gvc, graph)
    else
        c.gviz_layout(gvc, graph);
    if (layout_rc != 0) {
        c.gviz_graph_close(graph);
        c.gviz_context_free(gvc);
        return 0;
    }

    // Extract positions and write JSON output (including cluster bounding boxes)
    const result = writeGraphvizOutput(out, graph, gvc, &nodes, node_count, &edges, edge_count, &groups, group_count, &cluster_ptrs, &edge_ptrs);

    // Cleanup
    c.gviz_free_layout(gvc, graph);
    c.gviz_graph_close(graph);
    c.gviz_context_free(gvc);

    return result;
}

// ── Graphviz C bridge FFI (from gviz_bridge.c) ──
// Only linked when targeting WASM (Graphviz C is compiled for wasm32-wasi only).
// Native test builds use parsing but not the layout C FFI.

const builtin = @import("builtin");
const is_wasm = builtin.cpu.arch == .wasm32;

const GvizPoint = extern struct {
    x: f64,
    y: f64,
};

const GvizSpline = extern struct {
    point_count: usize,
    points: [*]const GvizPoint,
    has_start_point: c_int,
    start_point: GvizPoint,
    has_end_point: c_int,
    end_point: GvizPoint,
};

const GvizBbox = extern struct {
    ll_x: f64,
    ll_y: f64,
    ur_x: f64,
    ur_y: f64,
};

// Graphviz C bridge: real extern declarations for WASM, no-op shims for native tests
pub const c = if (is_wasm) struct {
    // Graph construction (programmatic — no DOT parser needed)
    pub extern fn gviz_graph_new(name: [*:0]const u8) ?*anyopaque;
    pub extern fn gviz_add_node(g: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
    pub extern fn gviz_add_edge(g: ?*anyopaque, tail: ?*anyopaque, head: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
    pub extern fn gviz_set_default_node_attr(g: ?*anyopaque, name: [*:0]const u8, value: [*:0]const u8) void;
    pub extern fn gviz_set_graph_attr(g: ?*anyopaque, name: [*:0]const u8, value: [*:0]const u8) void;
    pub extern fn gviz_set_subgraph_attr(g: ?*anyopaque, name: [*:0]const u8, value: [*:0]const u8) void;
    pub extern fn gviz_set_node_attr(g: ?*anyopaque, n: ?*anyopaque, name: [*:0]const u8, value: [*:0]const u8) void;
    pub extern fn gviz_set_default_edge_attr(g: ?*anyopaque, name: [*:0]const u8, value: [*:0]const u8) void;
    pub extern fn gviz_set_edge_attr(g: ?*anyopaque, e: ?*anyopaque, name: [*:0]const u8, value: [*:0]const u8) void;

    // Cluster subgraphs
    pub extern fn gviz_add_subgraph(g: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
    pub extern fn gviz_subgraph_add_node(subg: ?*anyopaque, n: ?*anyopaque) ?*anyopaque;

    // Context and layout
    pub extern fn gviz_context_new() ?*anyopaque;
    pub extern fn gviz_context_free(ctx: ?*anyopaque) void;
    pub extern fn gviz_graph_close(g: ?*anyopaque) void;
    pub extern fn gviz_layout(ctx: ?*anyopaque, g: ?*anyopaque) c_int;
        pub extern fn gviz_layout_nop2(ctx: ?*anyopaque, g: ?*anyopaque) c_int;
    pub extern fn gviz_layout_circo(ctx: ?*anyopaque, g: ?*anyopaque) c_int;
    pub extern fn gviz_free_layout(ctx: ?*anyopaque, g: ?*anyopaque) void;

    // Node iteration
    pub extern fn gviz_first_node(g: ?*anyopaque) ?*anyopaque;
    pub extern fn gviz_next_node(g: ?*anyopaque, n: ?*anyopaque) ?*anyopaque;
    pub extern fn gviz_node_name(n: ?*anyopaque) ?[*:0]const u8;
    pub extern fn gviz_node_coord(n: ?*anyopaque, x: *f64, y: *f64) void;

    // Edge iteration
    pub extern fn gviz_first_out_edge(g: ?*anyopaque, n: ?*anyopaque) ?*anyopaque;
    pub extern fn gviz_next_out_edge(g: ?*anyopaque, e: ?*anyopaque) ?*anyopaque;
    pub extern fn gviz_edge_head(e: ?*anyopaque) ?*anyopaque;
    pub extern fn gviz_edge_tail(e: ?*anyopaque) ?*anyopaque;
    pub extern fn gviz_edge_spline(e: ?*anyopaque, out: *GvizSpline) c_int;
    pub extern fn gviz_edge_label_pos(e: ?*anyopaque, x: *f64, y: *f64) c_int;
    pub extern fn gviz_graph_bbox(g: ?*anyopaque) GvizBbox;
} else struct {
    pub fn gviz_graph_new(_: [*:0]const u8) ?*anyopaque { return null; }
    pub fn gviz_add_node(_: ?*anyopaque, _: [*:0]const u8) ?*anyopaque { return null; }
    pub fn gviz_add_edge(_: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque, _: [*:0]const u8) ?*anyopaque { return null; }
    pub fn gviz_set_default_node_attr(_: ?*anyopaque, _: [*:0]const u8, _: [*:0]const u8) void {}
    pub fn gviz_set_graph_attr(_: ?*anyopaque, _: [*:0]const u8, _: [*:0]const u8) void {}
    pub fn gviz_set_subgraph_attr(_: ?*anyopaque, _: [*:0]const u8, _: [*:0]const u8) void {}
    pub fn gviz_set_node_attr(_: ?*anyopaque, _: ?*anyopaque, _: [*:0]const u8, _: [*:0]const u8) void {}
    pub fn gviz_set_default_edge_attr(_: ?*anyopaque, _: [*:0]const u8, _: [*:0]const u8) void {}
    pub fn gviz_set_edge_attr(_: ?*anyopaque, _: ?*anyopaque, _: [*:0]const u8, _: [*:0]const u8) void {}
    pub fn gviz_add_subgraph(_: ?*anyopaque, _: [*:0]const u8) ?*anyopaque { return null; }
    pub fn gviz_subgraph_add_node(_: ?*anyopaque, _: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_context_new() ?*anyopaque { return null; }
    pub fn gviz_context_free(_: ?*anyopaque) void {}
    pub fn gviz_graph_close(_: ?*anyopaque) void {}
    pub fn gviz_layout(_: ?*anyopaque, _: ?*anyopaque) c_int { return -1; }
        pub fn gviz_layout_nop2(_: ?*anyopaque, _: ?*anyopaque) c_int { return -1; }
    pub fn gviz_layout_circo(_: ?*anyopaque, _: ?*anyopaque) c_int { return -1; }
    pub fn gviz_free_layout(_: ?*anyopaque, _: ?*anyopaque) void {}
    pub fn gviz_first_node(_: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_next_node(_: ?*anyopaque, _: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_node_name(_: ?*anyopaque) ?[*:0]const u8 { return null; }
    pub fn gviz_node_coord(_: ?*anyopaque, _: *f64, _: *f64) void {}
    pub fn gviz_first_out_edge(_: ?*anyopaque, _: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_next_out_edge(_: ?*anyopaque, _: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_edge_head(_: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_edge_tail(_: ?*anyopaque) ?*anyopaque { return null; }
    pub fn gviz_edge_spline(_: ?*anyopaque, _: *GvizSpline) c_int { return 0; }
    pub fn gviz_edge_label_pos(_: ?*anyopaque, _: *f64, _: *f64) c_int { return 0; }
    pub fn gviz_graph_bbox(_: ?*anyopaque) GvizBbox { return .{ .ll_x = 0, .ll_y = 0, .ur_x = 0, .ur_y = 0 }; }
};

// ── Constants ──

const MAX_NODES = 256;
const MAX_EDGES = 512;
const MAX_GROUPS = 32;
const MAX_GROUP_CHILDREN = 64;

const Node = struct {
    id_slice: []const u8,
    width: i32,
    height: i32,
    row: ?i32,
    col: ?i32,
    abs_x: ?i32,
    abs_y: ?i32,
};

const Edge = struct {
    from_slice: []const u8,
    to_slice: []const u8,
    label_slice: []const u8,
};

const Group = struct {
    id_slice: []const u8,
    label_slice: []const u8,
    parent_slice: []const u8, // "" if top-level
    children_slices: [MAX_GROUP_CHILDREN][]const u8,
    child_count: usize,
};

/// Create a Graphviz cluster subgraph under a parent graph/cluster.
fn createCluster(parent: *anyopaque, gi: usize, g: Group, nodes: *[MAX_NODES]Node, node_count: usize, node_ptrs: *[MAX_NODES]?*anyopaque) ?*anyopaque {
    // Graphviz requires cluster names to start with "cluster"
    var cluster_name: [64]u8 = undefined;
    const prefix = "cluster_";
    @memcpy(cluster_name[0..prefix.len], prefix);
    const idx_len = writeInt(cluster_name[prefix.len..], @intCast(gi));
    cluster_name[prefix.len + idx_len] = 0;

    const subg = c.gviz_add_subgraph(parent, @ptrCast(&cluster_name)) orelse return null;

    // Set cluster label and margin (margin ensures padding for TS-side group rect)
    const label_z = nullTerminate(g.label_slice) orelse return subg;
    c.gviz_set_graph_attr(subg, "label", label_z);
    c.gviz_set_graph_attr(subg, "style", "dashed");
    c.gviz_set_graph_attr(subg, "margin", "20");

    // Add child nodes to the cluster subgraph
    for (g.children_slices[0..g.child_count]) |child_id| {
        for (nodes[0..node_count], 0..) |n, ni| {
            if (std.mem.eql(u8, n.id_slice, child_id)) {
                if (node_ptrs[ni]) |np| {
                    _ = c.gviz_subgraph_add_node(subg, np);
                }
                break;
            }
        }
    }
    return subg;
}

// ── Output: Extract Graphviz Layout Results ──

fn writeGraphvizOutput(out: []u8, graph: *anyopaque, _: *anyopaque, nodes: *[MAX_NODES]Node, node_count: usize, edges: *[MAX_EDGES]Edge, edge_count: usize, groups: *[MAX_GROUPS]Group, group_count: usize, cluster_ptrs: *[MAX_GROUPS]?*anyopaque, edge_ptrs: *[MAX_EDGES]?*anyopaque) usize {
    var w: usize = 0;
    // Reserve margin to detect buffer exhaustion (return 0 = error)
    const MARGIN = 256;

    // Get bounding box for Y-flip (Graphviz Y-up → Excalidraw Y-down)
    const bb = c.gviz_graph_bbox(graph);
    const y_max = bb.ur_y;

    w += copySlice(out[w..], "{\"nodes\":[");

    // Store node bounding boxes for label-vs-node collision avoidance
    const NodeBox = struct { cx: i32, cy: i32, hw: i32, hh: i32 };
    var node_boxes: [MAX_NODES]NodeBox = undefined;
    var node_box_count: usize = 0;

    // Extract node positions
    var first_node = true;
    var n_ptr = c.gviz_first_node(graph);
    while (n_ptr) |n| {
        const name_ptr = c.gviz_node_name(n);
        if (name_ptr == null) {
            n_ptr = c.gviz_next_node(graph, n);
            continue;
        }
        const name_slice = std.mem.span(name_ptr.?);

        // Find matching input node for dimensions
        var input_w: i32 = 180;
        var input_h: i32 = 80;
        for (nodes[0..node_count]) |in_node| {
            if (std.mem.eql(u8, in_node.id_slice, name_slice)) {
                input_w = in_node.width;
                input_h = in_node.height;
                break;
            }
        }

        // Get node center position (Graphviz points, 72 DPI, Y-up)
        var cx: f64 = 0;
        var cy: f64 = 0;
        c.gviz_node_coord(n, &cx, &cy);

        // Convert to Excalidraw coordinates (Y-down, top-left corner)
        // Use pinned absX/absY if the input node has them (fromFile re-render)
        var node_x = @as(i32, @intFromFloat(cx)) - @divTrunc(input_w, 2);
        var node_y = @as(i32, @intFromFloat(y_max - cy)) - @divTrunc(input_h, 2);
        for (nodes[0..node_count]) |in_node| {
            if (std.mem.eql(u8, in_node.id_slice, name_slice)) {
                if (in_node.abs_x) |ax| node_x = ax;
                if (in_node.abs_y) |ay| node_y = ay;
                break;
            }
        }

        // Store node bounding box (center coords, half-dimensions)
        if (node_box_count < MAX_NODES) {
            node_boxes[node_box_count] = .{
                .cx = node_x + @divTrunc(input_w, 2),
                .cy = node_y + @divTrunc(input_h, 2),
                .hw = @divTrunc(input_w, 2),
                .hh = @divTrunc(input_h, 2),
            };
            node_box_count += 1;
        }

        if (!first_node) w += copySlice(out[w..], ",");
        first_node = false;

        w += copySlice(out[w..], "{\"id\":\"");
        w += copySlice(out[w..], name_slice);
        w += copySlice(out[w..], "\",\"x\":");
        w += writeInt(out[w..], node_x);
        w += copySlice(out[w..], ",\"y\":");
        w += writeInt(out[w..], node_y);
        w += copySlice(out[w..], "}");
        if (w + MARGIN > out.len) return 0; // buffer exhausted

        n_ptr = c.gviz_next_node(graph, n);
    }

    w += copySlice(out[w..], "],\"edges\":[");

    // Extract edge splines and label positions from Graphviz
    const EdgeInfo = struct {
        spline: SplineResult,
        edge_idx: usize,
        label_x: i32,
        label_y: i32,
        label_w: i32,
        has_label: bool,
    };
    var edge_infos: [MAX_EDGES]EdgeInfo = undefined;
    var edge_info_count: usize = 0;

    for (edges[0..edge_count], 0..) |e, ei| {
        const ep = edge_ptrs[ei] orelse continue;
        const spline = extractSplineFromEdge(ep, e.from_slice, e.to_slice, y_max, nodes, node_count);
        if (spline.point_count < 2) continue;

        var info = EdgeInfo{
            .spline = spline,
            .edge_idx = ei,
            .label_x = 0,
            .label_y = 0,
            .label_w = if (e.label_slice.len > 0) @as(i32, @intCast(e.label_slice.len)) * 8 + 16 else 0,
            .has_label = e.label_slice.len > 0,
        };

        // Read label position from Graphviz (computed natively during layout)
        if (info.has_label) {
            var lx: f64 = 0;
            var ly: f64 = 0;
            if (c.gviz_edge_label_pos(ep, &lx, &ly) != 0) {
                // Graphviz Y-up → Excalidraw Y-down
                info.label_x = @intFromFloat(lx);
                info.label_y = @intFromFloat(y_max - ly);
            } else {
                // Fallback: midpoint of longest segment
                var best_len: i32 = 0;
                var best_seg: usize = 0;
                var s: usize = 0;
                while (s + 1 < spline.point_count) : (s += 1) {
                    const dx = absInt(spline.points_x[s + 1] - spline.points_x[s]);
                    const dy = absInt(spline.points_y[s + 1] - spline.points_y[s]);
                    const seg_len = dx + dy;
                    if (seg_len > best_len) { best_len = seg_len; best_seg = s; }
                }
                info.label_x = @divTrunc(spline.points_x[best_seg] + spline.points_x[best_seg + 1], 2);
                info.label_y = @divTrunc(spline.points_y[best_seg] + spline.points_y[best_seg + 1], 2);
            }
        }

        edge_infos[edge_info_count] = info;
        edge_info_count += 1;
    }

    // Post-Graphviz collision fix: Graphviz doesn't know actual text widths
    // (gvtextlayout returns false in WASM), so nearby labels may still overlap.
    // Push overlapping labels apart on whichever axis needs less movement.
    const LABEL_H: i32 = 24;
    const MIN_GAP: i32 = 48;
    var pass: usize = 0;
    while (pass < 10) : (pass += 1) {
        var shifted = false;
        for (0..edge_info_count) |i| {
            if (!edge_infos[i].has_label) continue;
            for (0..i) |j| {
                if (!edge_infos[j].has_label) continue;

                const i_hw = @divTrunc(edge_infos[i].label_w, 2);
                const j_hw = @divTrunc(edge_infos[j].label_w, 2);
                const hh = @divTrunc(LABEL_H, 2);

                // Check bounding-box overlap with gap
                const x_overlap = (i_hw + j_hw + MIN_GAP) - absInt(edge_infos[i].label_x - edge_infos[j].label_x);
                const y_overlap = (hh + hh + MIN_GAP) - absInt(edge_infos[i].label_y - edge_infos[j].label_y);

                if (x_overlap > 0 and y_overlap > 0) {
                    // Push apart on axis needing less movement
                    if (x_overlap < y_overlap) {
                        const half = @divTrunc(x_overlap + 1, 2);
                        if (edge_infos[i].label_x >= edge_infos[j].label_x) {
                            edge_infos[i].label_x += half;
                            edge_infos[j].label_x -= half;
                        } else {
                            edge_infos[i].label_x -= half;
                            edge_infos[j].label_x += half;
                        }
                    } else {
                        const half = @divTrunc(y_overlap + 1, 2);
                        if (edge_infos[i].label_y >= edge_infos[j].label_y) {
                            edge_infos[i].label_y += half;
                            edge_infos[j].label_y -= half;
                        } else {
                            edge_infos[i].label_y -= half;
                            edge_infos[j].label_y += half;
                        }
                    }
                    shifted = true;
                    break; // re-check from next pass after shifting
                }
            }
        }
        if (!shifted) break;
    }

    // Label-vs-edge-line collision fix: if a label's bounding box intersects
    // another edge's polyline segments, push the label perpendicular to that
    // segment until clear. Graphviz doesn't know our actual label widths so
    // its native placement often lands on top of crossing arrows.
    const LINE_LABEL_GAP: i32 = 10;
    var line_pass: usize = 0;
    while (line_pass < 10) : (line_pass += 1) {
        var line_shifted = false;
        for (0..edge_info_count) |i| {
            if (!edge_infos[i].has_label) continue;
            const lhw = @divTrunc(edge_infos[i].label_w, 2);
            const lhh = @divTrunc(LABEL_H, 2);
            const half_extent: i32 = @max(lhw, lhh) + LINE_LABEL_GAP;

            for (0..edge_info_count) |j| {
                if (i == j) continue;  // skip label's own edge
                const other = edge_infos[j].spline;
                if (other.point_count < 2) continue;

                var s: usize = 0;
                while (s + 1 < other.point_count) : (s += 1) {
                    const x1 = other.points_x[s];
                    const y1 = other.points_y[s];
                    const x2 = other.points_x[s + 1];
                    const y2 = other.points_y[s + 1];

                    // Shortest distance from label center to segment (x1,y1)-(x2,y2).
                    const dx = x2 - x1;
                    const dy = y2 - y1;
                    const len_sq = dx * dx + dy * dy;
                    if (len_sq == 0) continue;
                    const px = edge_infos[i].label_x - x1;
                    const py = edge_infos[i].label_y - y1;
                    var t_num = px * dx + py * dy;
                    if (t_num < 0) t_num = 0;
                    if (t_num > len_sq) t_num = len_sq;
                    // Projection point
                    const proj_x = x1 + @divTrunc(t_num * dx, len_sq);
                    const proj_y = y1 + @divTrunc(t_num * dy, len_sq);
                    const d_x = edge_infos[i].label_x - proj_x;
                    const d_y = edge_infos[i].label_y - proj_y;
                    const dist_sq = d_x * d_x + d_y * d_y;
                    const min_sq = half_extent * half_extent;
                    if (dist_sq < min_sq) {
                        // Push label along (d_x, d_y) perpendicular-to-segment direction.
                        // If exactly on segment, pick arbitrary perpendicular.
                        var nx: i32 = d_x;
                        var ny: i32 = d_y;
                        if (dist_sq == 0) { nx = -dy; ny = dx; }
                        // Scale to push the label clear of the segment + gap.
                        const push_mag: i32 = half_extent + 2;
                        const n_len = isqrt(@as(i64, nx) * nx + @as(i64, ny) * ny);
                        if (n_len == 0) continue;
                        edge_infos[i].label_x = proj_x + @divTrunc(nx * push_mag, n_len);
                        edge_infos[i].label_y = proj_y + @divTrunc(ny * push_mag, n_len);
                        line_shifted = true;
                        break;
                    }
                }
                if (line_shifted) break;
            }
        }
        if (!line_shifted) break;
    }

    // Label-vs-node collision fix: push labels outside node bounding boxes.
    // Graphviz places labels at edge midpoints which can land on top of nodes.
    const NODE_LABEL_GAP: i32 = 16;
    var node_pass: usize = 0;
    while (node_pass < 10) : (node_pass += 1) {
        var node_shifted = false;
        for (0..edge_info_count) |i| {
            if (!edge_infos[i].has_label) continue;
            const lhw = @divTrunc(edge_infos[i].label_w, 2);
            const lhh = @divTrunc(LABEL_H, 2);

            for (node_boxes[0..node_box_count]) |nb| {
                // Check if label bounding box overlaps node bounding box
                const dx = absInt(edge_infos[i].label_x - nb.cx);
                const dy = absInt(edge_infos[i].label_y - nb.cy);
                const min_dx = lhw + nb.hw + NODE_LABEL_GAP;
                const min_dy = lhh + nb.hh + NODE_LABEL_GAP;

                if (dx < min_dx and dy < min_dy) {
                    // Overlap detected — push label to nearest edge of node + gap
                    const push_x = min_dx - dx; // how much to push on X
                    const push_y = min_dy - dy; // how much to push on Y

                    if (push_x < push_y) {
                        // Push horizontally (less movement needed)
                        if (edge_infos[i].label_x >= nb.cx) {
                            edge_infos[i].label_x += push_x;
                        } else {
                            edge_infos[i].label_x -= push_x;
                        }
                    } else {
                        // Push vertically
                        if (edge_infos[i].label_y >= nb.cy) {
                            edge_infos[i].label_y += push_y;
                        } else {
                            edge_infos[i].label_y -= push_y;
                        }
                    }
                    node_shifted = true;
                }
            }
        }
        if (!node_shifted) break;
    }

    // Second pass: write edge JSON with label positions
    var first_edge = true;
    for (edge_infos[0..edge_info_count]) |info| {
        const e = edges[info.edge_idx];
        const spline = info.spline;

        if (!first_edge) w += copySlice(out[w..], ",");
        first_edge = false;

        w += copySlice(out[w..], "{\"from\":\"");
        w += copySlice(out[w..], e.from_slice);
        w += copySlice(out[w..], "\",\"to\":\"");
        w += copySlice(out[w..], e.to_slice);
        w += copySlice(out[w..], "\",\"points\":[");

        for (0..spline.point_count) |pi| {
            if (pi > 0) w += copySlice(out[w..], ",");
            w += copySlice(out[w..], "[");
            w += writeInt(out[w..], spline.points_x[pi]);
            w += copySlice(out[w..], ",");
            w += writeInt(out[w..], spline.points_y[pi]);
            w += copySlice(out[w..], "]");
        }

        w += copySlice(out[w..], "],\"startFixedPoint\":[");
        w += writeFloat(out[w..], spline.start_fixed_point[0]);
        w += copySlice(out[w..], ",");
        w += writeFloat(out[w..], spline.start_fixed_point[1]);
        w += copySlice(out[w..], "],\"endFixedPoint\":[");
        w += writeFloat(out[w..], spline.end_fixed_point[0]);
        w += copySlice(out[w..], ",");
        w += writeFloat(out[w..], spline.end_fixed_point[1]);
        w += copySlice(out[w..], "]");

        // Include label position if edge has a label
        if (info.has_label) {
            w += copySlice(out[w..], ",\"labelX\":");
            w += writeInt(out[w..], info.label_x);
            w += copySlice(out[w..], ",\"labelY\":");
            w += writeInt(out[w..], info.label_y);
        }

        w += copySlice(out[w..], "}");
        if (w + MARGIN > out.len) return 0; // buffer exhausted
    }

    w += copySlice(out[w..], "]");

    // Output cluster (group) bounding boxes from Graphviz
    if (group_count > 0) {
        w += copySlice(out[w..], ",\"groups\":[");
        var first_group = true;
        for (0..group_count) |gi| {
            const cluster = cluster_ptrs[gi] orelse continue;
            const cbb = c.gviz_graph_bbox(cluster);

            if (!first_group) w += copySlice(out[w..], ",");
            first_group = false;

            // Convert Graphviz bbox (Y-up, points) → Excalidraw (Y-down, pixels)
            const gx = @as(i32, @intFromFloat(cbb.ll_x));
            const gy = @as(i32, @intFromFloat(y_max - cbb.ur_y));
            const gw = @as(i32, @intFromFloat(cbb.ur_x - cbb.ll_x));
            const gh = @as(i32, @intFromFloat(cbb.ur_y - cbb.ll_y));

            w += copySlice(out[w..], "{\"id\":\"");
            w += copySlice(out[w..], groups[gi].id_slice);
            w += copySlice(out[w..], "\",\"x\":");
            w += writeInt(out[w..], gx);
            w += copySlice(out[w..], ",\"y\":");
            w += writeInt(out[w..], gy);
            w += copySlice(out[w..], ",\"width\":");
            w += writeInt(out[w..], gw);
            w += copySlice(out[w..], ",\"height\":");
            w += writeInt(out[w..], gh);
            w += copySlice(out[w..], "}");
            if (w + MARGIN > out.len) return 0; // buffer exhausted
        }
        w += copySlice(out[w..], "]");
    }

    w += copySlice(out[w..], "}");
    return w;
}

const MAX_SPLINE_POINTS = 64;

const SplineResult = struct {
    points_x: [MAX_SPLINE_POINTS]i32,
    points_y: [MAX_SPLINE_POINTS]i32,
    point_count: usize,
    start_fixed_point: [2]f64,
    end_fixed_point: [2]f64,
};

/// Extract spline data directly from a Graphviz edge pointer (no search needed).
/// This fixes multi-edge overlap: each edge pointer is unique even for parallel edges.
fn extractSplineFromEdge(edge: *anyopaque, from_name: []const u8, to_name: []const u8, y_max: f64, nodes: *[MAX_NODES]Node, node_count: usize) SplineResult {
    var result = SplineResult{
        .points_x = undefined,
        .points_y = undefined,
        .point_count = 0,
        .start_fixed_point = .{ 0.5, 0.5 },
        .end_fixed_point = .{ 0.5, 0.5 },
    };

    var spline: GvizSpline = undefined;
    if (c.gviz_edge_spline(edge, &spline) == 0) return result;

    // Graphviz returns splines as a sequence of cubic bezier segments: for N
    // segments there are 3N+1 control points (p0, p1, p2, p3=p0', p1', p2', p3',…).
    // Sample each cubic bezier at SAMPLES_PER_SEG points so the rendered
    // polyline follows the curve instead of cutting straight between knots.
    // With SAMPLES_PER_SEG=6 we fit up to 10 bezier segments within the
    // MAX_SPLINE_POINTS=64 cap (6*10+1=61 points).
    const SAMPLES_PER_SEG: usize = 6;
    if (spline.point_count >= 4 and (spline.point_count - 1) % 3 == 0) {
        const segments = (spline.point_count - 1) / 3;
        // First point of the entire spline
        {
            const p0 = spline.points[0];
            result.points_x[result.point_count] = @intFromFloat(p0.x);
            result.points_y[result.point_count] = @intFromFloat(y_max - p0.y);
            result.point_count += 1;
        }
        for (0..segments) |seg| {
            const p0 = spline.points[seg * 3];
            const p1 = spline.points[seg * 3 + 1];
            const p2 = spline.points[seg * 3 + 2];
            const p3 = spline.points[seg * 3 + 3];
            // Sample the cubic bezier at t=1/S, 2/S, ..., 1.0
            for (1..SAMPLES_PER_SEG + 1) |k| {
                if (result.point_count >= MAX_SPLINE_POINTS) break;
                const t: f64 = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(SAMPLES_PER_SEG));
                const u: f64 = 1.0 - t;
                const x = u * u * u * p0.x + 3.0 * u * u * t * p1.x + 3.0 * u * t * t * p2.x + t * t * t * p3.x;
                const y = u * u * u * p0.y + 3.0 * u * u * t * p1.y + 3.0 * u * t * t * p2.y + t * t * t * p3.y;
                result.points_x[result.point_count] = @intFromFloat(x);
                result.points_y[result.point_count] = @intFromFloat(y_max - y);
                result.point_count += 1;
            }
        }
    } else {
        // Not a well-formed bezier sequence — fall back to raw knot sampling.
        var i: usize = 0;
        while (i < spline.point_count and result.point_count < MAX_SPLINE_POINTS) : (i += 3) {
            const pt = spline.points[i];
            result.points_x[result.point_count] = @intFromFloat(pt.x);
            result.points_y[result.point_count] = @intFromFloat(y_max - pt.y);
            result.point_count += 1;
        }
        if (spline.point_count > 0 and (spline.point_count - 1) % 3 != 0) {
            const last_pt = spline.points[spline.point_count - 1];
            if (result.point_count < MAX_SPLINE_POINTS) {
                result.points_x[result.point_count] = @intFromFloat(last_pt.x);
                result.points_y[result.point_count] = @intFromFloat(y_max - last_pt.y);
                result.point_count += 1;
            }
        }
    }

    // Compute fixedPoints using sp/ep (arrow tip positions on node boundary)
    if (result.point_count >= 2) {
        const fix_start_x: f64 = if (spline.has_start_point != 0) spline.start_point.x else @floatFromInt(result.points_x[0]);
        const fix_start_y: f64 = if (spline.has_start_point != 0) (y_max - spline.start_point.y) else @floatFromInt(result.points_y[0]);
        const fix_end_x: f64 = if (spline.has_end_point != 0) spline.end_point.x else @floatFromInt(result.points_x[result.point_count - 1]);
        const fix_end_y: f64 = if (spline.has_end_point != 0) (y_max - spline.end_point.y) else @floatFromInt(result.points_y[result.point_count - 1]);

        // Source node fixedPoint
        const tail = c.gviz_edge_tail(edge);
        if (tail != null) {
            for (nodes[0..node_count]) |sn| {
                if (std.mem.eql(u8, sn.id_slice, from_name)) {
                    var scx: f64 = 0;
                    var scy: f64 = 0;
                    c.gviz_node_coord(tail, &scx, &scy);
                    if (sn.width == 0 or sn.height == 0) break;
                    const snx = scx - @as(f64, @floatFromInt(sn.width)) / 2.0;
                    const sny = (y_max - scy) - @as(f64, @floatFromInt(sn.height)) / 2.0;
                    result.start_fixed_point[0] = clamp01((fix_start_x - snx) / @as(f64, @floatFromInt(sn.width)));
                    result.start_fixed_point[1] = clamp01((fix_start_y - sny) / @as(f64, @floatFromInt(sn.height)));
                    break;
                }
            }
        }
        // Target node fixedPoint
        const head = c.gviz_edge_head(edge);
        if (head != null) {
            for (nodes[0..node_count]) |tn| {
                if (std.mem.eql(u8, tn.id_slice, to_name)) {
                    if (tn.width == 0 or tn.height == 0) break;
                    var tcx: f64 = 0;
                    var tcy: f64 = 0;
                    c.gviz_node_coord(head, &tcx, &tcy);
                    const tnx = tcx - @as(f64, @floatFromInt(tn.width)) / 2.0;
                    const tny = (y_max - tcy) - @as(f64, @floatFromInt(tn.height)) / 2.0;
                    result.end_fixed_point[0] = clamp01((fix_end_x - tnx) / @as(f64, @floatFromInt(tn.width)));
                    result.end_fixed_point[1] = clamp01((fix_end_y - tny) / @as(f64, @floatFromInt(tn.height)));
                    break;
                }
            }
        }
    }

    return result;
}

fn absInt(v: i32) i32 {
    if (v == std.math.minInt(i32)) return std.math.maxInt(i32);
    return if (v < 0) -v else v;
}

fn isqrt(v: i64) i32 {
    if (v <= 0) return 0;
    return @intCast(std.math.sqrt(@as(u64, @intCast(v))));
}

fn clamp01(v: f64) f64 {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

// ── Null-termination helper ──

/// Copy a slice into a stack buffer and null-terminate it for C FFI.
/// Returns a sentinel-terminated pointer, or null if the slice is too long.
const NT_BUF_COUNT = 16;
const NT_BUF_SIZE = 256;
var nt_bufs: [NT_BUF_COUNT][NT_BUF_SIZE]u8 = undefined;
var nt_idx: usize = 0;

fn nullTerminate(slice: []const u8) ?[*:0]const u8 {
    if (slice.len >= NT_BUF_SIZE) return null;
    const idx = nt_idx % NT_BUF_COUNT;
    nt_idx += 1;
    @memcpy(nt_bufs[idx][0..slice.len], slice);
    nt_bufs[idx][slice.len] = 0;
    return @ptrCast(&nt_bufs[idx]);
}

// ── Float formatting ──

fn writeFloat(out: []u8, val: f64) usize {
    // Guard against extreme values that would overflow i64/i32
    const clamped = std.math.clamp(val, -1e9, 1e9);
    const negative = clamped < 0;
    const abs_val = if (negative) -clamped else clamped;
    const int_part = @as(i64, @intFromFloat(abs_val));
    var frac = @as(i64, @intFromFloat((abs_val - @as(f64, @floatFromInt(int_part))) * 100.0 + 0.5));
    // Clamp frac to 99 — rounding can push it to 100
    if (frac > 99) frac = 99;

    var w: usize = 0;
    if (out.len < 16) return 0; // need space for sign + int + dot + frac
    if (negative) {
        out[w] = '-';
        w += 1;
    }
    w += writeInt(out[w..], @intCast(int_part));
    out[w] = '.';
    w += 1;
    if (frac < 10) {
        out[w] = '0';
        w += 1;
    }
    w += writeInt(out[w..], @intCast(frac));
    return w;
}

// ── JSON Parsing ──

fn parseNodes(json: []const u8, out_nodes: *[MAX_NODES]Node) !usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (pos < json.len and json[pos] != '{') : (pos += 1) {}
    while (pos < json.len and count < MAX_NODES) {
        if (json[pos] == '{') {
            var node = Node{ .id_slice = &.{}, .width = 180, .height = 80, .row = null, .col = null, .abs_x = null, .abs_y = null };
            const obj_end = findMatchingBrace(json[pos..]) + pos;
            const obj = json[pos..obj_end];
            node.id_slice = extractStringField(obj, "id") orelse &.{};
            node.width = extractIntField(obj, "width") orelse 180;
            node.height = extractIntField(obj, "height") orelse 80;
            node.row = extractIntField(obj, "row");
            node.col = extractIntField(obj, "col");
            node.abs_x = extractIntField(obj, "absX");
            node.abs_y = extractIntField(obj, "absY");
            out_nodes[count] = node;
            count += 1;
            pos = obj_end;
        }
        pos += 1;
    }
    return count;
}

fn parseEdges(json: []const u8, out_edges: *[MAX_EDGES]Edge) !usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (pos < json.len and json[pos] != '{') : (pos += 1) {}
    while (pos < json.len and count < MAX_EDGES) {
        if (json[pos] == '{') {
            var edge = Edge{ .from_slice = &.{}, .to_slice = &.{}, .label_slice = &.{} };
            const obj_end = findMatchingBrace(json[pos..]) + pos;
            const obj = json[pos..obj_end];
            edge.from_slice = extractStringField(obj, "from") orelse &.{};
            edge.to_slice = extractStringField(obj, "to") orelse &.{};
            edge.label_slice = extractStringField(obj, "label") orelse &.{};
            out_edges[count] = edge;
            count += 1;
            pos = obj_end;
        }
        pos += 1;
    }
    return count;
}

/// Parse groups JSON: [{"id":"g1","label":"Group","children":["n1","n2"]}, ...]
fn parseGroups(json: []const u8, out_groups: *[MAX_GROUPS]Group) !usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (pos < json.len and json[pos] != '{') : (pos += 1) {}
    while (pos < json.len and count < MAX_GROUPS) {
        if (json[pos] == '{') {
            const obj_end = findMatchingBrace(json[pos..]) + pos;
            const obj = json[pos..obj_end];

            var group = Group{
                .id_slice = extractStringField(obj, "id") orelse &.{},
                .label_slice = extractStringField(obj, "label") orelse &.{},
                .parent_slice = extractStringField(obj, "parent") orelse &.{},
                .children_slices = undefined,
                .child_count = 0,
            };

            // Parse children array: find "children":[ then extract strings
            if (std.mem.indexOf(u8, obj, "\"children\"")) |ci| {
                var cp = ci + "\"children\"".len;
                // Skip to '['
                while (cp < obj.len and obj[cp] != '[') : (cp += 1) {}
                if (cp < obj.len) {
                    cp += 1; // skip '['
                    while (cp < obj.len and obj[cp] != ']' and group.child_count < MAX_GROUP_CHILDREN) {
                        if (obj[cp] == '"') {
                            cp += 1; // skip opening quote
                            const str_start = cp;
                            while (cp < obj.len and obj[cp] != '"') : (cp += 1) {}
                            group.children_slices[group.child_count] = obj[str_start..cp];
                            group.child_count += 1;
                            if (cp < obj.len) cp += 1; // skip closing quote
                        } else {
                            cp += 1;
                        }
                    }
                }
            }

            out_groups[count] = group;
            count += 1;
            pos = obj_end;
        }
        pos += 1;
    }
    return count;
}

const findMatchingBrace = util.findMatchingBrace;
const extractStringField = util.extractStringField;
const extractIntField = util.extractIntField;
const copySlice = util.copySlice;
const writeInt = util.writeInt;

// ── Tests ──

test "node parsing" {
    const nodes_json =
        \\[{"id":"a","width":180,"height":80,"row":0,"col":0},{"id":"b","width":180,"height":80,"row":1,"col":1}]
    ;
    var nodes: [MAX_NODES]Node = undefined;
    const count = try parseNodes(nodes_json, &nodes);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expect(std.mem.eql(u8, nodes[0].id_slice, "a"));
    try std.testing.expect(std.mem.eql(u8, nodes[1].id_slice, "b"));
    try std.testing.expectEqual(@as(i32, 180), nodes[0].width);
}

test "edge parsing" {
    const edges_json =
        \\[{"from":"a","to":"b"},{"from":"b","to":"c"}]
    ;
    var edges: [MAX_EDGES]Edge = undefined;
    const count = try parseEdges(edges_json, &edges);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expect(std.mem.eql(u8, edges[0].from_slice, "a"));
    try std.testing.expect(std.mem.eql(u8, edges[0].to_slice, "b"));
}
