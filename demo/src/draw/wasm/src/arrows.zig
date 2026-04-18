const std = @import("std");
const util = @import("util.zig");

/// Route arrows between Excalidraw elements.
///
/// Input: JSON array of Excalidraw elements (shapes + arrows)
/// Output: JSON array of corrected arrow elements with proper endpoints and elbow routing
///
/// The router:
/// 1. Finds shape bounding boxes
/// 2. For each arrow, calculates source/target edge intersection points
/// 3. Generates elbow routing points (90-degree corners)
/// 4. Staggers multiple arrows from the same edge
pub fn routeArrows(elements_json: []const u8, out: []u8) !usize {
    // Parse shapes and arrows from input
    var shapes: [128]Shape = undefined;
    var shape_count: usize = 0;
    var arrow_starts: [128]ArrowRef = undefined;
    var arrow_count: usize = 0;

    parseElements(elements_json, &shapes, &shape_count, &arrow_starts, &arrow_count);

    if (arrow_count == 0) return 0;

    // Count outgoing edges per source node for stagger computation
    var out_counts: [128]usize = [_]usize{0} ** 128;
    var out_indexes: [128]usize = [_]usize{0} ** 128;

    for (arrow_starts[0..arrow_count]) |arrow| {
        const idx = findShapeIdx(shapes[0..shape_count], arrow.from_id);
        if (idx) |i| {
            out_counts[i] += 1;
        }
    }

    // For each arrow, calculate endpoints
    var written: usize = 0;
    written += copySlice(out[written..], "[");
    var emitted: usize = 0;

    for (arrow_starts[0..arrow_count]) |arrow| {
        const from_shape = findShape(shapes[0..shape_count], arrow.from_id);
        const to_shape = findShape(shapes[0..shape_count], arrow.to_id);

        if (from_shape == null or to_shape == null) continue;

        if (emitted > 0) written += copySlice(out[written..], ",");
        emitted += 1;

        const src = from_shape.?;
        const tgt = to_shape.?;

        // Get stagger info for this arrow
        const src_idx = findShapeIdx(shapes[0..shape_count], arrow.from_id) orelse continue;
        const total_out = out_counts[src_idx];
        const edge_idx = out_indexes[src_idx];
        out_indexes[src_idx] += 1;

        // Stagger ratio: distribute arrows across edge
        const stagger = computeStagger(edge_idx, total_out);

        // Calculate edge points
        const src_cx = src.x + @divTrunc(src.w, 2);
        const src_cy = src.y + @divTrunc(src.h, 2);
        const tgt_cx = tgt.x + @divTrunc(tgt.w, 2);
        const tgt_cy = tgt.y + @divTrunc(tgt.h, 2);

        const d_x = tgt_cx - src_cx;
        const d_y = tgt_cy - src_cy;
        const abs_dx = if (d_x < 0) -d_x else d_x;
        const abs_dy = if (d_y < 0) -d_y else d_y;

        var sx: i32 = undefined;
        var sy: i32 = undefined;
        var tx: i32 = undefined;
        var ty: i32 = undefined;
        var source_edge: enum { top, bottom, left, right } = .bottom;
        var target_edge: enum { top, bottom, left, right } = .top;

        if (abs_dy > abs_dx) {
            // Vertical
            if (d_y > 0) {
                source_edge = .bottom;
                target_edge = .top;
                sx = src.x + applyStagger(src.w, stagger);
                sy = src.y + src.h;
                tx = tgt.x + applyStagger(tgt.w, stagger);
                ty = tgt.y;
            } else {
                source_edge = .top;
                target_edge = .bottom;
                sx = src.x + applyStagger(src.w, stagger);
                sy = src.y;
                tx = tgt.x + applyStagger(tgt.w, stagger);
                ty = tgt.y + tgt.h;
            }
        } else {
            // Horizontal
            if (d_x > 0) {
                source_edge = .right;
                target_edge = .left;
                sx = src.x + src.w;
                sy = src.y + applyStagger(src.h, stagger);
                tx = tgt.x;
                ty = tgt.y + applyStagger(tgt.h, stagger);
            } else {
                source_edge = .left;
                target_edge = .right;
                sx = src.x;
                sy = src.y + applyStagger(src.h, stagger);
                tx = tgt.x + tgt.w;
                ty = tgt.y + applyStagger(tgt.h, stagger);
            }
        }

        const dx = tx - sx;
        const dy = ty - sy;

        // Elbow routing
        written += copySlice(out[written..], "{\"id\":\"");
        written += copySlice(out[written..], arrow.id_slice);
        written += copySlice(out[written..], "\",\"x\":");
        written += writeInt(out[written..], sx);
        written += copySlice(out[written..], ",\"y\":");
        written += writeInt(out[written..], sy);
        written += copySlice(out[written..], ",\"points\":");

        if (source_edge == .bottom and target_edge == .top) {
            if (abs_dx < 10) {
                // Straight vertical
                written += copySlice(out[written..], "[[0,0],[0,");
                written += writeInt(out[written..], dy);
                written += copySlice(out[written..], "]]");
            } else {
                // Elbow: down, across, down
                const mid_y = @divTrunc(dy, 2);
                written += copySlice(out[written..], "[[0,0],[0,");
                written += writeInt(out[written..], mid_y);
                written += copySlice(out[written..], "],[");
                written += writeInt(out[written..], dx);
                written += copySlice(out[written..], ",");
                written += writeInt(out[written..], mid_y);
                written += copySlice(out[written..], "],[");
                written += writeInt(out[written..], dx);
                written += copySlice(out[written..], ",");
                written += writeInt(out[written..], dy);
                written += copySlice(out[written..], "]]");
            }
        } else if (source_edge == .right and target_edge == .left) {
            if (abs_dy < 10) {
                // Straight horizontal
                written += copySlice(out[written..], "[[0,0],[");
                written += writeInt(out[written..], dx);
                written += copySlice(out[written..], ",0]]");
            } else {
                // Elbow: right, down, right
                const mid_x = @divTrunc(dx, 2);
                written += copySlice(out[written..], "[[0,0],[");
                written += writeInt(out[written..], mid_x);
                written += copySlice(out[written..], ",0],[");
                written += writeInt(out[written..], mid_x);
                written += copySlice(out[written..], ",");
                written += writeInt(out[written..], dy);
                written += copySlice(out[written..], "],[");
                written += writeInt(out[written..], dx);
                written += copySlice(out[written..], ",");
                written += writeInt(out[written..], dy);
                written += copySlice(out[written..], "]]");
            }
        } else {
            // Straight line for other cases
            written += copySlice(out[written..], "[[0,0],[");
            written += writeInt(out[written..], dx);
            written += copySlice(out[written..], ",");
            written += writeInt(out[written..], dy);
            written += copySlice(out[written..], "]]");
        }

        written += copySlice(out[written..], "}");
    }

    written += copySlice(out[written..], "]");
    return written;
}

/// Compute stagger ratio for an arrow at given index out of total count.
/// Ratios: [0.5, 0.35, 0.65, 0.2, 0.8] for indices 0..4, then 0.5 fallback.
fn computeStagger(idx: usize, total: usize) i32 {
    if (total <= 1) return 50; // center (represents 0.5 * 100)
    const positions = [_]i32{ 50, 35, 65, 20, 80 };
    if (idx < positions.len) return positions[idx];
    return 50;
}

/// Apply stagger ratio (0-100) to a dimension.
fn applyStagger(dim: i32, stagger: i32) i32 {
    return @divTrunc(dim * stagger, 100);
}

const Shape = struct {
    id_slice: []const u8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
};

const ArrowRef = struct {
    id_slice: []const u8,
    from_id: []const u8,
    to_id: []const u8,
};

fn parseElements(
    json: []const u8,
    shapes: *[128]Shape,
    shape_count: *usize,
    arrows_out: *[128]ArrowRef,
    arrow_count: *usize,
) void {
    var pos: usize = 0;

    while (pos < json.len) : (pos += 1) {
        if (json[pos] != '{') continue;

        const obj_end = findMatchingBrace(json[pos..]) + pos;
        const obj = json[pos..obj_end];

        const elem_type = extractStringField(obj, "type");
        if (elem_type == null) continue;

        if (std.mem.eql(u8, elem_type.?, "arrow")) {
            if (arrow_count.* < 128) {
                // Extract startBinding.elementId and endBinding.elementId
                const from_id = extractNestedStringField(obj, "startBinding", "elementId");
                const to_id = extractNestedStringField(obj, "endBinding", "elementId");
                arrows_out[arrow_count.*] = .{
                    .id_slice = extractStringField(obj, "id") orelse &.{},
                    .from_id = from_id orelse &.{},
                    .to_id = to_id orelse &.{},
                };
                arrow_count.* += 1;
            }
        } else if (std.mem.eql(u8, elem_type.?, "rectangle") or std.mem.eql(u8, elem_type.?, "ellipse")) {
            if (shape_count.* < 128) {
                shapes[shape_count.*] = .{
                    .id_slice = extractStringField(obj, "id") orelse &.{},
                    .x = extractIntField(obj, "x") orelse 0,
                    .y = extractIntField(obj, "y") orelse 0,
                    .w = extractIntField(obj, "width") orelse 180,
                    .h = extractIntField(obj, "height") orelse 80,
                };
                shape_count.* += 1;
            }
        }

        pos = obj_end;
    }
}

fn findShape(shapes: []const Shape, id: []const u8) ?Shape {
    for (shapes) |s| {
        if (std.mem.eql(u8, s.id_slice, id)) return s;
    }
    return null;
}

fn findShapeIdx(shapes: []const Shape, id: []const u8) ?usize {
    for (shapes, 0..) |s, i| {
        if (std.mem.eql(u8, s.id_slice, id)) return i;
    }
    return null;
}

const findMatchingBrace = util.findMatchingBrace;
const extractStringField = util.extractStringField;
const extractNestedStringField = util.extractNestedStringField;
const extractIntField = util.extractIntField;
const copySlice = util.copySlice;
const writeInt = util.writeInt;

test "arrow routing basic" {
    const elements =
        \\[{"id":"box1","type":"rectangle","x":100,"y":100,"width":180,"height":80},
        \\{"id":"box2","type":"rectangle","x":100,"y":400,"width":180,"height":80},
        \\{"id":"arr1","type":"arrow","startBinding":{"elementId":"box1"},"endBinding":{"elementId":"box2"}}]
    ;
    var out: [4096]u8 = undefined;
    const written = try routeArrows(elements, &out);
    try std.testing.expect(written > 0);
}

test "arrow elbow routing with horizontal offset" {
    // b1 at (100,100), b2 at (400,400) — vertical relationship (dy=300, dx=300, equal so horizontal)
    // source_edge=right, target_edge=left, abs_dy > 10 → elbow with 4 points
    const elements =
        \\[{"id":"b1","type":"rectangle","x":100,"y":100,"width":180,"height":80},
        \\{"id":"b2","type":"rectangle","x":400,"y":400,"width":180,"height":80},
        \\{"id":"a1","type":"arrow","startBinding":{"elementId":"b1"},"endBinding":{"elementId":"b2"}}]
    ;
    var out: [4096]u8 = undefined;
    const written = try routeArrows(elements, &out);
    try std.testing.expect(written > 0);
    const result = out[0..written];
    // Elbow routing produces 4 points: [[0,0],[mid,0],[mid,dy],[dx,dy]]
    // Count opening brackets '[' within the points value — 4 points = 5 brackets (outer + 4 inner)
    var bracket_count: usize = 0;
    var found_points = false;
    for (result, 0..) |c, idx| {
        if (idx + 7 < result.len and std.mem.eql(u8, result[idx .. idx + 7], "points\"")) {
            found_points = true;
        }
        if (found_points and c == '[') bracket_count += 1;
        // Stop at the closing of the points array (two consecutive ]s)
        if (found_points and bracket_count > 0 and c == '}') break;
    }
    // At least 5 brackets: outer [ + 4 inner [  (4-point elbow)
    try std.testing.expect(bracket_count >= 5);
}

test "arrow stagger with multiple arrows" {
    // Two arrows from same source — should have different positions
    const elements =
        \\[{"id":"s1","type":"rectangle","x":100,"y":100,"width":180,"height":80},
        \\{"id":"t1","type":"rectangle","x":100,"y":400,"width":180,"height":80},
        \\{"id":"t2","type":"rectangle","x":400,"y":400,"width":180,"height":80},
        \\{"id":"a1","type":"arrow","startBinding":{"elementId":"s1"},"endBinding":{"elementId":"t1"}},
        \\{"id":"a2","type":"arrow","startBinding":{"elementId":"s1"},"endBinding":{"elementId":"t2"}}]
    ;
    var out: [8192]u8 = undefined;
    const written = try routeArrows(elements, &out);
    try std.testing.expect(written > 0);
    // Both arrows should appear in output
    const result = out[0..written];
    try std.testing.expect(std.mem.indexOf(u8, result, "\"a1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"a2\"") != null);
}
