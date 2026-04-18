const std = @import("std");
const util = @import("util.zig");

/// Validate Excalidraw elements for structural correctness.
///
/// Checks:
/// 1. Every shape with boundElements has a matching text element with containerId
/// 2. No duplicate IDs
/// 3. Arrow startBinding/endBinding reference existing elements
/// 4. No overlapping shapes (within margin)
///
/// Output: JSON array of error objects: [{"type":"missing_text","id":"box_1","msg":"..."}]
/// Returns 0 if no errors.
pub fn validate(elements_json: []const u8, out: []u8) !usize {
    var ids: [256][]const u8 = undefined;
    var id_count: usize = 0;
    var container_ids: [256][]const u8 = undefined;
    var container_count: usize = 0;
    var bound_text_ids: [256][]const u8 = undefined;
    var bound_text_count: usize = 0;
    var binding_refs: [256][]const u8 = undefined;
    var binding_count: usize = 0;

    // Shape bounding boxes for overlap check
    var shape_rects: [256]ShapeRect = undefined;
    var shape_rect_count: usize = 0;

    // Parse all elements
    var pos: usize = 0;
    while (pos < elements_json.len) : (pos += 1) {
        if (elements_json[pos] != '{') continue;

        const obj_end = findMatchingBrace(elements_json[pos..]) + pos;
        const obj = elements_json[pos..obj_end];

        const id = extractStringField(obj, "id");
        if (id) |id_val| {
            if (id_count < 256) {
                ids[id_count] = id_val;
                id_count += 1;
            }
        }

        // Check for containerId (text elements bound to shapes)
        const container = extractStringField(obj, "containerId");
        if (container) |cid| {
            if (container_count < 256) {
                container_ids[container_count] = cid;
                container_count += 1;
            }
        }

        // Check for boundElements containing text type entry:
        // Must be an array (not null) and "text" must appear within the boundElements value.
        // Handles optional whitespace between "boundElements": and [
        const be_key_start = std.mem.indexOf(u8, obj, "\"boundElements\"");
        if (be_key_start) |bk_pos| {
            // Skip past key + any whitespace + colon + whitespace to find '['
            var bk_scan = bk_pos + "\"boundElements\"".len;
            while (bk_scan < obj.len and obj[bk_scan] == ' ') : (bk_scan += 1) {}
            if (bk_scan < obj.len and obj[bk_scan] == ':') {
                bk_scan += 1;
                while (bk_scan < obj.len and obj[bk_scan] == ' ') : (bk_scan += 1) {}
            }
            const is_array = bk_scan < obj.len and obj[bk_scan] == '[';
            const after_be = obj[bk_scan..];
            if (is_array and std.mem.indexOf(u8, after_be, "\"text\"") != null) {
                if (id) |id_val| {
                    if (bound_text_count < 256) {
                        bound_text_ids[bound_text_count] = id_val;
                        bound_text_count += 1;
                    }
                }
            }
        }

        // Check arrow bindings
        const elem_type = extractStringField(obj, "type");
        if (elem_type) |t| {
            if (std.mem.eql(u8, t, "arrow")) {
                const start_ref = extractNestedStringField(obj, "startBinding", "elementId");
                const end_ref = extractNestedStringField(obj, "endBinding", "elementId");
                if (start_ref) |r| {
                    if (binding_count < 256) {
                        binding_refs[binding_count] = r;
                        binding_count += 1;
                    }
                }
                if (end_ref) |r| {
                    if (binding_count < 256) {
                        binding_refs[binding_count] = r;
                        binding_count += 1;
                    }
                }
            }

            // Collect shape bounding boxes for overlap check
            // Skip group boundaries (dashed stroke + transparent background)
            const stroke_style = extractStringField(obj, "strokeStyle");
            const bg = extractStringField(obj, "backgroundColor");
            const is_group = stroke_style != null and bg != null and
                std.mem.eql(u8, stroke_style.?, "dashed") and
                std.mem.eql(u8, bg.?, "transparent");
            if ((std.mem.eql(u8, t, "rectangle") or std.mem.eql(u8, t, "ellipse")) and
                !is_group)
            {
                if (shape_rect_count < 256) {
                    shape_rects[shape_rect_count] = .{
                        .id_slice = id orelse &.{},
                        .x = extractIntField(obj, "x") orelse 0,
                        .y = extractIntField(obj, "y") orelse 0,
                        .w = extractIntField(obj, "width") orelse 0,
                        .h = extractIntField(obj, "height") orelse 0,
                    };
                    shape_rect_count += 1;
                }
            }
        }

        pos = obj_end;
    }

    // Run checks
    var written: usize = 0;
    var error_count: usize = 0;
    written += copySlice(out[written..], "[");

    // Check 1: shapes with boundElements have matching text
    for (bound_text_ids[0..bound_text_count]) |shape_id| {
        var found = false;
        for (container_ids[0..container_count]) |cid| {
            if (std.mem.eql(u8, cid, shape_id)) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (error_count > 0) written += copySlice(out[written..], ",");
            written += copySlice(out[written..], "{\"type\":\"missing_text\",\"id\":\"");
            written += copySliceJsonEscaped(out[written..], shape_id);
            written += copySlice(out[written..], "\",\"msg\":\"Shape has boundElements but no text element with matching containerId\"}");
            error_count += 1;
        }
    }

    // Check 2: duplicate IDs
    for (ids[0..id_count], 0..) |id_a, i| {
        for (ids[i + 1 .. id_count]) |id_b| {
            if (std.mem.eql(u8, id_a, id_b)) {
                if (error_count > 0) written += copySlice(out[written..], ",");
                written += copySlice(out[written..], "{\"type\":\"duplicate_id\",\"id\":\"");
                written += copySliceJsonEscaped(out[written..], id_a);
                written += copySlice(out[written..], "\",\"msg\":\"Duplicate element ID\"}");
                error_count += 1;
                break;
            }
        }
    }

    // Check 3: arrow bindings reference existing elements
    for (binding_refs[0..binding_count]) |ref| {
        if (ref.len == 0) continue;
        var found = false;
        for (ids[0..id_count]) |id_val| {
            if (std.mem.eql(u8, id_val, ref)) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (error_count > 0) written += copySlice(out[written..], ",");
            written += copySlice(out[written..], "{\"type\":\"dangling_ref\",\"id\":\"");
            written += copySliceJsonEscaped(out[written..], ref);
            written += copySlice(out[written..], "\",\"msg\":\"Arrow binding references non-existent element\"}");
            error_count += 1;
        }
    }

    // Check 4: overlapping shapes (5px margin tolerance)
    const margin: i32 = 5;
    for (shape_rects[0..shape_rect_count], 0..) |a, ai| {
        for (shape_rects[ai + 1 .. shape_rect_count]) |b| {
            // Check if bounding boxes overlap (allowing margin)
            const a_right = a.x + a.w - margin;
            const a_bottom = a.y + a.h - margin;
            const b_right = b.x + b.w - margin;
            const b_bottom = b.y + b.h - margin;

            const a_left = a.x + margin;
            const a_top = a.y + margin;
            const b_left = b.x + margin;
            const b_top = b.y + margin;

            if (a_left < b_right and a_right > b_left and
                a_top < b_bottom and a_bottom > b_top)
            {
                if (error_count > 0) written += copySlice(out[written..], ",");
                written += copySlice(out[written..], "{\"type\":\"overlap\",\"id\":\"");
                written += copySliceJsonEscaped(out[written..], a.id_slice);
                written += copySlice(out[written..], "\",\"msg\":\"Shape overlaps with ");
                written += copySliceJsonEscaped(out[written..], b.id_slice);
                written += copySlice(out[written..], "\"}");
                error_count += 1;
            }
        }
    }

    written += copySlice(out[written..], "]");

    // Return 0 if no errors (caller interprets as "all valid")
    if (error_count == 0) return 0;
    return written;
}

const ShapeRect = struct {
    id_slice: []const u8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
};

const findMatchingBrace = util.findMatchingBrace;
const extractStringField = util.extractStringField;
const extractNestedStringField = util.extractNestedStringField;
const extractIntField = util.extractIntField;
const copySlice = util.copySlice;
const copySliceJsonEscaped = util.copySliceJsonEscaped;

test "validate valid elements" {
    const elements =
        \\[{"id":"box1","type":"rectangle","boundElements":[{"type":"text","id":"box1-text"}]},
        \\{"id":"box1-text","type":"text","containerId":"box1"}]
    ;
    var out: [4096]u8 = undefined;
    const written = try validate(elements, &out);
    try std.testing.expectEqual(@as(usize, 0), written);
}

test "validate missing text" {
    const elements =
        \\[{"id":"box1","type":"rectangle","boundElements":[{"type":"text","id":"box1-text"}]}]
    ;
    var out: [4096]u8 = undefined;
    const written = try validate(elements, &out);
    try std.testing.expect(written > 0);
    try std.testing.expect(std.mem.indexOf(u8, out[0..written], "missing_text") != null);
}

test "validate overlap detection" {
    // Two rectangles at the same position should overlap
    const elements =
        \\[{"id":"a","type":"rectangle","x":100,"y":100,"width":180,"height":80},
        \\{"id":"b","type":"rectangle","x":110,"y":110,"width":180,"height":80}]
    ;
    var out: [4096]u8 = undefined;
    const written = try validate(elements, &out);
    try std.testing.expect(written > 0);
    try std.testing.expect(std.mem.indexOf(u8, out[0..written], "overlap") != null);
}

test "validate no overlap when far apart" {
    const elements =
        \\[{"id":"a","type":"rectangle","x":100,"y":100,"width":180,"height":80},
        \\{"id":"b","type":"rectangle","x":500,"y":500,"width":180,"height":80}]
    ;
    var out: [4096]u8 = undefined;
    const written = try validate(elements, &out);
    try std.testing.expectEqual(@as(usize, 0), written);
}
