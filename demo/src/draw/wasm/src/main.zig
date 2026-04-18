const std = @import("std");
const layout = @import("layout.zig");
const arrows = @import("arrows.zig");
const validate_mod = @import("validate.zig");
const compress_mod = @import("compress.zig");

/// Bump allocator backed by a fixed buffer (no imports needed for WASM).
/// 64MB heap for element data and layout scratch.
var heap_buf: [64 * 1024 * 1024]u8 = undefined;
var heap_offset: usize = 0;

export fn alloc(size: usize) usize {
    const aligned = std.mem.alignForward(usize, heap_offset, 8);
    if (aligned + size > heap_buf.len) return 0;
    heap_offset = aligned + size;
    return @intFromPtr(&heap_buf[aligned]);
}

export fn dealloc(_: usize, _: usize) void {
    // Bump allocator — no individual dealloc
}

export fn resetHeap() void {
    heap_offset = 0;
}

/// Auto-layout: position nodes in a layered graph layout.
/// Input: nodes JSON + edges JSON + groups JSON. Output: positioned nodes + edge routes JSON.
export fn layoutGraph(
    nodes_ptr: [*]const u8,
    nodes_len: usize,
    edges_ptr: [*]const u8,
    edges_len: usize,
    groups_ptr: [*]const u8,
    groups_len: usize,
    out_ptr: [*]u8,
    out_cap: usize,
    opts_ptr: [*]const u8,
    opts_len: usize,
) usize {
    const nodes_slice = nodes_ptr[0..nodes_len];
    const edges_slice = edges_ptr[0..edges_len];
    const groups_slice = groups_ptr[0..groups_len];
    const opts_slice = opts_ptr[0..opts_len];
    const out_slice = out_ptr[0..out_cap];

    return layout.layoutGraph(nodes_slice, edges_slice, groups_slice, opts_slice, out_slice) catch 0;
}

/// Route arrows: calculate arrow endpoints and elbow points.
export fn routeArrows(
    elem_ptr: [*]const u8,
    elem_len: usize,
    out_ptr: [*]u8,
    out_cap: usize,
) usize {
    const elem_slice = elem_ptr[0..elem_len];
    const out_slice = out_ptr[0..out_cap];

    return arrows.routeArrows(elem_slice, out_slice) catch 0;
}

/// Validate Excalidraw elements for structural correctness.
export fn validate(
    elem_ptr: [*]const u8,
    elem_len: usize,
    out_ptr: [*]u8,
    out_cap: usize,
) usize {
    const elem_slice = elem_ptr[0..elem_len];
    const out_slice = out_ptr[0..out_cap];

    return validate_mod.validate(elem_slice, out_slice) catch 0;
}


/// Compress data using zlib format (matching pako.deflate).
export fn zlibCompress(
    in_ptr: [*]const u8,
    in_len: usize,
    out_ptr: [*]u8,
    out_cap: usize,
) usize {
    const in_slice = in_ptr[0..in_len];
    const out_slice = out_ptr[0..out_cap];

    return compress_mod.zlibCompress(in_slice, out_slice) catch 0;
}


test "alloc and reset" {
    resetHeap();
    const ptr = alloc(64);
    try std.testing.expect(ptr != 0);
    resetHeap();
    const ptr2 = alloc(64);
    try std.testing.expectEqual(ptr, ptr2);
}
