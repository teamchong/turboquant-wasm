const std = @import("std");
const turboquant = @import("turboquant.zig");

// ---------------------------------------------------------------------------
// WASM memory allocator
// ---------------------------------------------------------------------------
const wasm_allocator = std.heap.wasm_allocator;

// ---------------------------------------------------------------------------
// Engine slot table — maps integer handles to Engine pointers
// ---------------------------------------------------------------------------
const MAX_ENGINES = 16;
var engine_slots: [MAX_ENGINES]?*turboquant.Engine = [_]?*turboquant.Engine{null} ** MAX_ENGINES;

fn findSlot() ?usize {
    for (engine_slots, 0..) |slot, i| {
        if (slot == null) return i;
    }
    return null;
}

/// Validate handle and return index. Returns null for negative or out-of-range handles.
fn resolveHandle(handle: i32) ?usize {
    if (handle < 0 or handle >= MAX_ENGINES) return null;
    return @intCast(handle);
}

// ---------------------------------------------------------------------------
// Exported C-ABI functions
// ---------------------------------------------------------------------------

/// Create an engine. Returns handle (>=0) on success, -1 on failure.
export fn tq_engine_create(dim: u32, seed: u32) i32 {
    const slot = findSlot() orelse return -1;

    const engine_ptr = wasm_allocator.create(turboquant.Engine) catch return -1;
    engine_ptr.* = turboquant.Engine.init(wasm_allocator, .{
        .dim = @intCast(dim),
        .seed = seed,
    }) catch {
        wasm_allocator.destroy(engine_ptr);
        return -1;
    };

    engine_slots[slot] = engine_ptr;
    return @intCast(slot);
}

/// Destroy an engine by handle.
export fn tq_engine_destroy(handle: i32) void {
    const idx = resolveHandle(handle) orelse return;
    if (engine_slots[idx]) |engine_ptr| {
        engine_ptr.deinit(wasm_allocator);
        wasm_allocator.destroy(engine_ptr);
        engine_slots[idx] = null;
    }
}

/// Encode a vector. Returns pointer to compressed data, writes length to out_len.
/// Caller must free the returned pointer with tq_free.
export fn tq_encode(handle: i32, input_ptr: [*]const f32, dim: u32, out_len: *u32) ?[*]u8 {
    const idx = resolveHandle(handle) orelse return null;
    const engine_ptr = engine_slots[idx] orelse return null;

    const input = input_ptr[0..dim];
    const compressed = engine_ptr.encode(wasm_allocator, input) catch return null;
    out_len.* = @intCast(compressed.len);
    return compressed.ptr;
}

/// Decode compressed data back to floats. Returns pointer to f32 array.
/// Caller must free the returned pointer with tq_free.
export fn tq_decode(handle: i32, compressed_ptr: [*]const u8, compressed_len: u32, out_len: *u32) ?[*]f32 {
    const idx = resolveHandle(handle) orelse return null;
    const engine_ptr = engine_slots[idx] orelse return null;

    const compressed = compressed_ptr[0..compressed_len];
    const decoded = engine_ptr.decode(wasm_allocator, compressed) catch return null;
    out_len.* = @intCast(decoded.len);
    return decoded.ptr;
}

/// Fast dot product between a query vector and compressed data.
export fn tq_dot(handle: i32, query_ptr: [*]const f32, dim: u32, compressed_ptr: [*]const u8, compressed_len: u32) f32 {
    const idx = resolveHandle(handle) orelse return 0;
    const engine_ptr = engine_slots[idx] orelse return 0;

    const q = query_ptr[0..dim];
    const compressed = compressed_ptr[0..compressed_len];
    return engine_ptr.dot(q, compressed);
}

/// Batch dot product: compute dot(query, compressed[i]) for i in 0..num_vectors.
/// compressed_ptr points to num_vectors * bytes_per_vector contiguous bytes.
/// Writes results to out_scores (caller-allocated f32 array of length num_vectors).
export fn tq_dot_batch(
    handle: i32,
    query_ptr: [*]const f32,
    dim: u32,
    compressed_ptr: [*]const u8,
    bytes_per_vector: u32,
    num_vectors: u32,
    out_scores: [*]f32,
) void {
    const idx = resolveHandle(handle) orelse return;
    const engine_ptr = engine_slots[idx] orelse return;

    const q = query_ptr[0..dim];
    const compressed = compressed_ptr[0 .. num_vectors * bytes_per_vector];
    const scores = out_scores[0..num_vectors];

    engine_ptr.dotBatch(q, compressed, bytes_per_vector, num_vectors, scores);
}

/// Allocate bytes in WASM linear memory (for JS to write into).
export fn tq_alloc(len: u32) ?[*]u8 {
    const slice = wasm_allocator.alloc(u8, len) catch return null;
    return slice.ptr;
}

/// Free a pointer previously returned by tq_encode, tq_decode, or tq_alloc.
export fn tq_free(ptr: [*]u8, len: u32) void {
    wasm_allocator.free(ptr[0..len]);
}

/// Allocate f32 array in WASM linear memory (for JS to write query vectors).
export fn tq_alloc_f32(count: u32) ?[*]f32 {
    const slice = wasm_allocator.alloc(f32, count) catch return null;
    return slice.ptr;
}

/// Free an f32 pointer.
export fn tq_free_f32(ptr: [*]f32, count: u32) void {
    wasm_allocator.free(ptr[0..count]);
}

test "resolveHandle rejects negative handles" {
    try std.testing.expectEqual(null, resolveHandle(-1));
    try std.testing.expectEqual(null, resolveHandle(-100));
    try std.testing.expectEqual(null, resolveHandle(std.math.minInt(i32)));
}

test "resolveHandle rejects out-of-range handles" {
    try std.testing.expectEqual(null, resolveHandle(MAX_ENGINES));
    try std.testing.expectEqual(null, resolveHandle(MAX_ENGINES + 1));
    try std.testing.expectEqual(null, resolveHandle(std.math.maxInt(i32)));
}

test "resolveHandle accepts valid handles" {
    try std.testing.expectEqual(@as(usize, 0), resolveHandle(0));
    try std.testing.expectEqual(@as(usize, 15), resolveHandle(MAX_ENGINES - 1));
}
