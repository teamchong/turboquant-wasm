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
    const idx: usize = @intCast(handle);
    if (idx >= MAX_ENGINES) return;
    if (engine_slots[idx]) |engine_ptr| {
        engine_ptr.deinit(wasm_allocator);
        wasm_allocator.destroy(engine_ptr);
        engine_slots[idx] = null;
    }
}

/// Encode a vector. Returns pointer to compressed data, writes length to out_len.
/// Caller must free the returned pointer with tq_free.
export fn tq_encode(handle: i32, input_ptr: [*]const f32, dim: u32, out_len: *u32) ?[*]u8 {
    const idx: usize = @intCast(handle);
    if (idx >= MAX_ENGINES) return null;
    const engine_ptr = engine_slots[idx] orelse return null;

    const input = input_ptr[0..dim];
    const compressed = engine_ptr.encode(wasm_allocator, input) catch return null;
    out_len.* = @intCast(compressed.len);
    return compressed.ptr;
}

/// Decode compressed data back to floats. Returns pointer to f32 array.
/// Caller must free the returned pointer with tq_free.
export fn tq_decode(handle: i32, compressed_ptr: [*]const u8, compressed_len: u32, out_len: *u32) ?[*]f32 {
    const idx: usize = @intCast(handle);
    if (idx >= MAX_ENGINES) return null;
    const engine_ptr = engine_slots[idx] orelse return null;

    const compressed = compressed_ptr[0..compressed_len];
    const decoded = engine_ptr.decode(wasm_allocator, compressed) catch return null;
    out_len.* = @intCast(decoded.len);
    return decoded.ptr;
}

/// Fast dot product between a query vector and compressed data.
export fn tq_dot(handle: i32, query_ptr: [*]const f32, dim: u32, compressed_ptr: [*]const u8, compressed_len: u32) f32 {
    const idx: usize = @intCast(handle);
    if (idx >= MAX_ENGINES) return 0;
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
    const idx: usize = @intCast(handle);
    if (idx >= MAX_ENGINES) return;
    const engine_ptr = engine_slots[idx] orelse return;

    const q = query_ptr[0..dim];

    // Pre-rotate query once (engine.dot rotates every call)
    engine_ptr.rot_op.rotate(q, engine_ptr.scratch_rotated);
    const rotated_q = engine_ptr.scratch_rotated;

    // Parse first header to get fixed layout (polar_bytes, qjl_bytes offsets)
    // All vectors share the same dim/structure — only max_r and gamma differ.
    const first = compressed_ptr[0..bytes_per_vector];
    const h0 = turboquant.format.readHeader(first) catch return;
    const polar_len = h0.polar_bytes;
    const qjl_len = h0.qjl_bytes;
    const polar_off = turboquant.format.HEADER_SIZE;
    const qjl_off = polar_off + polar_len;

    for (0..num_vectors) |i| {
        const base = i * bytes_per_vector;
        const blob = compressed_ptr[base .. base + bytes_per_vector];

        // Read only max_r and gamma directly (bytes 14-21), skip full header parse
        const max_r: f32 = @bitCast(std.mem.readInt(u32, blob[14..18], .little));
        const gamma: f32 = @bitCast(std.mem.readInt(u32, blob[18..22], .little));

        const polar_data = blob[polar_off .. polar_off + polar_len];
        const qjl_data = blob[qjl_off .. qjl_off + qjl_len];

        const polar_sum = turboquant.polar.dotProduct(rotated_q, polar_data, max_r);
        const qjl_sum = turboquant.qjl.estimateDotWithWorkspace(rotated_q, qjl_data, gamma, &engine_ptr.qjl_workspace);

        out_scores[i] = polar_sum + qjl_sum;
    }
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
