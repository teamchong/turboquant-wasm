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
