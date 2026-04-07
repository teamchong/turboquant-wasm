//! C-ABI bridge: exposes TQStream operations for ORT's C++ attention kernel.
//! Same WASM binary, shared linear memory, direct function calls.

const std = @import("std");
const turboquant = @import("turboquant.zig");
const tqstream = @import("tqstream.zig");
const Engine = turboquant.Engine;
const TQStream = tqstream.TQStream;

const wasm_allocator = std.heap.wasm_allocator;

// Global engine and stream slots for the attention kernel.
// One engine per head_dim, one stream per (layer, kv_type).
const MAX_ENGINES = 4;
const MAX_STREAMS = 256; // layers × (K + V)

var engines: [MAX_ENGINES]?*Engine = [_]?*Engine{null} ** MAX_ENGINES;
var engine_dims: [MAX_ENGINES]usize = [_]usize{0} ** MAX_ENGINES;
var streams: [MAX_STREAMS]?*TQStream = [_]?*TQStream{null} ** MAX_STREAMS;

fn getOrCreateEngine(dim: usize) ?*Engine {
    // Find existing
    for (engines, engine_dims) |e, d| {
        if (e != null and d == dim) return e;
    }
    // Create new
    for (&engines, &engine_dims) |slot, dim_slot| {
        if (slot.* == null) {
            const ptr = wasm_allocator.create(Engine) catch return null;
            ptr.* = Engine.init(wasm_allocator, .{ .dim = dim, .seed = 42 }) catch {
                wasm_allocator.destroy(ptr);
                return null;
            };
            slot.* = ptr;
            dim_slot.* = dim;
            return ptr;
        }
    }
    return null;
}

/// Create a TQStream for a KV cache layer. Returns stream ID or -1.
export fn tq_kv_create(head_dim: u32, max_positions: u32) i32 {
    const engine = getOrCreateEngine(head_dim) orelse return -1;
    for (streams, 0..) |*slot, i| {
        if (slot.* == null) {
            const ptr = wasm_allocator.create(TQStream) catch return -1;
            ptr.* = TQStream.init(wasm_allocator, engine, max_positions) catch {
                wasm_allocator.destroy(ptr);
                return -1;
            };
            slot.* = ptr;
            return @intCast(i);
        }
    }
    return -1;
}

/// Destroy a TQStream.
export fn tq_kv_destroy(stream_id: i32) void {
    if (stream_id < 0 or stream_id >= MAX_STREAMS) return;
    const idx: usize = @intCast(stream_id);
    if (streams[idx]) |ptr| {
        ptr.deinit();
        wasm_allocator.destroy(ptr);
        streams[idx] = null;
    }
}

/// Append a K or V vector to the stream. Returns 0 on success, -1 on failure.
export fn tq_kv_append(stream_id: i32, data_ptr: [*]const f32, dim: u32) i32 {
    if (stream_id < 0 or stream_id >= MAX_STREAMS) return -1;
    const idx: usize = @intCast(stream_id);
    const s = streams[idx] orelse return -1;
    s.append(data_ptr[0..dim]) catch return -1;
    return 0;
}

/// Compute dot products between query vector and all compressed K vectors.
/// Writes scores to out_scores. Returns number of scores written.
export fn tq_kv_dot_batch(
    stream_id: i32,
    query_ptr: [*]const f32,
    dim: u32,
    out_scores: [*]f32,
    max_scores: u32,
) i32 {
    if (stream_id < 0 or stream_id >= MAX_STREAMS) return -1;
    const idx: usize = @intCast(stream_id);
    const s = streams[idx] orelse return -1;
    if (s.length == 0) return 0;

    const n = @min(s.length, max_scores);
    const compressed = s.serialize();
    const q = query_ptr[0..dim];

    // Find the engine for this stream's dim
    const engine = s.engine;
    engine.dotBatch(q, compressed, s.bytes_per_vector, n, out_scores[0..n]);

    return @intCast(n);
}

/// Decode a single position from compressed store into output buffer.
export fn tq_kv_decode_position(stream_id: i32, position: u32, out_ptr: [*]f32, dim: u32) i32 {
    if (stream_id < 0 or stream_id >= MAX_STREAMS) return -1;
    const idx: usize = @intCast(stream_id);
    const s = streams[idx] orelse return -1;
    s.decodePosition(position, out_ptr[0..dim]) catch return -1;
    return 0;
}

/// Get the number of vectors stored in the stream.
export fn tq_kv_length(stream_id: i32) u32 {
    if (stream_id < 0 or stream_id >= MAX_STREAMS) return 0;
    const idx: usize = @intCast(stream_id);
    const s = streams[idx] orelse return 0;
    return @intCast(s.length);
}

/// Get compressed size in bytes.
export fn tq_kv_compressed_size(stream_id: i32) u32 {
    if (stream_id < 0 or stream_id >= MAX_STREAMS) return 0;
    const idx: usize = @intCast(stream_id);
    const s = streams[idx] orelse return 0;
    return @intCast(s.length * s.bytes_per_vector);
}
