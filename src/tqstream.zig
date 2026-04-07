const std = @import("std");
const turboquant = @import("turboquant.zig");
const format = @import("format.zig");
const Engine = turboquant.Engine;

pub const TQStreamError = error{
    OutOfMemory,
    EvictedRange,
    InvalidRange,
    EncodeFailed,
    DecodeFailed,
};

pub const TQStream = struct {
    engine: *Engine,
    dim: usize,
    bytes_per_vector: usize,

    compressed: []u8,
    decompressed: []f32,

    length: usize,
    capacity: usize,
    evict_start: usize,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, engine: *Engine, max_positions: usize) !TQStream {
        const dim = engine.dim;
        const bpv = engine.bytesPerVector();

        const compressed = try allocator.alloc(u8, max_positions * bpv);
        errdefer allocator.free(compressed);
        const decompressed = try allocator.alloc(f32, max_positions * dim);
        errdefer allocator.free(decompressed);

        return .{
            .engine = engine,
            .dim = dim,
            .bytes_per_vector = bpv,
            .compressed = compressed,
            .decompressed = decompressed,
            .length = 0,
            .capacity = max_positions,
            .evict_start = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(s: *TQStream) void {
        s.allocator.free(s.compressed);
        s.allocator.free(s.decompressed);
    }

    pub fn append(s: *TQStream, vector: []const f32) TQStreamError!void {
        if (vector.len != s.dim) return TQStreamError.InvalidRange;
        if (s.length >= s.capacity) try s.grow();

        const pos = s.length;
        const comp_offset = pos * s.bytes_per_vector;
        const decomp_offset = pos * s.dim;

        // Encode directly into compressed store
        _ = s.engine.encodeInto(vector, s.compressed[comp_offset .. comp_offset + s.bytes_per_vector]) catch
            return TQStreamError.EncodeFailed;

        // Decode directly into decompressed buffer
        s.engine.decodeInto(
            s.compressed[comp_offset .. comp_offset + s.bytes_per_vector],
            s.decompressed[decomp_offset .. decomp_offset + s.dim],
        ) catch return TQStreamError.DecodeFailed;

        s.length = pos + 1;
    }

    pub fn appendBatch(s: *TQStream, vectors: []const f32, count: usize) TQStreamError!void {
        if (vectors.len < count * s.dim) return TQStreamError.InvalidRange;
        for (0..count) |i| {
            try s.append(vectors[i * s.dim .. (i + 1) * s.dim]);
        }
    }

    pub fn getDecompressed(s: *const TQStream, start: usize, end: usize) TQStreamError![]const f32 {
        if (start < s.evict_start) return TQStreamError.EvictedRange;
        if (end > s.length or start > end) return TQStreamError.InvalidRange;
        return s.decompressed[start * s.dim .. end * s.dim];
    }

    pub fn getCompressedSlice(s: *const TQStream, start: usize, end: usize) TQStreamError![]const u8 {
        if (end > s.length or start > end) return TQStreamError.InvalidRange;
        return s.compressed[start * s.bytes_per_vector .. end * s.bytes_per_vector];
    }

    pub fn serialize(s: *const TQStream) []const u8 {
        return s.compressed[0 .. s.length * s.bytes_per_vector];
    }

    /// Evict decompressed data for positions [0, up_to).
    /// Compressed data is retained for all positions.
    /// Evicted positions return EvictedRange from getDecompressed.
    pub fn evictFront(s: *TQStream, up_to: usize) void {
        if (up_to > s.evict_start and up_to <= s.length) {
            s.evict_start = up_to;
        }
    }

    pub fn rewind(s: *TQStream, position: usize) void {
        if (position < s.length) {
            s.length = position;
            if (s.evict_start > position) s.evict_start = position;
        }
    }

    fn grow(s: *TQStream) TQStreamError!void {
        const new_cap = s.capacity * 2;

        const new_comp = s.allocator.alloc(u8, new_cap * s.bytes_per_vector) catch
            return TQStreamError.OutOfMemory;

        const new_decomp = s.allocator.alloc(f32, new_cap * s.dim) catch {
            s.allocator.free(new_comp);
            return TQStreamError.OutOfMemory;
        };

        @memcpy(new_comp[0 .. s.length * s.bytes_per_vector], s.compressed[0 .. s.length * s.bytes_per_vector]);
        @memcpy(new_decomp[0 .. s.length * s.dim], s.decompressed[0 .. s.length * s.dim]);

        s.allocator.free(s.compressed);
        s.allocator.free(s.decompressed);
        s.compressed = new_comp;
        s.decompressed = new_decomp;
        s.capacity = new_cap;
    }
};

// Reexport for convenience
pub fn computeBytesPerVector(dim: usize) usize {
    return format.HEADER_SIZE + @import("polar.zig").polarBytesNeeded(dim) + @import("qjl.zig").qjlBytesNeeded(dim);
}

// ============================================================
// Tests
// ============================================================

const testing = std.testing;

fn makeTestEngine(allocator: std.mem.Allocator, dim: usize) !Engine {
    return Engine.init(allocator, .{ .dim = dim, .seed = 42 });
}

fn makeRandomVector(comptime dim: usize, seed: u64) [dim]f32 {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    var v: [dim]f32 = undefined;
    for (&v) |*x| x.* = r.float(f32) * 2.0 - 1.0;
    return v;
}

test "append single vector roundtrip" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 16);
    defer stream.deinit();

    const v = makeRandomVector(8, 1);
    try stream.append(&v);
    try testing.expectEqual(@as(usize, 1), stream.length);

    // Verify decompressed matches standalone encode→decode
    const encoded = try engine.encode(allocator, &v);
    defer allocator.free(encoded);
    const decoded = try engine.decode(allocator, encoded);
    defer allocator.free(decoded);

    const stream_decoded = try stream.getDecompressed(0, 1);
    for (0..8) |i| {
        try testing.expectApproxEqAbs(decoded[i], stream_decoded[i], 1e-6);
    }
}

test "append 100 vectors" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 16);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 128);
    defer stream.deinit();

    for (0..100) |i| {
        const v = makeRandomVector(16, i + 100);
        try stream.append(&v);
    }

    try testing.expectEqual(@as(usize, 100), stream.length);
    try testing.expectEqual(@as(usize, 100 * stream.bytes_per_vector), stream.serialize().len);

    const all = try stream.getDecompressed(0, 100);
    for (all) |val| {
        try testing.expect(std.math.isFinite(val));
    }
}

test "appendBatch matches sequential append" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    // Sequential
    var seq = try TQStream.init(allocator, &engine, 16);
    defer seq.deinit();
    var vectors: [10 * 8]f32 = undefined;
    for (0..10) |i| {
        const v = makeRandomVector(8, i + 200);
        @memcpy(vectors[i * 8 .. (i + 1) * 8], &v);
        try seq.append(&v);
    }

    // Batch
    var batch = try TQStream.init(allocator, &engine, 16);
    defer batch.deinit();
    try batch.appendBatch(&vectors, 10);

    // Compressed bytes should be identical
    try testing.expectEqualSlices(u8, seq.serialize(), batch.serialize());
}

test "getDecompressed slice" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 128);
    defer stream.deinit();

    for (0..100) |i| {
        const v = makeRandomVector(8, i + 300);
        try stream.append(&v);
    }

    const slice = try stream.getDecompressed(50, 60);
    try testing.expectEqual(@as(usize, 10 * 8), slice.len);

    const full = try stream.getDecompressed(0, 100);
    try testing.expectEqualSlices(f32, full[50 * 8 .. 60 * 8], slice);
}

test "evict front" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 128);
    defer stream.deinit();

    for (0..100) |i| {
        const v = makeRandomVector(8, i + 400);
        try stream.append(&v);
    }

    stream.evictFront(50);
    try testing.expectEqual(@as(usize, 50), stream.evict_start);

    // Compressed data still accessible for all 100
    const comp = try stream.getCompressedSlice(0, 100);
    try testing.expectEqual(@as(usize, 100 * stream.bytes_per_vector), comp.len);

    // Decompressed access to evicted range returns error
    try testing.expectError(TQStreamError.EvictedRange, stream.getDecompressed(0, 10));

    // Non-evicted range still accessible
    const valid = try stream.getDecompressed(50, 100);
    try testing.expectEqual(@as(usize, 50 * 8), valid.len);
}

test "rewind truncates" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 64);
    defer stream.deinit();

    for (0..50) |i| {
        const v = makeRandomVector(8, i + 500);
        try stream.append(&v);
    }

    stream.rewind(25);
    try testing.expectEqual(@as(usize, 25), stream.length);
    try testing.expectEqual(@as(usize, 25 * stream.bytes_per_vector), stream.serialize().len);
}

test "serialize is dotBatch compatible" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 32);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 64);
    defer stream.deinit();

    var vecs: [10][32]f32 = undefined;
    for (0..10) |i| {
        vecs[i] = makeRandomVector(32, i + 600);
        try stream.append(&vecs[i]);
    }

    // Use serialize() output with dotBatch
    const query = makeRandomVector(32, 999);
    var scores: [10]f32 = undefined;
    engine.dotBatch(&query, stream.serialize(), stream.bytes_per_vector, 10, &scores);

    // Compare with individual dot calls
    for (0..10) |i| {
        const comp = try stream.getCompressedSlice(i, i + 1);
        const single_score = engine.dot(&query, comp);
        try testing.expectApproxEqAbs(single_score, scores[i], 1e-5);
    }
}

test "computeBytesPerVector matches encode" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 8, 16, 32, 64, 128, 256 };
    for (dims) |dim| {
        var engine = try makeTestEngine(allocator, dim);
        defer engine.deinit(allocator);

        const computed = computeBytesPerVector(dim);
        const actual = engine.bytesPerVector();
        try testing.expectEqual(computed, actual);

        const v = makeRandomVector(256, dim)[0..dim];
        const encoded = try engine.encode(allocator, v);
        defer allocator.free(encoded);
        try testing.expectEqual(computed, encoded.len);
    }
}

test "evicted range returns error" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 32);
    defer stream.deinit();

    for (0..20) |i| {
        const v = makeRandomVector(8, i + 700);
        try stream.append(&v);
    }

    stream.evictFront(10);
    try testing.expectError(TQStreamError.EvictedRange, stream.getDecompressed(0, 5));
    try testing.expectError(TQStreamError.EvictedRange, stream.getDecompressed(5, 15));
    _ = try stream.getDecompressed(10, 20); // should succeed
}

test "append performance vs allocating encode+decode" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 256);
    defer engine.deinit(allocator);

    const v = makeRandomVector(256, 42);
    const n = 200;

    // Old: allocating encode + decode
    var timer = try std.time.Timer.start();
    for (0..n) |_| {
        const encoded = try engine.encode(allocator, &v);
        const decoded = try engine.decode(allocator, encoded);
        allocator.free(decoded);
        allocator.free(encoded);
    }
    const old_ns = timer.read();

    // New: TQStream append
    var stream = try TQStream.init(allocator, &engine, n + 16);
    defer stream.deinit();

    timer = try std.time.Timer.start();
    for (0..n) |_| {
        try stream.append(&v);
    }
    const new_ns = timer.read();

    const old_us = @as(f64, @floatFromInt(old_ns)) / 1000.0 / @as(f64, @floatFromInt(n));
    const new_us = @as(f64, @floatFromInt(new_ns)) / 1000.0 / @as(f64, @floatFromInt(n));

    std.debug.print("\n  dim=256 n={}: old={d:.1}us/vec stream={d:.1}us/vec ratio={d:.2}x\n", .{ n, old_us, new_us, old_us / new_us });

    // TQStream should not be more than 2x slower than allocating path
    try testing.expect(new_us < old_us * 2.0);
}

test "encodeInto matches encode byte-for-byte" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 64);
    defer engine.deinit(allocator);

    const v = makeRandomVector(64, 42);

    // Allocating encode
    const encoded = try engine.encode(allocator, &v);
    defer allocator.free(encoded);

    // Zero-alloc encodeInto
    var buf: [256]u8 = undefined;
    const written = try engine.encodeInto(&v, &buf);

    try testing.expectEqual(encoded.len, written);
    try testing.expectEqualSlices(u8, encoded, buf[0..written]);
}
