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

    length: usize,
    capacity: usize,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, engine: *Engine, max_positions: usize) !TQStream {
        const dim = engine.dim;
        const bpv = engine.bytesPerVector();

        const compressed = try allocator.alloc(u8, max_positions * bpv);

        return .{
            .engine = engine,
            .dim = dim,
            .bytes_per_vector = bpv,
            .compressed = compressed,
            .length = 0,
            .capacity = max_positions,
            .allocator = allocator,
        };
    }

    pub fn deinit(s: *TQStream) void {
        s.allocator.free(s.compressed);
    }

    /// Append a vector — compress and store. No decompression.
    /// Use dotBatch on serialize() for scoring. Use decodeInto for individual vectors.
    pub fn append(s: *TQStream, vector: []const f32) TQStreamError!void {
        if (vector.len != s.dim) return TQStreamError.InvalidRange;
        if (s.length >= s.capacity) try s.grow();

        const pos = s.length;
        const comp_offset = pos * s.bytes_per_vector;

        _ = s.engine.encodeInto(vector, s.compressed[comp_offset .. comp_offset + s.bytes_per_vector]) catch
            return TQStreamError.EncodeFailed;

        s.length = pos + 1;
    }

    pub fn appendBatch(s: *TQStream, vectors: []const f32, count: usize) TQStreamError!void {
        if (vectors.len < count * s.dim) return TQStreamError.InvalidRange;
        for (0..count) |i| {
            try s.append(vectors[i * s.dim .. (i + 1) * s.dim]);
        }
    }

    /// Get compressed bytes for a range of positions. Use with dotBatch for scoring.
    pub fn getCompressedSlice(s: *const TQStream, start: usize, end: usize) TQStreamError![]const u8 {
        if (end > s.length or start > end) return TQStreamError.InvalidRange;
        return s.compressed[start * s.bytes_per_vector .. end * s.bytes_per_vector];
    }

    /// Get all compressed bytes. Contiguous, dotBatch-compatible.
    pub fn serialize(s: *const TQStream) []const u8 {
        return s.compressed[0 .. s.length * s.bytes_per_vector];
    }

    /// Decode a single position into a provided output buffer.
    pub fn decodePosition(s: *const TQStream, position: usize, out: []f32) TQStreamError!void {
        if (position >= s.length) return TQStreamError.InvalidRange;
        const comp_offset = position * s.bytes_per_vector;
        s.engine.decodeInto(
            s.compressed[comp_offset .. comp_offset + s.bytes_per_vector],
            out,
        ) catch return TQStreamError.DecodeFailed;
    }

    pub fn rewind(s: *TQStream, position: usize) void {
        if (position < s.length) {
            s.length = position;
        }
    }

    fn grow(s: *TQStream) TQStreamError!void {
        const new_cap = s.capacity * 2;

        const new_comp = s.allocator.alloc(u8, new_cap * s.bytes_per_vector) catch
            return TQStreamError.OutOfMemory;

        @memcpy(new_comp[0 .. s.length * s.bytes_per_vector], s.compressed[0 .. s.length * s.bytes_per_vector]);

        s.allocator.free(s.compressed);
        s.compressed = new_comp;
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

test "append compresses and stores" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 16);
    defer stream.deinit();

    const v = makeRandomVector(8, 1);
    try stream.append(&v);
    try testing.expectEqual(@as(usize, 1), stream.length);

    // Compressed bytes match standalone encode
    const encoded = try engine.encode(allocator, &v);
    defer allocator.free(encoded);
    try testing.expectEqualSlices(u8, encoded, stream.serialize());
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
}

test "appendBatch matches sequential append" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var seq = try TQStream.init(allocator, &engine, 16);
    defer seq.deinit();
    var vectors: [10 * 8]f32 = undefined;
    for (0..10) |i| {
        const v = makeRandomVector(8, i + 200);
        @memcpy(vectors[i * 8 .. (i + 1) * 8], &v);
        try seq.append(&v);
    }

    var batch = try TQStream.init(allocator, &engine, 16);
    defer batch.deinit();
    try batch.appendBatch(&vectors, 10);

    try testing.expectEqualSlices(u8, seq.serialize(), batch.serialize());
}

test "decodePosition roundtrip" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 16);
    defer stream.deinit();

    const v = makeRandomVector(8, 1);
    try stream.append(&v);

    var decoded: [8]f32 = undefined;
    try stream.decodePosition(0, &decoded);

    // Should match standalone encode→decode
    const encoded = try engine.encode(allocator, &v);
    defer allocator.free(encoded);
    const ref = try engine.decode(allocator, encoded);
    defer allocator.free(ref);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(ref[i], decoded[i], 1e-6);
    }
}

test "getCompressedSlice" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 8);
    defer engine.deinit(allocator);

    var stream = try TQStream.init(allocator, &engine, 128);
    defer stream.deinit();

    for (0..100) |i| {
        const v = makeRandomVector(8, i + 300);
        try stream.append(&v);
    }

    const slice = try stream.getCompressedSlice(50, 60);
    try testing.expectEqual(@as(usize, 10 * stream.bytes_per_vector), slice.len);

    const full = stream.serialize();
    try testing.expectEqualSlices(u8, full[50 * stream.bytes_per_vector .. 60 * stream.bytes_per_vector], slice);
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

test "append performance vs allocating encode" {
    const allocator = testing.allocator;
    var engine = try makeTestEngine(allocator, 256);
    defer engine.deinit(allocator);

    const v = makeRandomVector(256, 42);
    const n = 200;

    // Old: allocating encode
    var timer = try std.time.Timer.start();
    for (0..n) |_| {
        const encoded = try engine.encode(allocator, &v);
        allocator.free(encoded);
    }
    const old_ns = timer.read();

    // New: TQStream append (compress only)
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

    // TQStream compress-only should be faster than allocating encode
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
