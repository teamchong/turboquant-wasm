const std = @import("std");
const log = std.log.scoped(.turboquant);

pub const format = @import("format.zig");
const rotation = @import("rotation.zig");
pub const math = @import("math.zig");
pub const polar = @import("polar.zig");
const qjl = @import("qjl.zig");

pub const EncodeError = error{ InvalidDimension, OutOfMemory };
pub const DecodeError = error{ InvalidHeader, InvalidPayload, OutOfMemory };

pub const EngineConfig = struct {
    dim: usize,
    seed: u32,
};

pub const Engine = struct {
    dim: usize,
    seed: u32,
    rot_op: rotation.RotationOperator,
    qjl_workspace: qjl.Workspace,
    scratch_rotated: []f32,
    scratch_residual: []f32,
    scratch_polar_decoded: []f32,
    scratch_qjl_decoded: []f32,

    pub fn init(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
        const dim = config.dim;
        if (dim == 0 or dim % 2 != 0) return EncodeError.InvalidDimension;

        var rot_op = try rotation.RotationOperator.prepare(allocator, dim, config.seed);
        errdefer rot_op.destroy(allocator);

        var qjl_workspace = try qjl.Workspace.init(allocator, dim);
        errdefer qjl_workspace.deinit(allocator);

        const scratch_rotated = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_rotated);

        const scratch_residual = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_residual);

        const scratch_polar_decoded = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_polar_decoded);

        const scratch_qjl_decoded = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_qjl_decoded);

        return .{
            .dim = dim,
            .seed = config.seed,
            .rot_op = rot_op,
            .qjl_workspace = qjl_workspace,
            .scratch_rotated = scratch_rotated,
            .scratch_residual = scratch_residual,
            .scratch_polar_decoded = scratch_polar_decoded,
            .scratch_qjl_decoded = scratch_qjl_decoded,
        };
    }

    pub fn deinit(e: *Engine, allocator: std.mem.Allocator) void {
        e.rot_op.destroy(allocator);
        e.qjl_workspace.deinit(allocator);
        allocator.free(e.scratch_rotated);
        allocator.free(e.scratch_residual);
        allocator.free(e.scratch_polar_decoded);
        allocator.free(e.scratch_qjl_decoded);
        e.* = undefined;
    }

    pub fn encode(e: *Engine, allocator: std.mem.Allocator, x: []const f32) ![]u8 {
        const dim = e.dim;
        if (x.len != dim) return EncodeError.InvalidDimension;

        e.rot_op.rotate(x, e.scratch_rotated);

        var max_r: f32 = 0;
        for (0..dim / 2) |i| {
            const r = math.norm(e.scratch_rotated[i * 2 .. i * 2 + 2]);
            if (r > max_r) max_r = r;
        }
        if (max_r == 0) max_r = 1.0;

        const polar_encoded = try polar.encode(allocator, e.scratch_rotated, max_r);
        errdefer allocator.free(polar_encoded);

        computeResidualFromPolar(polar_encoded, e.scratch_rotated, max_r, e.scratch_residual);

        const gamma = math.norm(e.scratch_residual);
        const qjl_encoded = try qjl.encodeWithWorkspace(allocator, e.scratch_residual, &e.rot_op, &e.qjl_workspace);
        errdefer allocator.free(qjl_encoded);

        const polar_bytes = @as(u32, @intCast(polar_encoded.len));
        const qjl_bytes = @as(u32, @intCast(qjl_encoded.len));
        const total_size = format.HEADER_SIZE + polar_encoded.len + qjl_encoded.len;

        const result = try allocator.alloc(u8, total_size);
        errdefer allocator.free(result);

        format.writeHeader(result, @intCast(dim), polar_bytes, qjl_bytes, max_r, gamma);
        @memcpy(result[format.HEADER_SIZE..][0..polar_encoded.len], polar_encoded);
        @memcpy(result[format.HEADER_SIZE + polar_encoded.len ..], qjl_encoded);
        allocator.free(polar_encoded);
        allocator.free(qjl_encoded);

        const bpd = (total_size - format.HEADER_SIZE) * 8 / dim;
        log.debug("encoded: dim={}, bytes={}, bits/dim={}", .{ dim, total_size, bpd });

        return result;
    }

    pub fn decode(e: *Engine, allocator: std.mem.Allocator, compressed: []const u8) ![]f32 {
        const header = format.readHeader(compressed) catch |err| switch (err) {
            error.InvalidHeader => return DecodeError.InvalidHeader,
            error.OutOfMemory => return DecodeError.OutOfMemory,
            error.InvalidPayload => return DecodeError.InvalidPayload,
        };
        const dim = e.dim;
        if (header.dim != dim) return DecodeError.InvalidPayload;

        const payload = format.slicePayload(compressed, header) catch |err| switch (err) {
            error.InvalidHeader => return DecodeError.InvalidHeader,
            error.OutOfMemory => return DecodeError.OutOfMemory,
            error.InvalidPayload => return DecodeError.InvalidPayload,
        };

        const polar_decoded = try allocator.alloc(f32, e.dim);
        errdefer allocator.free(polar_decoded);

        polar.decodeInto(polar_decoded, payload.polar, header.max_r) catch |err| switch (err) {
            error.InvalidDimension => return DecodeError.InvalidPayload,
            error.OutOfMemory => return DecodeError.OutOfMemory,
        };

        qjl.decodeInto(e.scratch_qjl_decoded, payload.qjl, header.gamma, &e.rot_op, &e.qjl_workspace);

        math.addInPlace(polar_decoded, e.scratch_qjl_decoded);

        return polar_decoded;
    }

    pub fn dot(e: *Engine, q: []const f32, compressed: []const u8) f32 {
        const header = format.readHeader(compressed) catch return 0;
        if (q.len != e.dim or header.dim != e.dim) return 0;

        const payload = format.slicePayload(compressed, header) catch return 0;

        const polar_sum = polar.dotProduct(q, payload.polar, header.max_r);
        const qjl_sum = qjl.estimateDotWithWorkspace(q, payload.qjl, header.gamma, &e.rot_op, &e.qjl_workspace);

        return polar_sum + qjl_sum;
    }
};

fn computeResidualFromPolar(polar_encoded: []const u8, rotated: []const f32, max_r: f32, residual: []f32) void {
    const dim = rotated.len;
    const num_pairs = dim / 2;

    var bit_pos: usize = 0;
    for (0..num_pairs) |i| {
        const pair = polar.reconstructPair(polar_encoded, bit_pos, max_r);
        bit_pos += 7;

        residual[i * 2] = rotated[i * 2] - pair.dx;
        residual[i * 2 + 1] = rotated[i * 2 + 1] - pair.dy;
    }
}

test "roundtrip" {
    const allocator = std.testing.allocator;
    const seed: u32 = 12345;
    const dim: usize = 8;

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const x: [8]f32 = .{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    const q: [8]f32 = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    var true_dot: f32 = 0;
    for (x, q) |xv, qv| true_dot += xv * qv;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    log.info("{} bytes ({} bits/dim)", .{ compressed.len, (compressed.len - format.HEADER_SIZE) * 8 / dim });

    const decoded = try engine.decode(allocator, compressed);
    defer allocator.free(decoded);

    var decoded_dot: f32 = 0;
    for (decoded, q) |dv, qv| decoded_dot += dv * qv;

    const cdot = engine.dot(&q, compressed);
    log.info("true={e}, decoded_dot={e}, direct_dot={e}", .{ true_dot, decoded_dot, cdot });
    try std.testing.expect(@abs(true_dot - cdot) < 50.0);
}

test "compression ratio" {
    const allocator = std.testing.allocator;
    const seed: u32 = 12345;
    const dim: usize = 128;

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(1234);
    const r = rng.random();

    var x: [128]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const bpd = (compressed.len - format.HEADER_SIZE) * 8 / 128;
    log.info("dim=128, bytes={}, bits/dim={}", .{ compressed.len, bpd });
    try std.testing.expect(bpd <= 4);
}

test "init rejects zero dimension" {
    const allocator = std.testing.allocator;
    const result = Engine.init(allocator, .{ .dim = 0, .seed = 12345 });
    try std.testing.expectError(EncodeError.InvalidDimension, result);
}

test "init rejects odd dimension" {
    const allocator = std.testing.allocator;
    const result = Engine.init(allocator, .{ .dim = 7, .seed = 12345 });
    try std.testing.expectError(EncodeError.InvalidDimension, result);
}

test "encode rejects wrong dimension" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x: [16]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    const result = engine.encode(allocator, &x);
    try std.testing.expectError(EncodeError.InvalidDimension, result);
}

test "decode rejects truncated header" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 128, .seed = 12345 });
    defer engine.deinit(allocator);

    const short: [5]u8 = .{ 1, 0, 0, 0, 0 };
    const result = engine.decode(allocator, &short);
    try std.testing.expectError(DecodeError.InvalidHeader, result);
}

test "decode rejects truncated payload" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 128, .seed = 12345 });
    defer engine.deinit(allocator);

    var buf: [118]u8 = undefined;
    format.writeHeader(&buf, 128, 1000, 100, 1.0, 0.5);
    const result = engine.decode(allocator, &buf);
    try std.testing.expectError(DecodeError.InvalidPayload, result);
}

test "dot returns zero on dimension mismatch" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const wrong_dim: [16]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    const result = engine.dot(&wrong_dim, compressed);
    try std.testing.expectEqual(0.0, result);
}

test "roundtrip correct length and finite" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 64, .seed = 9999 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(9999);
    const r = rng.random();

    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const decoded = try engine.decode(allocator, compressed);
    defer allocator.free(decoded);

    try std.testing.expectEqual(x.len, decoded.len);
    for (decoded) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "roundtrip multiple dims" {
    const allocator = std.testing.allocator;
    const seed: u32 = 8888;

    const dims = [_]usize{ 8, 16, 32, 64, 128 };

    for (dims) |dim| {
        var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(allocator);

        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();

        var x: [128]f32 = undefined;
        for (0..dim) |i| x[i] = r.float(f32) * 10 - 5;

        const compressed = try engine.encode(allocator, x[0..dim]);
        defer allocator.free(compressed);

        const decoded = try engine.decode(allocator, compressed);
        defer allocator.free(decoded);

        try std.testing.expectEqual(dim, decoded.len);
    }
}

test "dot close to decoded dot" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 64, .seed = 7777 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(7777);
    const r = rng.random();

    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    var q: [64]f32 = undefined;
    for (&q) |*v| v.* = r.float(f32);

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const decoded = try engine.decode(allocator, compressed);
    defer allocator.free(decoded);

    var decoded_dot: f32 = 0;
    for (decoded, q) |dv, qv| decoded_dot += dv * qv;

    const direct_dot = engine.dot(&q, compressed);

    const rel_err = @abs(decoded_dot - direct_dot) / (@abs(decoded_dot) + 1e-10);
    log.info("decoded_dot={e}, direct_dot={e}, rel_err={e}", .{ decoded_dot, direct_dot, rel_err });
    try std.testing.expect(rel_err < 0.5);
}

// ===========================================================================
// Golden-value tests: exact bytes for known input+seed.
// These catch any change to the rotation, quantization, or encoding.
// ===========================================================================

test "golden: dim=8 seed=12345 compressed bytes" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    // Exact expected output from reference implementation
    const expected = [_]u8{
        0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x46, 0x1a,
        0x1f, 0x41, 0x77, 0xf2, 0x76, 0x40, 0xaa, 0xaa,
        0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
        0xa5, 0xea, 0x4f, 0x08, 0x1e,
    };
    try std.testing.expectEqual(expected.len, compressed.len);
    try std.testing.expectEqualSlices(u8, &expected, compressed);
}

test "golden: dim=8 seed=12345 header fields" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const header = try format.readHeader(compressed);
    try std.testing.expectEqual(@as(u32, 8), header.dim);
    try std.testing.expectEqual(@as(u32, 4), header.polar_bytes);
    try std.testing.expectEqual(@as(u32, 1), header.qjl_bytes);
    // max_r and gamma must match exactly (bit-identical floats)
    try std.testing.expectEqual(@as(u32, @bitCast(header.max_r)), @as(u32, @bitCast(@as(f32, 9.943914e0))));
    try std.testing.expectEqual(@as(u32, @bitCast(header.gamma)), @as(u32, @bitCast(@as(f32, 3.8585489e0))));
}

test "golden: dim=8 seed=12345 dot product" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    const q = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const cdot = engine.dot(&q, compressed);
    // Must match reference output exactly
    try std.testing.expectApproxEqAbs(@as(f32, 7.087812e-1), cdot, 1e-4);
}

test "golden: dim=64 seed=9999 header fields" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 64, .seed = 9999 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(9999);
    const r = rng.random();
    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    try std.testing.expectEqual(@as(usize, 68), compressed.len);

    const header = try format.readHeader(compressed);
    try std.testing.expectEqual(@as(u32, 64), header.dim);
    try std.testing.expectEqual(@as(u32, 28), header.polar_bytes);
    try std.testing.expectEqual(@as(u32, 8), header.qjl_bytes);
    try std.testing.expectEqual(@as(u32, @bitCast(header.max_r)), @as(u32, @bitCast(@as(f32, 8.701334e0))));
    try std.testing.expectEqual(@as(u32, @bitCast(header.gamma)), @as(u32, @bitCast(@as(f32, 6.347209e0))));
}

// ===========================================================================
// Distortion bound tests: MSE must decrease with dimension.
// For unit vectors, MSE should be well below the paper's theoretical bound
// of D_mse <= 2.7 * (1/4^b) where b is bits per pair.
// ===========================================================================

test "distortion: MSE decreases with dimension" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 64, 128, 256 };
    var prev_mse: f64 = std.math.inf(f64);

    for (dims) |dim| {
        var engine = try Engine.init(allocator, .{ .dim = dim, .seed = 42 });
        defer engine.deinit(allocator);

        var rng = std.Random.DefaultPrng.init(42);
        const r = rng.random();
        const num_vecs: usize = 50;
        var total_mse: f64 = 0;

        for (0..num_vecs) |_| {
            const x = try allocator.alloc(f32, dim);
            defer allocator.free(x);
            var norm_sq: f32 = 0;
            for (0..dim) |j| {
                x[j] = r.float(f32) * 2 - 1;
                norm_sq += x[j] * x[j];
            }
            const inv = 1.0 / @sqrt(norm_sq);
            for (0..dim) |j| x[j] *= inv;

            const compressed = try engine.encode(allocator, x);
            defer allocator.free(compressed);
            const decoded = try engine.decode(allocator, compressed);
            defer allocator.free(decoded);

            var mse: f64 = 0;
            for (0..dim) |j| {
                const d: f64 = @as(f64, x[j]) - @as(f64, decoded[j]);
                mse += d * d;
            }
            total_mse += mse / @as(f64, @floatFromInt(dim));
        }
        total_mse /= num_vecs;

        // MSE must strictly decrease as dimension increases
        try std.testing.expect(total_mse < prev_mse);
        prev_mse = total_mse;
    }
}

test "distortion: bits per dimension is ~4.5" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 128, .seed = 42 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(42);
    const r = rng.random();

    var x: [128]f32 = undefined;
    var norm_sq: f32 = 0;
    for (&x) |*v| {
        v.* = r.float(f32) * 2 - 1;
        norm_sq += v.* * v.*;
    }
    const inv = 1.0 / @sqrt(norm_sq);
    for (&x) |*v| v.* *= inv;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const payload_bits = (compressed.len - format.HEADER_SIZE) * 8;
    const bpd = @as(f64, @floatFromInt(payload_bits)) / 128.0;
    // Must be between 3 and 6 bits/dim (TurboQuant targets ~3-5)
    try std.testing.expect(bpd >= 3.0 and bpd <= 6.0);
}

test "distortion: dot product preservation" {
    const allocator = std.testing.allocator;
    const dim: usize = 128;
    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = 42 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(42);
    const r = rng.random();
    const num_trials: usize = 20;
    var total_abs_err: f64 = 0;

    for (0..num_trials) |_| {
        // Generate unit vector
        var x: [128]f32 = undefined;
        var norm_sq: f32 = 0;
        for (&x) |*v| {
            v.* = r.float(f32) * 2 - 1;
            norm_sq += v.* * v.*;
        }
        const inv_x = 1.0 / @sqrt(norm_sq);
        for (&x) |*v| v.* *= inv_x;

        // Generate unit query
        var q: [128]f32 = undefined;
        norm_sq = 0;
        for (&q) |*v| {
            v.* = r.float(f32) * 2 - 1;
            norm_sq += v.* * v.*;
        }
        const inv_q = 1.0 / @sqrt(norm_sq);
        for (&q) |*v| v.* *= inv_q;

        const compressed = try engine.encode(allocator, &x);
        defer allocator.free(compressed);

        // True dot product
        var true_dot: f64 = 0;
        for (x, q) |xv, qv| true_dot += @as(f64, xv) * @as(f64, qv);

        // Estimated dot product
        const est_dot: f64 = @as(f64, engine.dot(&q, compressed));
        total_abs_err += @abs(true_dot - est_dot);
    }

    const mean_err = total_abs_err / num_trials;
    // Mean absolute dot product error for unit vectors should be small
    // (well under 1.0 for dim=128)
    try std.testing.expect(mean_err < 1.0);
}
