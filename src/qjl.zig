const std = @import("std");
const rotation = @import("rotation.zig");
const math = @import("math.zig");

pub const QjlError = error{ InvalidDimension, OutOfMemory };

const SQRT_PI_OVER_2: f32 = 1.2533141373155003;
const LANE_COUNT = std.simd.suggestVectorLength(f32) orelse 4;

pub const Workspace = struct {
    projected: []f32,
    sign_vec: []f32,
    st_sign: []f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize) QjlError!Workspace {
        return .{
            .projected = try allocator.alloc(f32, dim),
            .sign_vec = try allocator.alloc(f32, dim),
            .st_sign = try allocator.alloc(f32, dim),
        };
    }

    pub fn deinit(w: *Workspace, allocator: std.mem.Allocator) void {
        allocator.free(w.projected);
        allocator.free(w.sign_vec);
        allocator.free(w.st_sign);
        w.* = .{ .projected = &.{}, .sign_vec = &.{}, .st_sign = &.{} };
    }
};

pub fn encodeWithWorkspace(
    allocator: std.mem.Allocator,
    residual: []const f32,
    rot_op: *const rotation.RotationOperator,
    workspace: *Workspace,
) QjlError![]u8 {
    const d = residual.len;
    if (d == 0) return QjlError.InvalidDimension;

    rot_op.matVecMul(residual, workspace.projected);

    const bits_bytes = (d + 7) / 8;
    const result = try allocator.alloc(u8, bits_bytes);
    @memset(result, 0);

    // Vectorized sign-bit packing: compare against zero, pack into bytes
    var i: usize = 0;
    // Process 8 elements at a time (one output byte)
    while (i + 8 <= d) : (i += 8) {
        var byte: u8 = 0;
        inline for (0..8) |bit| {
            if (workspace.projected[i + bit] > 0) {
                byte |= @as(u8, 1) << bit;
            }
        }
        result[i / 8] = byte;
    }
    // Remaining bits
    while (i < d) : (i += 1) {
        if (workspace.projected[i] > 0) {
            result[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        }
    }
    return result;
}

pub fn decodeInto(
    out: []f32,
    qjl_bits: []const u8,
    gamma: f32,
    rot_op: *const rotation.RotationOperator,
    workspace: *Workspace,
) void {
    const dim = out.len;
    if (dim == 0) return;

    rotation.signToVector(qjl_bits, dim, workspace.sign_vec);
    rot_op.matVecMulTransposed(workspace.sign_vec, workspace.st_sign);

    // Vectorized scaling
    const scale_val = SQRT_PI_OVER_2 / @as(f32, @floatFromInt(dim)) * gamma;
    const scale_vec: @Vector(LANE_COUNT, f32) = @splat(scale_val);

    var i: usize = 0;
    while (i + LANE_COUNT <= dim) : (i += LANE_COUNT) {
        const v: @Vector(LANE_COUNT, f32) = workspace.st_sign[i..][0..LANE_COUNT].*;
        @as(*[LANE_COUNT]f32, @ptrCast(out[i..])).* = v * scale_vec;
    }
    while (i < dim) : (i += 1) {
        out[i] = workspace.st_sign[i] * scale_val;
    }
}

pub fn estimateDotWithWorkspace(
    q: []const f32,
    qjl_bits: []const u8,
    gamma: f32,
    rot_op: *const rotation.RotationOperator,
    workspace: *Workspace,
) f32 {
    const d = q.len;
    if (d == 0) return 0;

    rot_op.matVecMul(q, workspace.projected);

    // Vectorized sign extraction
    rotation.signToVector(qjl_bits, d, workspace.sign_vec);

    const dot_sum = math.dot(workspace.projected, workspace.sign_vec);

    const scale_val = SQRT_PI_OVER_2 / @as(f32, @floatFromInt(d)) * gamma;
    return dot_sum * scale_val;
}

test "encodeWithWorkspace rejects zero dimension" {
    const allocator = std.testing.allocator;
    const data: [0]f32 = .{};

    var ws = try Workspace.init(allocator, 0);
    defer ws.deinit(allocator);

    var rot_op = try rotation.RotationOperator.prepare(allocator, 0, 12345);
    defer rot_op.destroy(allocator);

    const result = encodeWithWorkspace(allocator, &data, &rot_op, &ws);
    try std.testing.expectError(QjlError.InvalidDimension, result);
}

test "Workspace init and deinit" {
    const allocator = std.testing.allocator;
    var ws = try Workspace.init(allocator, 128);
    defer ws.deinit(allocator);

    try std.testing.expectEqual(128, ws.projected.len);
    try std.testing.expectEqual(128, ws.sign_vec.len);
    try std.testing.expectEqual(128, ws.st_sign.len);
}

test "encode/decode roundtrip with RotationOperator" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const seed: u32 = 12345;

    var rot_op = try rotation.RotationOperator.prepare(allocator, dim, seed);
    defer rot_op.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const residual = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0, -16.0 };
    var gamma: f32 = 0;
    for (residual) |v| gamma += v * v;
    gamma = @sqrt(gamma);

    const encoded = try encodeWithWorkspace(allocator, &residual, &rot_op, &ws);
    defer allocator.free(encoded);

    var decoded: [dim]f32 = undefined;
    decodeInto(&decoded, encoded, gamma, &rot_op, &ws);

    for (decoded) |d| {
        try std.testing.expect(std.math.isFinite(d));
    }
}

test "estimateDotWithWorkspace deterministic" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const seed: u32 = 54321;

    var rot_op = try rotation.RotationOperator.prepare(allocator, dim, seed);
    defer rot_op.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const q = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const residual = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    const gamma: f32 = 1.5;

    const encoded = try encodeWithWorkspace(allocator, &residual, &rot_op, &ws);
    defer allocator.free(encoded);

    const est1 = estimateDotWithWorkspace(&q, encoded, gamma, &rot_op, &ws);
    const est2 = estimateDotWithWorkspace(&q, encoded, gamma, &rot_op, &ws);

    try std.testing.expectEqual(est1, est2);
}

test "zero gamma gives zero decoded residual" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const seed: u32 = 12345;

    var rot_op = try rotation.RotationOperator.prepare(allocator, dim, seed);
    defer rot_op.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const bits = [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF };

    var decoded: [dim]f32 = undefined;
    decodeInto(&decoded, &bits, 0.0, &rot_op, &ws);

    for (decoded) |v| {
        try std.testing.expectEqual(0.0, v);
    }
}

test "zero gamma gives zero estimated dot" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    const seed: u32 = 12345;

    var rot_op = try rotation.RotationOperator.prepare(allocator, dim, seed);
    defer rot_op.destroy(allocator);

    var ws = try Workspace.init(allocator, dim);
    defer ws.deinit(allocator);

    const q = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const bits = [_]u8{ 0xAA, 0x55, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44 };

    const est = estimateDotWithWorkspace(&q, &bits, 0.0, &rot_op, &ws);
    try std.testing.expectEqual(0.0, est);
}
