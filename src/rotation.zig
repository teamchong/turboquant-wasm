const std = @import("std");

const RNG_MULTIPLIER: u64 = 1103515245;
const RNG_INCREMENT: u64 = 12345;

const LANE_COUNT = std.simd.suggestVectorLength(f32) orelse 4;

const RotationError = error{OutOfMemory};

fn nextRng(seed: *u64) u64 {
    seed.* = seed.* *% RNG_MULTIPLIER +% RNG_INCREMENT;
    return seed.*;
}

fn randF32(seed: *u64) f32 {
    const val = @as(f32, @floatFromInt(nextRng(seed) % (1 << 31))) / @as(f32, @floatFromInt(1 << 31));
    return if (val == 0) 0.00001 else val;
}

fn randGaussian(seed: *u64) f32 {
    const v1 = randF32(seed);
    const v2 = randF32(seed);
    return @sqrt(-2.0 * @log(v1)) * @cos(2.0 * std.math.pi * v2);
}

fn gaussianCoeff(seed: u64, row: usize, col: usize) f32 {
    var rng_state = seed +% @as(u64, row * 31 + col);
    return randGaussian(&rng_state);
}

pub fn signToVector(sign_bits: []const u8, dim: usize, output: []f32) void {
    // Process 8 bits (1 byte) at a time — unpack to 8 floats
    var i: usize = 0;
    while (i + 8 <= dim) : (i += 8) {
        const byte = sign_bits[i / 8];
        inline for (0..8) |b| {
            output[i + b] = if ((byte >> b) & 1 == 1) 1.0 else -1.0;
        }
    }
    // Remainder
    while (i < dim) : (i += 1) {
        const bit = (sign_bits[i / 8] >> @intCast(i % 8)) & 1;
        output[i] = if (bit == 1) 1.0 else -1.0;
    }
}

inline fn rowDot(matrix_row: []const f32, input: []const f32, d: usize) f32 {
    var sum_vec: @Vector(LANE_COUNT, f32) = @splat(0);

    var col: usize = 0;
    while (col + LANE_COUNT <= d) : (col += LANE_COUNT) {
        const m: @Vector(LANE_COUNT, f32) = matrix_row[col..][0..LANE_COUNT].*;
        const v: @Vector(LANE_COUNT, f32) = input[col..][0..LANE_COUNT].*;
        sum_vec = @mulAdd(@Vector(LANE_COUNT, f32), m, v, sum_vec);
    }

    var total = @reduce(.Add, sum_vec);

    while (col < d) : (col += 1) {
        total = @mulAdd(f32, matrix_row[col], input[col], total);
    }
    return total;
}

fn orthogonalize(allocator: std.mem.Allocator, matrix: []f32, dim: usize) RotationError!void {
    const cols = try allocator.alloc(f64, dim * dim);
    defer allocator.free(cols);

    for (0..dim) |col| {
        for (0..dim) |row| {
            cols[col * dim + row] = @as(f64, matrix[row * dim + col]);
        }
    }

    for (0..dim) |i| {
        for (0..i) |j| {
            var dot: f64 = 0;
            for (0..dim) |row| {
                dot += cols[i * dim + row] * cols[j * dim + row];
            }
            for (0..dim) |row| {
                cols[i * dim + row] -= dot * cols[j * dim + row];
            }
        }
        var col_norm: f64 = 0;
        for (0..dim) |row| {
            const v = cols[i * dim + row];
            col_norm += v * v;
        }
        col_norm = @sqrt(col_norm);
        if (col_norm > 0) {
            const inv = 1.0 / col_norm;
            for (0..dim) |row| {
                cols[i * dim + row] *= inv;
            }
        }
    }

    for (0..dim) |col| {
        for (0..dim) |row| {
            matrix[row * dim + col] = @as(f32, @floatCast(cols[col * dim + row]));
        }
    }
}

pub const RotationOperator = struct {
    dim: usize,
    seed: u32,
    matrix: []f32,
    matrix_t: []f32,

    pub fn prepare(allocator: std.mem.Allocator, dim: usize, seed: u32) RotationError!RotationOperator {
        const matrix = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix);

        const matrix_t = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix_t);

        for (0..dim) |i| {
            for (0..dim) |j| {
                matrix[i * dim + j] = gaussianCoeff(seed, i, j);
            }
        }

        try orthogonalize(allocator, matrix, dim);

        for (0..dim) |i| {
            for (0..dim) |j| {
                matrix_t[j * dim + i] = matrix[i * dim + j];
            }
        }

        return .{
            .dim = dim,
            .seed = seed,
            .matrix = matrix,
            .matrix_t = matrix_t,
        };
    }

    pub fn destroy(op: *RotationOperator, allocator: std.mem.Allocator) void {
        allocator.free(op.matrix);
        allocator.free(op.matrix_t);
    }

    pub fn matVecMul(op: *const RotationOperator, input: []const f32, output: []f32) void {
        const d = op.dim;
        std.debug.assert(input.len == d and output.len == d);

        for (0..d) |i| {
            const row_start = i * d;
            output[i] = rowDot(op.matrix[row_start..], input, d);
        }
    }

    pub fn matVecMulTransposed(op: *const RotationOperator, input: []const f32, output: []f32) void {
        const d = op.dim;
        std.debug.assert(input.len == d and output.len == d);

        for (0..d) |i| {
            const row_start = i * d;
            output[i] = rowDot(op.matrix_t[row_start..], input, d);
        }
    }

    pub fn rotate(op: *const RotationOperator, input: []const f32, output: []f32) void {
        op.matVecMul(input, output);
    }
};

test "gaussianCoeff deterministic" {
    const c1 = gaussianCoeff(12345, 0, 0);
    const c2 = gaussianCoeff(12345, 0, 0);
    try std.testing.expectEqual(c1, c2);
}

test "gaussianCoeff differs by position" {
    const c1 = gaussianCoeff(12345, 0, 0);
    const c2 = gaussianCoeff(12345, 0, 1);
    try std.testing.expect(c1 != c2);
}

test "signToVector roundtrip" {
    const bits = [_]u8{0b10110101};
    var output: [8]f32 = undefined;
    signToVector(&bits, 8, &output);

    try std.testing.expect(output[0] == 1.0);
    try std.testing.expect(output[1] == -1.0);
    try std.testing.expect(output[2] == 1.0);
    try std.testing.expect(output[3] == -1.0);
    try std.testing.expect(output[4] == 1.0);
    try std.testing.expect(output[5] == 1.0);
    try std.testing.expect(output[6] == -1.0);
    try std.testing.expect(output[7] == 1.0);
}

test "RotationOperator.prepare produces finite values" {
    const allocator = std.testing.allocator;
    const dim: usize = 4;
    const seed: u32 = 12345;

    var op = try RotationOperator.prepare(allocator, dim, seed);
    defer op.destroy(allocator);

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    op.matVecMul(&input, &output);

    for (output) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "RotationOperator orthogonality" {
    const allocator = std.testing.allocator;
    const dim: usize = 32;
    const seed: u32 = 42;

    var op = try RotationOperator.prepare(allocator, dim, seed);
    defer op.destroy(allocator);

    for (0..dim) |i| {
        for (0..dim) |j| {
            var dot: f32 = 0;
            for (0..dim) |row| {
                dot += op.matrix[row * dim + i] * op.matrix[row * dim + j];
            }
            if (i == j) {
                try std.testing.expectApproxEqAbs(@as(f32, 1.0), dot, 1e-4);
            } else {
                try std.testing.expectApproxEqAbs(@as(f32, 0.0), dot, 1e-4);
            }
        }
    }
}

test "rotation preserves norm" {
    const allocator = std.testing.allocator;
    const dim: usize = 64;
    const seed: u32 = 42;

    var op = try RotationOperator.prepare(allocator, dim, seed);
    defer op.destroy(allocator);

    var rng = std.Random.DefaultPrng.init(12345);
    const r = rng.random();

    for (0..10) |_| {
        var input: [64]f32 = undefined;
        var output: [64]f32 = undefined;

        var input_norm_sq: f32 = 0;
        for (&input) |*v| {
            v.* = r.float(f32) * 2 - 1;
            input_norm_sq += v.* * v.*;
        }
        const input_norm = @sqrt(input_norm_sq);

        op.matVecMul(&input, &output);

        var output_norm_sq: f32 = 0;
        for (output) |v| {
            output_norm_sq += v * v;
        }
        const output_norm = @sqrt(output_norm_sq);

        try std.testing.expectApproxEqAbs(input_norm, output_norm, input_norm * 1e-3);
    }
}

test "RotationOperator transpose is inverse" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const seed: u32 = 42;

    var op = try RotationOperator.prepare(allocator, dim, seed);
    defer op.destroy(allocator);

    const input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var rotated: [16]f32 = undefined;
    var recovered: [16]f32 = undefined;

    op.matVecMul(&input, &rotated);
    op.matVecMulTransposed(&rotated, &recovered);

    for (input, recovered) |orig, rec| {
        try std.testing.expectApproxEqAbs(orig, rec, 0.01);
    }
}

test "RotationOperator deterministic same seed" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var op1 = try RotationOperator.prepare(allocator, 4, 12345);
    defer op1.destroy(allocator);

    var op2 = try RotationOperator.prepare(allocator, 4, 12345);
    defer op2.destroy(allocator);

    var output1: [4]f32 = undefined;
    var output2: [4]f32 = undefined;

    op1.matVecMul(&input, &output1);
    op2.matVecMul(&input, &output2);

    try std.testing.expectEqualSlices(f32, &output1, &output2);
}

test "RotationOperator differs across seeds" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var op1 = try RotationOperator.prepare(allocator, 4, 11111);
    defer op1.destroy(allocator);

    var op2 = try RotationOperator.prepare(allocator, 4, 22222);
    defer op2.destroy(allocator);

    var output1: [4]f32 = undefined;
    var output2: [4]f32 = undefined;

    op1.matVecMul(&input, &output1);
    op2.matVecMul(&input, &output2);

    const same = for (output1, output2) |a, b| {
        if (a != b) break false;
    } else true;

    try std.testing.expect(!same);
}

test "RotationOperator zero input stays zero" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    var op = try RotationOperator.prepare(allocator, 4, 99999);
    defer op.destroy(allocator);

    var output: [4]f32 = undefined;
    op.matVecMul(&input, &output);

    for (output) |v| {
        try std.testing.expectEqual(0.0, v);
    }
}
