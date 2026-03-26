const std = @import("std");

const RNG_MULTIPLIER: u64 = 1103515245;
const RNG_INCREMENT: u64 = 12345;

const LANE_COUNT = std.simd.suggestVectorLength(f32) orelse 4;

pub fn nextRng(seed: *u64) u64 {
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

pub fn gaussianCoeff(seed: u64, row: usize, col: usize) f32 {
    var rng_state = seed +% @as(u64, row * 31 + col);
    return randGaussian(&rng_state);
}

pub fn matVecMul(
    input: []const f32,
    output: []f32,
    seed: u32,
) void {
    const d = input.len;
    std.debug.assert(input.len == output.len);

    for (0..d) |i| {
        var sum: f32 = 0;
        for (0..d) |j| {
            sum += gaussianCoeff(seed, i, j) * input[j];
        }
        output[i] = sum;
    }
}

pub fn matVecMulTransposed(
    input: []const f32,
    output: []f32,
    seed: u32,
) void {
    const d = input.len;
    std.debug.assert(input.len == output.len);

    for (0..d) |i| {
        var sum: f32 = 0;
        for (0..d) |j| {
            sum += gaussianCoeff(seed, j, i) * input[j];
        }
        output[i] = sum;
    }
}

pub fn projectSign(
    input: []const f32,
    seed: u32,
    output: []u8,
) void {
    const d = input.len;
    const bytes_needed = (d + 7) / 8;
    std.debug.assert(output.len >= bytes_needed);

    const projected = std.heap.page_allocator.alloc(f32, d) catch unreachable;
    defer std.heap.page_allocator.free(projected);
    matVecMul(input, projected, seed);

    @memset(output[0..bytes_needed], 0);
    for (0..d) |i| {
        if (projected[i] > 0) {
            output[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        }
    }
}

pub fn signToVector(sign_bits: []const u8, dim: usize, output: []f32) void {
    for (0..dim) |i| {
        const bit = (sign_bits[i / 8] >> @intCast(i % 8)) & 1;
        output[i] = if (bit == 1) 1.0 else -1.0;
    }
}

pub fn rotate(
    allocator: std.mem.Allocator,
    input: []const f32,
    seed: u32,
) std.mem.Allocator.Error![]f32 {
    const d = input.len;
    const result = try allocator.alloc(f32, d);
    errdefer allocator.free(result);
    matVecMul(input, result, seed);
    return result;
}

inline fn rowDot(matrix_row: []const f32, input: []const f32, d: usize) f32 {
    var sum_vec: @Vector(LANE_COUNT, f32) = @splat(0);

    var col: usize = 0;
    while (col + LANE_COUNT <= d) : (col += LANE_COUNT) {
        const m: @Vector(LANE_COUNT, f32) = matrix_row[col..][0..LANE_COUNT].*;
        const i: @Vector(LANE_COUNT, f32) = input[col..][0..LANE_COUNT].*;
        sum_vec += m * i;
    }

    var total = @reduce(.Add, sum_vec);

    while (col < d) : (col += 1) {
        total += matrix_row[col] * input[col];
    }
    return total;
}

/// Modified Gram-Schmidt orthogonalization on columns of a row-major f32 matrix.
/// Uses f64 intermediate precision for numerical stability, then writes back to f32.
fn orthogonalize(matrix: []f32, dim: usize) void {
    // Work in f64 for precision
    const cols = std.heap.page_allocator.alloc(f64, dim * dim) catch return;
    defer std.heap.page_allocator.free(cols);

    // Copy to f64 column-major for easier column access
    for (0..dim) |col| {
        for (0..dim) |row| {
            cols[col * dim + row] = @as(f64, matrix[row * dim + col]);
        }
    }

    // Modified Gram-Schmidt in f64
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

    // Write back to f32 row-major
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

    pub fn prepare(allocator: std.mem.Allocator, dim: usize, seed: u32) !RotationOperator {
        const matrix = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix);

        const matrix_t = try allocator.alloc(f32, dim * dim);
        errdefer allocator.free(matrix_t);

        // Step 1: Generate random Gaussian matrix
        for (0..dim) |i| {
            for (0..dim) |j| {
                matrix[i * dim + j] = gaussianCoeff(seed, i, j);
            }
        }

        // Step 2: Orthogonalize via modified Gram-Schmidt (QR decomposition)
        // This produces a Haar-distributed random orthogonal matrix
        orthogonalize(matrix, dim);

        // Step 3: Compute transpose (R^T = R^{-1} for orthogonal matrices)
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

test "matVecMul consistent with definition" {
    const input = [_]f32{ 1.0, 0.0 };
    var output: [2]f32 = undefined;
    matVecMul(&input, &output, 999);

    const s00 = gaussianCoeff(999, 0, 0);
    const s01 = gaussianCoeff(999, 0, 1);
    const expected0 = s00 * 1.0 + s01 * 0.0;

    try std.testing.expectEqual(expected0, output[0]);
}

test "matVecMulTransposed consistent" {
    const input = [_]f32{ 1.0, 0.0 };
    var output: [2]f32 = undefined;
    matVecMulTransposed(&input, &output, 999);

    const s00 = gaussianCoeff(999, 0, 0);
    const s10 = gaussianCoeff(999, 1, 0);
    const expected0 = s00 * 1.0 + s10 * 0.0;

    try std.testing.expectEqual(expected0, output[0]);
}

test "rotate deterministic same seed" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const result1 = try rotate(allocator, &input, 12345);
    defer allocator.free(result1);

    const result2 = try rotate(allocator, &input, 12345);
    defer allocator.free(result2);

    try std.testing.expectEqualSlices(f32, result1, result2);
}

test "rotate differs across seeds" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const result1 = try rotate(allocator, &input, 11111);
    defer allocator.free(result1);

    const result2 = try rotate(allocator, &input, 22222);
    defer allocator.free(result2);

    const same = for (result1, result2) |a, b| {
        if (a != b) break false;
    } else true;

    try std.testing.expect(!same);
}

test "rotate zero input stays zero" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const result = try rotate(allocator, &input, 99999);
    defer allocator.free(result);

    for (result) |v| {
        try std.testing.expectEqual(0.0, v);
    }
}

test "projectSign produces valid bits" {
    const input = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0 };
    var bits: [1]u8 = undefined;
    projectSign(&input, 55555, &bits);

    try std.testing.expect(bits[0] != 0);
}

test "signToVector roundtrip" {
    const input = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0 };
    var bits: [1]u8 = undefined;
    projectSign(&input, 77777, &bits);

    var recovered: [8]f32 = undefined;
    signToVector(&bits, 8, &recovered);

    for (recovered) |v| {
        try std.testing.expect(v == 1.0 or v == -1.0);
    }
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

    // Verify R^T * R ≈ I by checking columns are orthonormal
    for (0..dim) |i| {
        for (0..dim) |j| {
            var dot: f32 = 0;
            for (0..dim) |row| {
                dot += op.matrix[row * dim + i] * op.matrix[row * dim + j];
            }
            if (i == j) {
                // Diagonal: column norm should be 1.0
                try std.testing.expectApproxEqAbs(@as(f32, 1.0), dot, 1e-4);
            } else {
                // Off-diagonal: columns should be orthogonal
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

    // Test with 10 random vectors
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

        // ||R*x|| should equal ||x|| for orthogonal R
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

    // R^T * R * x ≈ x for orthogonal R
    for (input, recovered) |orig, rec| {
        try std.testing.expectApproxEqAbs(orig, rec, 0.01);
    }
}
