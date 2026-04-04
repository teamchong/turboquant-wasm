const std = @import("std");

pub const MatrixError = error{ InvalidDimension, OutOfMemory };

pub const RNG_MULTIPLIER: u64 = 1103515245;
pub const RNG_INCREMENT: u64 = 12345;

pub fn gaussianRandom(seed: u64) f32 {
    var s = seed;
    const u1_val = @as(f32, @floatFromInt(@mod(s * RNG_MULTIPLIER + RNG_INCREMENT, 1 << 31))) / @as(f32, @floatFromInt(1 << 31));
    s = s * RNG_MULTIPLIER + RNG_INCREMENT;
    const u2_val = @as(f32, @floatFromInt(@mod(s * RNG_MULTIPLIER + RNG_INCREMENT, 1 << 31))) / @as(f32, @floatFromInt(1 << 31));
    const u1_clamped = if (u1_val == 0) 0.00001 else u1_val;
    return @sqrt(-2.0 * @log(u1_clamped)) * @cos(2.0 * std.math.pi * u2_val);
}

pub fn generateGaussianMatrix(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u32) MatrixError![]f32 {
    const total = rows * cols;
    const matrix = try allocator.alloc(f32, total);
    errdefer allocator.free(matrix);

    var rng_seed: u64 = seed;
    for (0..total) |i| {
        rng_seed = rng_seed *% RNG_MULTIPLIER +% RNG_INCREMENT;
        const rand1 = @as(f32, @floatFromInt(@mod(rng_seed, 1 << 31))) / @as(f32, @floatFromInt(1 << 31));
        rng_seed = rng_seed *% RNG_MULTIPLIER +% RNG_INCREMENT;
        const rand2 = @as(f32, @floatFromInt(@mod(rng_seed, 1 << 31))) / @as(f32, @floatFromInt(1 << 31));
        const clamped = if (rand1 == 0) 0.00001 else rand1;
        matrix[i] = @sqrt(-2.0 * @log(clamped)) * @cos(2.0 * std.math.pi * rand2);
    }

    return matrix;
}

pub fn matVecMul(matrix: []const f32, rows: usize, cols: usize, v: []const f32, result: []f32) void {
    for (0..rows) |i| {
        var sum: f32 = 0;
        for (0..cols) |j| {
            sum += matrix[i * cols + j] * v[j];
        }
        result[i] = sum;
    }
}

pub fn signBits(v: []const f32, result: []u8) void {
    for (v, 0..) |val, i| {
        result[i / 8] = if (val > 0)
            result[i / 8] | (@as(u8, 1) << @intCast(i % 8))
        else
            result[i / 8] & ~(@as(u8, 1) << @intCast(i % 8));
    }
}

pub fn extractSign(v: []f32, data: []const u8, start_bit: usize) void {
    for (0..v.len) |i| {
        const bit = (data[start_bit / 8 + i / 8] >> @intCast(i % 8)) & 1;
        v[i] = if (bit == 1) 1.0 else -1.0;
    }
}

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |av, bv| {
        sum += av * bv;
    }
    return sum;
}

pub fn vectorNorm(x: []const f32) f32 {
    var sum: f32 = 0;
    for (x) |v| {
        sum += v * v;
    }
    return @sqrt(sum);
}

pub fn vectorSubtract(a: []const f32, b: []const f32, result: []f32) void {
    for (0..a.len) |i| {
        result[i] = a[i] - b[i];
    }
}

pub fn vectorScale(v: []f32, scalar: f32) void {
    for (v) |*val| {
        val.* *= scalar;
    }
}

pub fn hadamardTransform(x: []f32) void {
    const d = x.len;
    var stride: usize = 1;
    while (stride < d) : (stride *= 2) {
        for (0..d / (stride * 2)) |block| {
            const base = block * stride * 2;
            for (0..stride) |j| {
                const u = x[base + j];
                const v = x[base + j + stride];
                x[base + j] = u + v;
                x[base + j + stride] = u - v;
            }
        }
    }
}

pub fn hadamardMatrixMultiply(allocator: std.mem.Allocator, x: []const f32, y: []f32, seed: u32) MatrixError!void {
    const d = x.len;
    const temp = try allocator.alloc(f32, d);
    defer allocator.free(temp);

    @memcpy(temp, x);
    hadamardTransform(temp);

    var rng_seed: u64 = seed;
    for (0..d) |i| {
        rng_seed = rng_seed *% RNG_MULTIPLIER +% RNG_INCREMENT;
        const sign: f32 = if (rng_seed % 2 == 0) 1.0 else -1.0;
        rng_seed = rng_seed *% RNG_MULTIPLIER +% RNG_INCREMENT;
        const scale = @as(f32, @floatFromInt(@mod(rng_seed, 10000))) / 10000.0;
        y[i] = temp[i] * sign * scale;
    }

    const scale_factor = @sqrt(1.0 / @as(f32, @floatFromInt(d)));
    vectorScale(y, scale_factor);
}
