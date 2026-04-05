const std = @import("std");

pub const PolarError = error{ InvalidDimension, OutOfMemory };

const R_BITS: u5 = 4;
const THETA_BITS: u5 = 3;
const BITS_PER_PAIR: u5 = R_BITS + THETA_BITS;
const R_LEVELS: f32 = 15.0;
const THETA_LEVELS: f32 = 7.0;

const ANGLE_BUCKETS: usize = 8;
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 2.0 * PI;

const polar_sin_table: [ANGLE_BUCKETS]f32 = init: {
    var table: [ANGLE_BUCKETS]f32 = undefined;
    for (0..ANGLE_BUCKETS) |i| {
        const theta = @as(f32, @floatFromInt(i)) / THETA_LEVELS * TWO_PI - PI;
        table[i] = @sin(theta);
    }
    break :init table;
};

const polar_cos_table: [ANGLE_BUCKETS]f32 = init: {
    var table: [ANGLE_BUCKETS]f32 = undefined;
    for (0..ANGLE_BUCKETS) |i| {
        const theta = @as(f32, @floatFromInt(i)) / THETA_LEVELS * TWO_PI - PI;
        table[i] = @cos(theta);
    }
    break :init table;
};

const direction_vectors: [ANGLE_BUCKETS][2]f32 = init: {
    var dirs: [ANGLE_BUCKETS][2]f32 = undefined;
    for (0..ANGLE_BUCKETS) |i| {
        const theta = @as(f32, @floatFromInt(i)) / THETA_LEVELS * TWO_PI - PI;
        dirs[i] = .{ @cos(theta), @sin(theta) };
    }
    break :init dirs;
};

// Precomputed (dx, dy) for all 128 possible 7-bit values (16 radii × 8 angles).
// Indexed by raw combined byte: pair_lut[combined] = { r/15 * cos(theta), r/15 * sin(theta) }
// Caller multiplies by max_r. One indexed load replaces unpack + 2 table lookups + multiply.
const pair_lut: [128][2]f32 = init: {
    var lut: [128][2]f32 = undefined;
    for (0..128) |c| {
        const r_norm = @as(f32, @floatFromInt((c >> 3) & 0xF)) / 15.0;
        const bucket = c & 0x7;
        const theta = @as(f32, @floatFromInt(bucket)) / THETA_LEVELS * TWO_PI - PI;
        lut[c] = .{ r_norm * @cos(theta), r_norm * @sin(theta) };
    }
    break :init lut;
};

pub fn cosTable() *const [8]f32 {
    return &polar_cos_table;
}

pub fn sinTable() *const [8]f32 {
    return &polar_sin_table;
}

pub inline fn reconstructPair(compressed: []const u8, bit_pos: usize, max_r: f32) struct { dx: f32, dy: f32 } {
    const byte_idx = bit_pos / 8;
    const bit_off: u4 = @intCast(bit_pos % 8);
    const window = @as(u16, compressed[byte_idx]) | (@as(u16, compressed[byte_idx + 1]) << 8);
    const combined: u7 = @intCast((window >> bit_off) & 0x7F);
    const entry = pair_lut[combined];
    return .{ .dx = entry[0] * max_r, .dy = entry[1] * max_r };
}

pub fn encode(
    allocator: std.mem.Allocator,
    rotated: []const f32,
    max_r: f32,
) PolarError![]u8 {
    const dim = rotated.len;
    if (dim == 0 or dim % 2 != 0) return PolarError.InvalidDimension;

    const num_pairs = dim / 2;
    const polar_bits = num_pairs * BITS_PER_PAIR;
    const polar_bytes = (polar_bits + 7) / 8;

    // +1 byte padding so 16-bit window reads in unpackOne never go out of bounds
    const result = try allocator.alloc(u8, polar_bytes + 1);
    @memset(result, 0);

    var bit_pos: usize = 0;
    for (0..num_pairs) |i| {
        const x = rotated[i * 2];
        const y = rotated[i * 2 + 1];
        const r = @sqrt(x * x + y * y);

        const r_quant = @as(u4, @intFromFloat(r / max_r * R_LEVELS));
        const theta_bucket = findNearestAngleBucket(x, y);
        const combined: u7 = (@as(u7, r_quant) << THETA_BITS) | theta_bucket;

        // Write LSB-first so 16-bit window extraction works without @bitReverse
        for (0..BITS_PER_PAIR) |j| {
            const bit = (combined >> @intCast(j)) & 1;
            if (bit == 1) {
                result[bit_pos / 8] |= @as(u8, 1) << @intCast(bit_pos % 8);
            }
            bit_pos += 1;
        }
    }

    return result;
}

pub fn decodeInto(
    out: []f32,
    compressed: []const u8,
    max_r: f32,
) PolarError!void {
    const dim = out.len;
    if (dim == 0 or dim % 2 != 0) return PolarError.InvalidDimension;

    const num_pairs = dim / 2;
    var bit_pos: usize = 0;

    // Unroll 2 pairs at a time for SIMD f32x4 store
    var i: usize = 0;
    while (i + 2 <= num_pairs) : (i += 2) {
        const p0 = reconstructPair(compressed, bit_pos, max_r);
        bit_pos += BITS_PER_PAIR;
        const p1 = reconstructPair(compressed, bit_pos, max_r);
        bit_pos += BITS_PER_PAIR;

        const vals: @Vector(4, f32) = .{ p0.dx, p0.dy, p1.dx, p1.dy };
        @as(*[4]f32, @ptrCast(out[i * 2 ..].ptr)).* = vals;
    }
    // Handle remaining pair
    if (i < num_pairs) {
        const pair = reconstructPair(compressed, bit_pos, max_r);
        out[i * 2] = pair.dx;
        out[i * 2 + 1] = pair.dy;
    }
}

pub fn dotProduct(
    q: []const f32,
    compressed: []const u8,
    max_r: f32,
) f32 {
    const dim = q.len;
    if (dim == 0 or dim % 2 != 0) return 0;

    const num_pairs = dim / 2;
    var sum: f32 = 0;
    var bit_pos: usize = 0;

    // Unroll 2 pairs at a time for SIMD f32x4 dot
    var i: usize = 0;
    while (i + 2 <= num_pairs) : (i += 2) {
        const p0 = reconstructPair(compressed, bit_pos, max_r);
        bit_pos += BITS_PER_PAIR;
        const p1 = reconstructPair(compressed, bit_pos, max_r);
        bit_pos += BITS_PER_PAIR;

        const q_vec: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(q[i * 2 ..].ptr)).*;
        const p_vec: @Vector(4, f32) = .{ p0.dx, p0.dy, p1.dx, p1.dy };
        sum += @reduce(.Add, q_vec * p_vec);
    }
    // Handle remaining pair
    if (i < num_pairs) {
        const pair = reconstructPair(compressed, bit_pos, max_r);
        sum += q[i * 2] * pair.dx + q[i * 2 + 1] * pair.dy;
    }

    return sum;
}

fn findNearestAngleBucket(x: f32, y: f32) u3 {
    var best_bucket: u3 = 0;
    var best_dot: f32 = -1.0;
    for (0..ANGLE_BUCKETS) |i| {
        const dot = x * direction_vectors[i][0] + y * direction_vectors[i][1];
        if (dot > best_dot) {
            best_dot = dot;
            best_bucket = @intCast(i);
        }
    }
    return best_bucket;
}

test "encode rejects odd dimension" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const result = encode(allocator, &data, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "encode rejects zero dimension" {
    const allocator = std.testing.allocator;
    const data: [0]f32 = .{};
    const result = encode(allocator, &data, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "decodeInto rejects odd dimension" {
    const compressed = [_]u8{0};
    var out: [3]f32 = undefined;
    const result = decodeInto(&out, &compressed, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "decodeInto rejects zero dimension" {
    const compressed = [_]u8{0};
    var out: [0]f32 = undefined;
    const result = decodeInto(&out, &compressed, 1.0);
    try std.testing.expectError(PolarError.InvalidDimension, result);
}

test "encode decode roundtrip" {
    const allocator = std.testing.allocator;
    const rotated = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const max_r: f32 = 5.0;

    const encoded = try encode(allocator, &rotated, max_r);
    defer allocator.free(encoded);

    var decoded: [4]f32 = undefined;
    try decodeInto(&decoded, encoded, max_r);

    try std.testing.expectEqual(rotated.len, decoded.len);
}

test "dotProduct matches decoded dot" {
    const allocator = std.testing.allocator;
    const rotated = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const max_r: f32 = 2.0;

    const encoded = try encode(allocator, &rotated, max_r);
    defer allocator.free(encoded);

    var decoded: [4]f32 = undefined;
    try decodeInto(&decoded, encoded, max_r);

    const decoded_dot = decoded[0] * q[0] + decoded[1] * q[1] + decoded[2] * q[2] + decoded[3] * q[3];
    const polar_dot = dotProduct(&q, encoded, max_r);

    try std.testing.expectEqual(decoded_dot, polar_dot);
}

test "all zero encodes safely" {
    const allocator = std.testing.allocator;
    const rotated = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const encoded = try encode(allocator, &rotated, 1.0);
    defer allocator.free(encoded);

    var decoded: [4]f32 = undefined;
    try decodeInto(&decoded, encoded, 1.0);

    for (decoded) |v| {
        try std.testing.expectEqual(0.0, v);
    }
}
