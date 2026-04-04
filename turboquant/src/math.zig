const std = @import("std");

const LANE_COUNT = std.simd.suggestVectorLength(f32) orelse 4;

inline fn dotVec(a: @Vector(LANE_COUNT, f32), b: @Vector(LANE_COUNT, f32)) f32 {
    return @reduce(.Add, a * b);
}

pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const len = a.len;

    var sum_vec: @Vector(LANE_COUNT, f32) = @splat(0);

    var i: usize = 0;
    while (i + LANE_COUNT <= len) : (i += LANE_COUNT) {
        const av = a[i..][0..LANE_COUNT];
        const bv = b[i..][0..LANE_COUNT];
        sum_vec = @mulAdd(@Vector(LANE_COUNT, f32), @as(@Vector(LANE_COUNT, f32), av[0..LANE_COUNT].*), @as(@Vector(LANE_COUNT, f32), bv[0..LANE_COUNT].*), sum_vec);
    }

    var total = dotVec(sum_vec, @splat(1));

    while (i < len) : (i += 1) {
        total += a[i] * b[i];
    }
    return total;
}

pub fn norm(x: []const f32) f32 {
    return @sqrt(dot(x, x));
}

pub fn scale(v: []f32, s: f32) void {
    const len = v.len;
    const s_vec: @Vector(LANE_COUNT, f32) = @splat(s);

    var i: usize = 0;
    while (i + LANE_COUNT <= len) : (i += LANE_COUNT) {
        const vv: @Vector(LANE_COUNT, f32) = v[i..][0..LANE_COUNT].*;
        @as(*[LANE_COUNT]f32, @ptrCast(v[i..])).* = vv * s_vec;
    }

    while (i < len) : (i += 1) {
        v[i] *= s;
    }
}

pub fn addScaled(out: []f32, a: []const f32, b: []const f32, scale_b: f32) void {
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);
    const len = a.len;
    const sb_vec: @Vector(LANE_COUNT, f32) = @splat(scale_b);

    var i: usize = 0;
    while (i + LANE_COUNT <= len) : (i += LANE_COUNT) {
        const av: @Vector(LANE_COUNT, f32) = a[i..][0..LANE_COUNT].*;
        const bv: @Vector(LANE_COUNT, f32) = b[i..][0..LANE_COUNT].*;
        @as(*[LANE_COUNT]f32, @ptrCast(out[i..])).* = av + bv * sb_vec;
    }

    while (i < len) : (i += 1) {
        out[i] = a[i] + b[i] * scale_b;
    }
}

pub fn sub(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);
    const len = a.len;

    var i: usize = 0;
    while (i + LANE_COUNT <= len) : (i += LANE_COUNT) {
        const av: @Vector(LANE_COUNT, f32) = a[i..][0..LANE_COUNT].*;
        const bv: @Vector(LANE_COUNT, f32) = b[i..][0..LANE_COUNT].*;
        @as(*[LANE_COUNT]f32, @ptrCast(out[i..])).* = av - bv;
    }

    while (i < len) : (i += 1) {
        out[i] = a[i] - b[i];
    }
}

pub fn copy(src: []const f32, dst: []f32) void {
    std.debug.assert(dst.len == src.len);
    @memcpy(dst, src);
}

pub fn zero(v: []f32) void {
    @memset(v, 0);
}

pub fn addInPlace(a: []f32, b: []const f32) void {
    std.debug.assert(a.len == b.len);
    const len = a.len;

    var i: usize = 0;
    while (i + LANE_COUNT <= len) : (i += LANE_COUNT) {
        const av: @Vector(LANE_COUNT, f32) = a[i..][0..LANE_COUNT].*;
        const bv: @Vector(LANE_COUNT, f32) = b[i..][0..LANE_COUNT].*;
        @as(*[LANE_COUNT]f32, @ptrCast(a[i..])).* = av + bv;
    }

    while (i < len) : (i += 1) {
        a[i] += b[i];
    }
}

test "dot simple" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    try std.testing.expectEqual(32.0, dot(&a, &b));
}

test "norm known vector" {
    const v = [_]f32{ 3.0, 4.0 };
    try std.testing.expectEqual(5.0, norm(&v));
}

test "scale in place" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    scale(&v, 2.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, 4.0, 6.0 }, &v);
}

test "sub produces residual" {
    const a = [_]f32{ 5.0, 10.0 };
    const b = [_]f32{ 2.0, 3.0 };
    var out: [2]f32 = undefined;
    sub(&a, &b, &out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 7.0 }, &out);
}

test "copy exact" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;
    copy(&src, &dst);
    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "zero clears" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    zero(&v);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 0.0 }, &v);
}

test "addInPlace" {
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 0.5, 1.0, 1.5 };
    addInPlace(&a, &b);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.5, 3.0, 4.5 }, &a);
}
