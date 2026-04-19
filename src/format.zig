const std = @import("std");

pub const FormatError = error{
    InvalidHeader,
    InvalidPayload,
    OutOfMemory,
};

pub const Header = packed struct {
    version: u8,
    dim: u32,
    reserved: u8,
    polar_bytes: u32,
    qjl_bytes: u32,
    max_r: f32,
    gamma: f32,
};

pub const PAYLOAD_VERSION: u8 = 1;
pub const HEADER_SIZE: usize = @sizeOf(Header);

pub fn writeHeader(
    out: []u8,
    dim: u32,
    polar_bytes: u32,
    qjl_bytes: u32,
    max_r: f32,
    gamma: f32,
) void {
    std.debug.assert(out.len >= HEADER_SIZE);
    out[0] = PAYLOAD_VERSION;
    std.mem.writeInt(u32, out[1..5], dim, .little);
    out[5] = 0;
    std.mem.writeInt(u32, out[6..10], polar_bytes, .little);
    std.mem.writeInt(u32, out[10..14], qjl_bytes, .little);
    const max_r_bits: u32 = @bitCast(max_r);
    std.mem.writeInt(u32, out[14..18], max_r_bits, .little);
    const gamma_bits: u32 = @bitCast(gamma);
    std.mem.writeInt(u32, out[18..22], gamma_bits, .little);
}

pub fn readHeader(data: []const u8) FormatError!Header {
    if (data.len < HEADER_SIZE) return FormatError.InvalidHeader;
    if (data[0] != PAYLOAD_VERSION) return FormatError.InvalidHeader;
    if (data[5] != 0) return FormatError.InvalidHeader;

    const dim = std.mem.readInt(u32, data[1..5], .little);
    const polar_bytes = std.mem.readInt(u32, data[6..10], .little);
    const qjl_bytes = std.mem.readInt(u32, data[10..14], .little);
    const max_r_bits = std.mem.readInt(u32, data[14..18], .little);
    const max_r: f32 = @bitCast(max_r_bits);
    const gamma_bits = std.mem.readInt(u32, data[18..22], .little);
    const gamma: f32 = @bitCast(gamma_bits);

    return Header{
        .version = data[0],
        .dim = dim,
        .reserved = 0,
        .polar_bytes = polar_bytes,
        .qjl_bytes = qjl_bytes,
        .max_r = max_r,
        .gamma = gamma,
    };
}

pub fn slicePayload(data: []const u8, header: Header) FormatError!struct { polar: []const u8, qjl: []const u8 } {
    const payload_start = HEADER_SIZE;
    // Guard against u32 overflow when summing polar_bytes + qjl_bytes
    const payload_len = std.math.add(u32, header.polar_bytes, header.qjl_bytes) catch return FormatError.InvalidPayload;
    const payload_end = payload_start + payload_len;
    if (data.len < payload_end) return FormatError.InvalidPayload;

    return .{
        .polar = data[payload_start .. payload_start + header.polar_bytes],
        .qjl = data[payload_start + header.polar_bytes .. payload_end],
    };
}

test "header roundtrip" {
    const dim: u32 = 128;
    const polar_bytes: u32 = 48;
    const qjl_bytes: u32 = 16;
    const max_r: f32 = 2.5;
    const gamma: f32 = 3.14159;

    var buf: [HEADER_SIZE]u8 = undefined;
    writeHeader(&buf, dim, polar_bytes, qjl_bytes, max_r, gamma);

    const header = try readHeader(&buf);
    try std.testing.expectEqual(dim, header.dim);
    try std.testing.expectEqual(polar_bytes, header.polar_bytes);
    try std.testing.expectEqual(qjl_bytes, header.qjl_bytes);
    try std.testing.expectEqual(max_r, header.max_r);
    try std.testing.expectEqual(gamma, header.gamma);
    try std.testing.expectEqual(@as(u8, 1), header.version);
}

test "reject short header" {
    const bad: [5]u8 = .{ 1, 0, 0, 0, 0 };
    const result = readHeader(&bad);
    try std.testing.expectError(FormatError.InvalidHeader, result);
}

test "reject wrong version" {
    var buf: [HEADER_SIZE]u8 = undefined;
    buf[0] = 99;
    const result = readHeader(&buf);
    try std.testing.expectError(FormatError.InvalidHeader, result);
}

test "reject nonzero reserved byte" {
    var buf: [HEADER_SIZE]u8 = undefined;
    buf[0] = 1;
    buf[5] = 1;
    const result = readHeader(&buf);
    try std.testing.expectError(FormatError.InvalidHeader, result);
}

test "payload slicing" {
    const dim: u32 = 8;
    const polar_bytes: u32 = 3;
    const qjl_bytes: u32 = 1;
    const max_r: f32 = 1.0;
    const gamma: f32 = 0.5;

    var buf: [HEADER_SIZE + polar_bytes + qjl_bytes]u8 = undefined;
    writeHeader(&buf, dim, polar_bytes, qjl_bytes, max_r, gamma);
    buf[HEADER_SIZE] = 0xAA;
    buf[HEADER_SIZE + 1] = 0xBB;
    buf[HEADER_SIZE + 2] = 0xCC;
    buf[HEADER_SIZE + 3] = 0xDD;

    const header = try readHeader(&buf);
    const payload = try slicePayload(&buf, header);

    try std.testing.expectEqual(@as(usize, 3), payload.polar.len);
    try std.testing.expectEqual(@as(usize, 1), payload.qjl.len);
    try std.testing.expectEqual(@as(u8, 0xAA), payload.polar[0]);
    try std.testing.expectEqual(@as(u8, 0xDD), payload.qjl[0]);
}

test "reject truncated payload" {
    var buf: [130]u8 = undefined;
    writeHeader(&buf, 8, 100, 1, 1.0, 0.5);
    const header = try readHeader(&buf);
    const result = slicePayload(&buf, header);
    try std.testing.expectError(FormatError.InvalidPayload, result);
}
