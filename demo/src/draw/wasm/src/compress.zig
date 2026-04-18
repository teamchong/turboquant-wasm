const std = @import("std");

/// Compress data using zlib with fixed-Huffman deflate (RFC 1950/1951).
/// Produces proper deflate-compressed output compatible with pako.inflate.
///
/// Format: [2-byte zlib header] [deflate blocks with fixed Huffman] [4-byte Adler-32]
pub fn zlibCompress(input: []const u8, out: []u8) !usize {
    if (out.len < 8) return error.NoSpaceLeft;

    // Zlib header: CMF=0x78 (CM=8 deflate, CINFO=7 32K window), FLG=0x9C (FLEVEL=2 default, FCHECK=28)
    // 0x789C % 31 == 0 ✓
    out[0] = 0x78;
    out[1] = 0x9C;

    var bw = BitWriter{ .out = out, .pos = 2 };

    // BFINAL=1 (single block), BTYPE=01 (fixed Huffman)
    try bw.writeBits(1, 1); // BFINAL
    try bw.writeBits(1, 2); // BTYPE=01 (fixed Huffman), LSB first: 01 → write value 1

    // Encode each input byte as a literal using RFC 1951 fixed Huffman codes
    for (input) |byte| {
        try writeFixedCode(&bw, @as(u16, byte));
    }

    // End of block (symbol 256)
    try writeFixedCode(&bw, 256);

    // Flush remaining bits (pad to byte boundary with zeros)
    try bw.flushBits();

    // Adler-32 checksum (big-endian, per RFC 1950)
    if (bw.pos + 4 > out.len) return error.NoSpaceLeft;
    const checksum = adler32(input);
    std.mem.writeInt(u32, out[bw.pos..][0..4], checksum, .big);

    return bw.pos + 4;
}

// ── Bit Writer ──

const BitWriter = struct {
    out: []u8,
    pos: usize = 0,
    bits: u32 = 0,
    nbits: u32 = 0, // bits buffered (max ~16 between flushes)

    /// Write `n` bits of `value` into the bitstream, LSB first.
    fn writeBits(self: *@This(), value: u32, n: u32) !void {
        self.bits |= value << @intCast(self.nbits);
        self.nbits += n;
        while (self.nbits >= 8) {
            if (self.pos >= self.out.len) return error.NoSpaceLeft;
            self.out[self.pos] = @truncate(self.bits);
            self.pos += 1;
            self.bits >>= 8;
            self.nbits -= 8;
        }
    }

    /// Flush remaining bits, padding with zeros to byte boundary.
    fn flushBits(self: *@This()) !void {
        if (self.nbits > 0) {
            if (self.pos >= self.out.len) return error.NoSpaceLeft;
            self.out[self.pos] = @truncate(self.bits);
            self.pos += 1;
            self.bits = 0;
            self.nbits = 0;
        }
    }
};

// ── Fixed Huffman Codes (RFC 1951 §3.2.6) ──
//
// Symbol    Bits  Code range
// 0-143      8    00110000 - 10111111
// 144-255    9    110010000 - 111111111
// 256-279    7    0000000 - 0010111
// 280-287    8    11000000 - 11000111
//
// Codes are written MSB-first into the LSB-first bitstream,
// so we reverse the bits before writing.

fn writeFixedCode(bw: *BitWriter, symbol: u16) !void {
    if (symbol <= 143) {
        const code: u16 = 0x30 + symbol;
        try bw.writeBits(reverseBits(code, 8), 8);
    } else if (symbol <= 255) {
        const code: u16 = 0x190 + (symbol - 144);
        try bw.writeBits(reverseBits(code, 9), 9);
    } else if (symbol <= 279) {
        const code: u16 = symbol - 256;
        try bw.writeBits(reverseBits(code, 7), 7);
    } else {
        const code: u16 = 0xC0 + (symbol - 280);
        try bw.writeBits(reverseBits(code, 8), 8);
    }
}

/// Reverse the bottom `n` bits of `code`.
fn reverseBits(code: u16, n: u32) u32 {
    var result: u32 = 0;
    var c: u32 = code;
    for (0..n) |_| {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    return result;
}

// ── Adler-32 (RFC 1950) ──

fn adler32(data: []const u8) u32 {
    const MOD_ADLER: u32 = 65521;
    var a: u32 = 1;
    var b: u32 = 0;
    for (data) |byte| {
        a = (a + byte) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    return (b << 16) | a;
}

// ── Tests ──

test "zlibCompress produces valid zlib header" {
    const input = "Hello, Excalidraw!";
    var compressed: [4096]u8 = undefined;
    const comp_len = try zlibCompress(input, &compressed);
    try std.testing.expect(comp_len > 0);
    // Verify zlib header
    try std.testing.expectEqual(@as(u8, 0x78), compressed[0]);
    try std.testing.expectEqual(@as(u8, 0x9C), compressed[1]);
    // Should be smaller or similar to stored blocks (not 2 + 5 + 18 + 4 = 29)
    try std.testing.expect(comp_len <= 40);
}

test "zlibCompress output is valid deflate structure" {
    const input = "Hello, Excalidraw!";
    var compressed: [4096]u8 = undefined;
    const comp_len = try zlibCompress(input, &compressed);

    // Verify zlib header
    try std.testing.expectEqual(@as(u8, 0x78), compressed[0]);
    try std.testing.expectEqual(@as(u8, 0x9C), compressed[1]);

    // Verify first deflate byte has BFINAL=1, BTYPE=01 (fixed Huffman)
    // Bits: BFINAL(1)=1, BTYPE(2)=01 → low 3 bits = 0b011 = 3
    try std.testing.expectEqual(@as(u8, 3), compressed[2] & 0x07);

    // Verify Adler-32 at end (big-endian)
    const expected_adler = adler32(input);
    const actual_adler = std.mem.readInt(u32, compressed[comp_len - 4 ..][0..4], .big);
    try std.testing.expectEqual(expected_adler, actual_adler);
}

test "zlibCompress empty input" {
    var compressed: [256]u8 = undefined;
    const comp_len = try zlibCompress("", &compressed);
    try std.testing.expect(comp_len > 0);
    try std.testing.expectEqual(@as(u8, 0x78), compressed[0]);
}

test "zlibCompress all byte values" {
    // Test with all 256 byte values (covers all fixed Huffman code ranges)
    var input: [256]u8 = undefined;
    for (&input, 0..) |*b, i| b.* = @truncate(i);

    var compressed: [8192]u8 = undefined;
    const comp_len = try zlibCompress(&input, &compressed);
    try std.testing.expect(comp_len > 0);
    // Should have valid zlib header
    try std.testing.expectEqual(@as(u8, 0x78), compressed[0]);
    try std.testing.expectEqual(@as(u8, 0x9C), compressed[1]);
}

test "zlibCompress large input" {
    var input: [2048]u8 = undefined;
    for (&input, 0..) |*b, i| b.* = @truncate(i);

    var compressed: [16384]u8 = undefined;
    const comp_len = try zlibCompress(&input, &compressed);
    try std.testing.expect(comp_len > 0);
}
