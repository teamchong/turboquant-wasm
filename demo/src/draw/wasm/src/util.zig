const std = @import("std");

/// Copy a slice into a destination buffer. Returns bytes written.
/// Writes as much as fits when dst is smaller than src (partial write).
pub fn copySlice(dst: []u8, src: []const u8) usize {
    const n = @min(dst.len, src.len);
    if (n > 0) @memcpy(dst[0..n], src[0..n]);
    return n;
}

/// Write a JSON-escaped string into a destination buffer. Returns bytes written.
/// Escapes ", \, and control characters (newline, tab, carriage return) for valid JSON.
pub fn copySliceJsonEscaped(dst: []u8, src: []const u8) usize {
    var written: usize = 0;
    for (src) |ch| {
        if (ch == '"' or ch == '\\') {
            if (written + 2 > dst.len) break;
            dst[written] = '\\';
            dst[written + 1] = ch;
            written += 2;
        } else if (ch == '\n') {
            if (written + 2 > dst.len) break;
            dst[written] = '\\';
            dst[written + 1] = 'n';
            written += 2;
        } else if (ch == '\r') {
            if (written + 2 > dst.len) break;
            dst[written] = '\\';
            dst[written + 1] = 'r';
            written += 2;
        } else if (ch == '\t') {
            if (written + 2 > dst.len) break;
            dst[written] = '\\';
            dst[written + 1] = 't';
            written += 2;
        } else if (ch < 0x20) {
            // Other control characters — skip (rare in practice)
            continue;
        } else {
            if (written >= dst.len) break;
            dst[written] = ch;
            written += 1;
        }
    }
    return written;
}

/// Write an i32 as decimal text into a buffer. Returns bytes written.
pub fn writeInt(dst: []u8, val: i32) usize {
    if (dst.len < 12) return 0; // max i32 is 11 chars + sign

    // Handle i32 min separately to avoid overflow on negation
    if (val == std.math.minInt(i32)) {
        const literal = "-2147483648";
        @memcpy(dst[0..literal.len], literal);
        return literal.len;
    }

    var buf: [12]u8 = undefined;
    var v = val;
    var len: usize = 0;

    if (v < 0) {
        dst[0] = '-';
        v = -v;
        len = 1;
    }

    if (v == 0) {
        dst[len] = '0';
        return len + 1;
    }

    var digit_count: usize = 0;
    var tmp = v;
    while (tmp > 0) : (tmp = @divTrunc(tmp, 10)) {
        buf[digit_count] = @intCast(@as(u32, @intCast(@rem(tmp, 10))) + '0');
        digit_count += 1;
    }

    var i: usize = 0;
    while (i < digit_count) : (i += 1) {
        dst[len + i] = buf[digit_count - 1 - i];
    }

    return len + digit_count;
}

/// Find the matching closing brace for a '{' at position 0.
/// Correctly handles quoted strings (including escaped quotes).
pub fn findMatchingBrace(json: []const u8) usize {
    var depth: i32 = 0;
    var in_string = false;
    var prev_backslash = false;
    for (json, 0..) |c, i| {
        if (in_string) {
            if (c == '"' and !prev_backslash) {
                in_string = false;
            }
            prev_backslash = (c == '\\' and !prev_backslash);
        } else {
            if (c == '"') in_string = true;
            if (c == '{') depth += 1;
            if (c == '}') {
                depth -= 1;
                if (depth == 0) return i + 1;
            }
        }
    }
    return json.len;
}

/// Extract a string value for a given field name from a JSON object slice.
/// Handles escaped quotes within string values.
/// Only matches JSON keys (requires ':' after the quoted name), not values.
pub fn extractStringField(obj: []const u8, field: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i + field.len + 3 < obj.len) : (i += 1) {
        if (obj[i] == '"' and i + 1 + field.len < obj.len and
            std.mem.eql(u8, obj[i + 1 .. i + 1 + field.len], field) and
            obj[i + 1 + field.len] == '"')
        {
            // Verify this is a key (colon must follow), not a string value
            var j = i + 1 + field.len + 1;
            while (j < obj.len and obj[j] == ' ') : (j += 1) {}
            if (j >= obj.len or obj[j] != ':') continue;
            j += 1; // skip colon
            while (j < obj.len and obj[j] == ' ') : (j += 1) {}
            if (j < obj.len and obj[j] == '"') {
                j += 1;
                const start = j;
                while (j < obj.len) {
                    if (obj[j] == '"') {
                        // Count consecutive preceding backslashes
                        var bs: usize = 0;
                        var k = j;
                        while (k > start and obj[k - 1] == '\\') {
                            bs += 1;
                            k -= 1;
                        }
                        // Quote is unescaped if preceded by even number of backslashes
                        if (bs % 2 == 0) break;
                    }
                    j += 1;
                }
                return obj[start..j];
            }
        }
    }
    return null;
}

/// Extract an integer value for a given field name from a JSON object slice.
/// The field parameter should be the bare field name (without quotes).
/// Only matches JSON keys (requires ':' after the quoted name), not values.
/// Returns null for missing fields or JSON null values.
pub fn extractIntField(obj: []const u8, field: []const u8) ?i32 {
    var i: usize = 0;
    while (i + field.len + 3 < obj.len) : (i += 1) {
        if (obj[i] == '"' and i + 1 + field.len < obj.len and
            std.mem.eql(u8, obj[i + 1 .. i + 1 + field.len], field) and
            obj[i + 1 + field.len] == '"')
        {
            // Verify this is a key (colon must follow), not a string value
            var j = i + 1 + field.len + 1;
            while (j < obj.len and obj[j] == ' ') : (j += 1) {}
            if (j >= obj.len or obj[j] != ':') continue;
            j += 1; // skip colon
            while (j < obj.len and obj[j] == ' ') : (j += 1) {}
            if (j >= obj.len) return null;
            if (j + 4 <= obj.len and std.mem.eql(u8, obj[j .. j + 4], "null")) return null;

            var negative = false;
            if (obj[j] == '-') {
                negative = true;
                j += 1;
            }
            // Must start with a digit; otherwise value is not numeric
            if (j >= obj.len or obj[j] < '0' or obj[j] > '9') return null;
            var val: i32 = 0;
            while (j < obj.len and obj[j] >= '0' and obj[j] <= '9') : (j += 1) {
                // Saturating arithmetic: clamps on overflow (values > i32 max)
                val = val *| 10 +| @as(i32, @intCast(obj[j] - '0'));
            }
            // Skip decimal portion for floats (e.g., "100.5" → 100)
            if (j < obj.len and obj[j] == '.') {
                j += 1;
                // Round: check first decimal digit
                const round_up = (j < obj.len and obj[j] >= '5' and obj[j] <= '9');
                if (round_up) val += 1;
            }
            return if (negative) -val else val;
        }
    }
    return null;
}

/// Extract a string field from a nested JSON object.
/// E.g., extractNestedStringField(obj, "startBinding", "elementId")
/// Only matches JSON keys (requires ':' after the quoted name), not values.
pub fn extractNestedStringField(obj: []const u8, outer: []const u8, inner: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i + outer.len + 3 < obj.len) : (i += 1) {
        if (obj[i] == '"' and i + 1 + outer.len < obj.len and
            std.mem.eql(u8, obj[i + 1 .. i + 1 + outer.len], outer) and
            obj[i + 1 + outer.len] == '"')
        {
            // Verify this is a key (colon must follow)
            var j = i + 1 + outer.len + 1;
            while (j < obj.len and obj[j] == ' ') : (j += 1) {}
            if (j >= obj.len or obj[j] != ':') continue;
            j += 1;
            while (j < obj.len and obj[j] != '{') : (j += 1) {}
            if (j >= obj.len) return null;
            const nested_end = findMatchingBrace(obj[j..]) + j;
            return extractStringField(obj[j..nested_end], inner);
        }
    }
    return null;
}
