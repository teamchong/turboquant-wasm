# TurboQuant

A Zig implementation of Google's TurboQuant vector compression library.

## Installation

Add to your `build.zig.zon`:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .path = "path/to/turboquant/turboquant",
        },
    },
}
```

## Usage

```zig
const turboquant = @import("turboquant");

// Create an engine for repeated operations
var engine = try turboquant.Engine.init(allocator, .{ .dim = 1024, .seed = 12345 });
defer engine.deinit(allocator);

// Encode
const compressed = try engine.encode(allocator, my_vector);
defer allocator.free(compressed);

// Decode
const decoded = try engine.decode(allocator, compressed);
defer allocator.free(decoded);

// Fast dot without decode
const score = engine.dot(query_vector, compressed);
```

## API

- `Engine.init(allocator, .{ .dim, .seed })` - Create engine
- `engine.deinit(allocator)` - Destroy engine
- `engine.encode(allocator, vector)` - Compress vector
- `engine.decode(allocator, compressed)` - Decompress
- `engine.dot(query, compressed)` - Dot product without full decode

## Performance

| Dim | encode | decode | dot |
|-----|--------|--------|-----|
| 256 | 189 µs | 84 µs | 73 µs |
| 512 | 586 µs | 279 µs | 265 µs |
| 1024 | 2105 µs | 1032 µs | 997 µs |

## Compression

- ~6x compression ratio at dim=1024
- ~3 bits/dim

## Building

```bash
cd turboquant
zig build-exe -O ReleaseFast -target aarch64-macos-none src/profile.zig
```

## License

MIT
