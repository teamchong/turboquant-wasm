const std = @import("std");
const turboquant = @import("turboquant.zig");

const BenchError = error{
    MissingArgs,
    InvalidOp,
    InvalidDim,
    InvalidIterations,
};

const Operation = enum {
    encode,
    decode,
    dot,
    compression,
    all,
};

const Config = struct {
    op: Operation,
    dim: usize,
    iterations: usize,
    seed: u32,
};

fn parseArgs(args: []const [:0]u8) BenchError!Config {
    if (args.len < 2) return BenchError.MissingArgs;

    const op_str = std.mem.sliceTo(args[1], 0);
    const op: Operation = if (std.mem.eql(u8, op_str, "encode")) .encode else if (std.mem.eql(u8, op_str, "decode")) .decode else if (std.mem.eql(u8, op_str, "dot")) .dot else if (std.mem.eql(u8, op_str, "compression")) .compression else if (std.mem.eql(u8, op_str, "all")) .all else return BenchError.InvalidOp;

    var dim: usize = 128;
    var iterations: usize = 100;

    if (args.len > 2) {
        const dim_str = std.mem.sliceTo(args[2], 0);
        dim = std.fmt.parseInt(usize, dim_str, 10) catch return BenchError.InvalidDim;
        if (dim == 0) return BenchError.InvalidDim;
    }

    if (args.len > 3) {
        const iter_str = std.mem.sliceTo(args[3], 0);
        iterations = std.fmt.parseInt(usize, iter_str, 10) catch return BenchError.InvalidIterations;
    }

    return .{
        .op = op,
        .dim = dim,
        .iterations = iterations,
        .seed = 12345,
    };
}

fn generateVector(allocator: std.mem.Allocator, dim: usize, seed: u32) ![]f32 {
    const data = try allocator.alloc(f32, dim);
    errdefer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (data) |*v| {
        v.* = r.float(f32) * 10 - 5;
    }

    return data;
}

fn runEncode(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        const compressed = try engine.encode(allocator, data);
        var timer = try std.time.Timer.start();
        const encoded = try engine.encode(allocator, data);
        total_ns += timer.read();
        allocator.free(compressed);
        allocator.free(encoded);
    }
    return total_ns / iterations;
}

fn runDecode(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try engine.encode(allocator, data);
    defer allocator.free(compressed);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        const decoded = try engine.decode(allocator, compressed);
        total_ns += timer.read();
        allocator.free(decoded);
    }
    return total_ns / iterations;
}

fn runDot(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try engine.encode(allocator, data);
    defer allocator.free(compressed);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        _ = engine.dot(query, compressed);
        total_ns += timer.read();
    }
    return total_ns / iterations;
}

fn runDotDecoded(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !u64 {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try engine.encode(allocator, data);
    defer allocator.free(compressed);

    var total_ns: u64 = 0;
    for (0..iterations) |_| {
        var timer = try std.time.Timer.start();
        const decoded = try engine.decode(allocator, compressed);
        var dot_prod: f32 = 0;
        for (0..dim) |i| {
            dot_prod += decoded[i] * query[i];
        }
        total_ns += timer.read();
        allocator.free(decoded);
    }
    return total_ns / iterations;
}

fn runCompression(allocator: std.mem.Allocator, dim: usize, seed: u32) !void {
    var engine = try turboquant.Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try engine.encode(allocator, data);
    defer allocator.free(compressed);

    const raw_bytes = dim * 4;
    const ratio = @as(f64, @floatFromInt(raw_bytes)) / @as(f64, @floatFromInt(compressed.len));
    const bits_per_dim = @as(f64, @floatFromInt(compressed.len * 8)) / @as(f64, @floatFromInt(dim));

    std.debug.print("{d} | {d} | {d:.2}x | {d:.2}\n", .{ dim, compressed.len, ratio, bits_per_dim });
}

pub fn main() void {
    const args = std.process.argsAlloc(std.heap.page_allocator) catch {
        std.debug.print("error: out of memory parsing args\n", .{});
        return;
    };
    defer std.process.argsFree(std.heap.page_allocator, args);

    const config = parseArgs(args) catch |err| {
        switch (err) {
            BenchError.MissingArgs => {
                std.debug.print("Usage: bench <op> [dim] [iterations]\n", .{});
                std.debug.print("  op: encode, decode, dot, compression, all\n", .{});
                std.debug.print("  dim: vector dimension (default: 128)\n", .{});
                std.debug.print("  iterations: default: 100\n", .{});
                std.debug.print("\nExamples:\n", .{});
                std.debug.print("  bench encode 1024        # encode at dim=1024\n", .{});
                std.debug.print("  bench decode 512 50       # decode at dim=512, 50 iterations\n", .{});
                std.debug.print("  bench compression         # show compression ratios\n", .{});
                std.debug.print("  bench all                 # run all benchmarks\n", .{});
            },
            BenchError.InvalidOp => {
                std.debug.print("error: invalid operation '{s}'\n", .{std.mem.sliceTo(args[1], 0)});
            },
            BenchError.InvalidDim => {
                std.debug.print("error: invalid dimension '{s}'\n", .{std.mem.sliceTo(args[2], 0)});
            },
            BenchError.InvalidIterations => {
                std.debug.print("error: invalid iterations '{s}'\n", .{std.mem.sliceTo(args[3], 0)});
            },
        }
        return;
    };

    const allocator = std.heap.page_allocator;

    switch (config.op) {
        .encode => {
            const ns_per_op = runEncode(allocator, config.dim, config.iterations, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("encode/dim={d}: {d} ns/op\n", .{ config.dim, ns_per_op });
        },
        .decode => {
            const ns_per_op = runDecode(allocator, config.dim, config.iterations, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("decode/dim={d}: {d} ns/op\n", .{ config.dim, ns_per_op });
        },
        .dot => {
            const ns_per_op = runDot(allocator, config.dim, config.iterations, config.seed) catch {
                std.debug.print("error: benchmark failed\n", .{});
                return;
            };
            std.debug.print("dot/dim={d}: {d} ns/op\n", .{ config.dim, ns_per_op });
        },
        .compression => {
            const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
            std.debug.print("\n=== COMPRESSION RATIOS ===\n", .{});
            std.debug.print("{s:>4} | {s:>6} | {s:>6} | {s:>8}\n", .{ "dim", "compressed", "ratio", "bits/dim" });
            std.debug.print("------|----------|----------|----------\n", .{});

            for (dims) |dim| {
                runCompression(allocator, dim, config.seed) catch {
                    std.debug.print("error: compression benchmark failed\n", .{});
                    return;
                };
            }
        },
        .all => {
            const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };

            std.debug.print("=== ENCODE BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns_per_op = runEncode(allocator, dim, config.iterations, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns_per_op });
            }

            std.debug.print("\n=== DECODE BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns_per_op = runDecode(allocator, dim, config.iterations, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns_per_op });
            }

            std.debug.print("\n=== DOT BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns_per_op = runDot(allocator, dim, config.iterations, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns_per_op });
            }

            std.debug.print("\n=== DOT DECODED BENCHMARK ===\n", .{});
            std.debug.print("{s:>4} | {s:>12}\n", .{ "dim", "ns/op" });
            std.debug.print("------|------------\n", .{});
            for (dims) |dim| {
                const ns_per_op = runDotDecoded(allocator, dim, config.iterations, config.seed) catch continue;
                std.debug.print("{d:>4} | {d:>12}\n", .{ dim, ns_per_op });
            }

            std.debug.print("\n=== COMPRESSION RATIOS ===\n", .{});
            std.debug.print("{s:>4} | {s:>6} | {s:>6} | {s:>8}\n", .{ "dim", "compressed", "ratio", "bits/dim" });
            std.debug.print("------|----------|----------|----------\n", .{});
            for (dims) |dim| {
                runCompression(allocator, dim, config.seed) catch continue;
            }
        },
    }
}
