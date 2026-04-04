const std = @import("std");
const tq = @import("turboquant");
const turboquant = tq;
const math_mod = tq.math;
const format = tq.format;
const polar = tq.polar;

const QualityError = error{
    MissingArgs,
    InvalidDim,
    InvalidParam,
    OddDimension,
};

const Config = struct {
    dim: usize,
    n: usize,
    k: usize,
    num_queries: usize,
    seed: u32,
};

fn parseArgs(args: []const [:0]u8) QualityError!Config {
    if (args.len < 2) return QualityError.MissingArgs;

    const dim_str = std.mem.sliceTo(args[1], 0);
    const dim = std.fmt.parseInt(usize, dim_str, 10) catch return QualityError.InvalidDim;
    if (dim == 0) return QualityError.InvalidDim;
    if (dim % 2 != 0) return QualityError.OddDimension;

    const n: usize = if (args.len > 2) std.fmt.parseInt(usize, std.mem.sliceTo(args[2], 0), 10) catch return QualityError.InvalidParam else 1000;
    const k: usize = if (args.len > 3) std.fmt.parseInt(usize, std.mem.sliceTo(args[3], 0), 10) catch return QualityError.InvalidParam else 10;
    const num_queries: usize = if (args.len > 4) std.fmt.parseInt(usize, std.mem.sliceTo(args[4], 0), 10) catch return QualityError.InvalidParam else 50;

    return .{ .dim = dim, .n = n, .k = k, .num_queries = num_queries, .seed = 42 };
}

fn generateUnitSphere(allocator: std.mem.Allocator, dim: usize, rng_state: *std.Random.DefaultPrng) ![]f32 {
    const vec = try allocator.alloc(f32, dim);
    errdefer allocator.free(vec);
    const r = rng_state.random();

    var norm_sq: f32 = 0;
    for (0..dim) |i| {
        vec[i] = r.float(f32) * 2 - 1;
        norm_sq += vec[i] * vec[i];
    }
    const inv_norm = 1.0 / @sqrt(norm_sq);
    for (0..dim) |i| vec[i] *= inv_norm;
    return vec;
}

const IndexScore = struct {
    idx: usize,
    score: f32,
};

fn scoreDesc(_: void, a: IndexScore, b: IndexScore) bool {
    return a.score > b.score;
}

pub fn main() void {
    const allocator = std.heap.page_allocator;

    const args = std.process.argsAlloc(allocator) catch {
        std.debug.print("error: out of memory\n", .{});
        return;
    };
    defer std.process.argsFree(allocator, args);

    const config = parseArgs(args) catch |err| {
        switch (err) {
            QualityError.MissingArgs => {
                std.debug.print("Usage: quality <dim> [N] [k] [num_queries]\n", .{});
                std.debug.print("  dim:         vector dimension (must be even)\n", .{});
                std.debug.print("  N:           database size (default 1000)\n", .{});
                std.debug.print("  k:           top-k for recall (default 10)\n", .{});
                std.debug.print("  num_queries: number of queries (default 50)\n", .{});
            },
            QualityError.InvalidDim => std.debug.print("error: invalid dimension\n", .{}),
            QualityError.OddDimension => std.debug.print("error: dimension must be even\n", .{}),
            QualityError.InvalidParam => std.debug.print("error: invalid parameter\n", .{}),
        }
        return;
    };

    std.debug.print("=== RECALL@{} BENCHMARK ===\n", .{config.k});
    std.debug.print("dim={}, N={}, queries={}\n\n", .{ config.dim, config.n, config.num_queries });

    // Init engine
    var engine = turboquant.Engine.init(allocator, .{ .dim = config.dim, .seed = config.seed }) catch {
        std.debug.print("error: engine init failed\n", .{});
        return;
    };
    defer engine.deinit(allocator);

    // Generate database vectors
    var db_rng = std.Random.DefaultPrng.init(config.seed);
    const db_vecs = allocator.alloc([]f32, config.n) catch {
        std.debug.print("error: alloc db_vecs\n", .{});
        return;
    };
    const db_rotated = allocator.alloc([]f32, config.n) catch {
        std.debug.print("error: alloc db_rotated\n", .{});
        return;
    };
    const db_compressed = allocator.alloc([]u8, config.n) catch {
        std.debug.print("error: alloc db_compressed\n", .{});
        return;
    };

    // Generate and encode all database vectors
    var encode_timer = std.time.Timer.start() catch unreachable;
    for (0..config.n) |i| {
        db_vecs[i] = generateUnitSphere(allocator, config.dim, &db_rng) catch {
            std.debug.print("error: generating vector {}\n", .{i});
            return;
        };

        db_rotated[i] = allocator.alloc(f32, config.dim) catch {
            std.debug.print("error: alloc rotated {}\n", .{i});
            return;
        };
        engine.rot_op.rotate(db_vecs[i], db_rotated[i]);

        db_compressed[i] = engine.encode(allocator, db_vecs[i]) catch {
            std.debug.print("error: encoding vector {}\n", .{i});
            return;
        };
    }
    const encode_ns = encode_timer.read();
    const encode_ms = @as(f64, @floatFromInt(encode_ns)) / 1_000_000.0;
    std.debug.print("encode: {d:.1}ms total ({d:.1}us/vec)\n", .{ encode_ms, encode_ms * 1000.0 / @as(f64, @floatFromInt(config.n)) });

    // Run queries
    var query_rng = std.Random.DefaultPrng.init(config.seed + 1000);
    var total_recall: f64 = 0;
    var total_polar_recall: f64 = 0;
    var sum_dot_err: f64 = 0;
    var query_ns_total: u64 = 0;

    const true_scores = allocator.alloc(IndexScore, config.n) catch {
        std.debug.print("error: alloc scores\n", .{});
        return;
    };
    const est_scores = allocator.alloc(IndexScore, config.n) catch {
        std.debug.print("error: alloc scores\n", .{});
        return;
    };
    const polar_scores = allocator.alloc(IndexScore, config.n) catch {
        std.debug.print("error: alloc scores\n", .{});
        return;
    };

    for (0..config.num_queries) |_| {
        const q = generateUnitSphere(allocator, config.dim, &query_rng) catch {
            std.debug.print("error: generating query\n", .{});
            return;
        };
        defer allocator.free(q);

        // True dot products (q dot R*x)
        for (0..config.n) |i| {
            true_scores[i] = .{ .idx = i, .score = math_mod.dot(q, db_rotated[i]) };
        }

        // Estimated dot products (full: polar + QJL)
        var query_timer = std.time.Timer.start() catch unreachable;
        for (0..config.n) |i| {
            est_scores[i] = .{ .idx = i, .score = engine.dot(q, db_compressed[i]) };
        }
        query_ns_total += query_timer.read();

        // Polar-only dot products
        for (0..config.n) |i| {
            const header = format.readHeader(db_compressed[i]) catch continue;
            const payload = format.slicePayload(db_compressed[i], header) catch continue;
            polar_scores[i] = .{ .idx = i, .score = polar.dotProduct(q, payload.polar, header.max_r) };
        }

        // Sort all by score descending
        std.mem.sort(IndexScore, true_scores, {}, scoreDesc);
        std.mem.sort(IndexScore, est_scores, {}, scoreDesc);
        std.mem.sort(IndexScore, polar_scores, {}, scoreDesc);

        // Recall@k: is true top-1 in estimated top-k?
        const true_best = true_scores[0].idx;
        for (0..config.k) |j| {
            if (est_scores[j].idx == true_best) {
                total_recall += 1;
                break;
            }
        }
        for (0..config.k) |j| {
            if (polar_scores[j].idx == true_best) {
                total_polar_recall += 1;
                break;
            }
        }

        // Accumulate dot product error
        for (0..config.n) |i| {
            const true_d: f64 = @as(f64, true_scores[i].score);
            const est_d: f64 = @as(f64, est_scores[i].score);
            sum_dot_err += @abs(true_d - est_d);
        }
    }

    const nq: f64 = @floatFromInt(config.num_queries);
    const recall = total_recall / nq;
    const polar_recall = total_polar_recall / nq;
    const query_ms = @as(f64, @floatFromInt(query_ns_total)) / 1_000_000.0;
    const mean_dot_err = sum_dot_err / (nq * @as(f64, @floatFromInt(config.n)));

    std.debug.print("query:  {d:.1}ms total ({d:.1}us/query)\n\n", .{ query_ms, query_ms * 1000.0 / nq });

    std.debug.print("=== RESULTS ===\n", .{});
    std.debug.print("recall@{} (polar+QJL): {d:.4}\n", .{ config.k, recall });
    std.debug.print("recall@{} (polar-only): {d:.4}\n", .{ config.k, polar_recall });
    std.debug.print("mean |dot error|:      {e}\n", .{@as(f32, @floatCast(mean_dot_err))});
}
