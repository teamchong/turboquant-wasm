const std = @import("std");

fn addTurboquantModules(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const math_mod = b.addModule("math", .{
        .root_source_file = b.path("src/math.zig"),
        .target = target,
    });
    _ = math_mod;
    const rotation_mod = b.addModule("rotation", .{
        .root_source_file = b.path("src/rotation.zig"),
        .target = target,
    });
    _ = rotation_mod;
    const polar_mod = b.addModule("polar", .{
        .root_source_file = b.path("src/polar.zig"),
        .target = target,
    });
    _ = polar_mod;
    const qjl_mod = b.addModule("qjl", .{
        .root_source_file = b.path("src/qjl.zig"),
        .target = target,
    });
    _ = qjl_mod;
    const format_mod = b.addModule("format", .{
        .root_source_file = b.path("src/format.zig"),
        .target = target,
    });
    _ = format_mod;

    const turboquant_mod = b.addModule("turboquant", .{
        .root_source_file = b.path("src/turboquant.zig"),
        .target = target,
    });
    turboquant_mod.addImport("math", b.modules.get("math").?);
    turboquant_mod.addImport("rotation", b.modules.get("rotation").?);
    turboquant_mod.addImport("polar", b.modules.get("polar").?);
    turboquant_mod.addImport("qjl", b.modules.get("qjl").?);
    turboquant_mod.addImport("format", b.modules.get("format").?);

    return turboquant_mod;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const turboquant_mod = addTurboquantModules(b, target);

    // -- Tests --
    const tests = b.addTest(.{
        .root_module = turboquant_mod,
    });

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // -- Quality benchmark --
    const quality_mod = b.addModule("quality", .{
        .root_source_file = b.path("benchmarks/quality.zig"),
        .target = target,
    });
    quality_mod.addImport("turboquant", turboquant_mod);

    const quality_exe = b.addExecutable(.{
        .name = "quality",
        .root_module = quality_mod,
    });

    const quality_run = b.addRunArtifact(quality_exe);
    if (b.args) |args| quality_run.addArgs(args);

    const quality_step = b.step("quality", "Run quality benchmarks");
    quality_step.dependOn(&quality_run.step);

    // -- WASM build (freestanding, relaxed SIMD) --
    const wasm_step = b.step("wasm", "Build WASM module with relaxed SIMD");

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = std.Target.wasm.featureSet(&.{ .simd128, .relaxed_simd }),
    });

    const wasm_turboquant = b.createModule(.{
        .root_source_file = b.path("src/turboquant.zig"),
        .target = wasm_target,
    });
    wasm_turboquant.addImport("math", b.createModule(.{
        .root_source_file = b.path("src/math.zig"),
        .target = wasm_target,
    }));
    wasm_turboquant.addImport("rotation", b.createModule(.{
        .root_source_file = b.path("src/rotation.zig"),
        .target = wasm_target,
    }));
    wasm_turboquant.addImport("polar", b.createModule(.{
        .root_source_file = b.path("src/polar.zig"),
        .target = wasm_target,
    }));
    wasm_turboquant.addImport("qjl", b.createModule(.{
        .root_source_file = b.path("src/qjl.zig"),
        .target = wasm_target,
    }));
    wasm_turboquant.addImport("format", b.createModule(.{
        .root_source_file = b.path("src/format.zig"),
        .target = wasm_target,
    }));

    const wasm_exports_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_exports.zig"),
        .target = wasm_target,
    });
    wasm_exports_mod.addImport("turboquant", wasm_turboquant);

    const wasm_lib = b.addExecutable(.{
        .name = "turboquant",
        .root_module = wasm_exports_mod,
    });
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;

    // Install to dist/ for npm packaging
    const install_wasm = b.addInstallArtifact(wasm_lib, .{
        .dest_dir = .{ .override = .{ .custom = "../dist" } },
    });
    wasm_step.dependOn(&install_wasm.step);

    // -- WASM LLM build (ORT + TurboQuant, one binary) --
    const llm_step = b.step("wasm-llm", "Build ORT+TQ unified WASM");

    const llm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .wasi,
        .cpu_features_add = std.Target.wasm.featureSet(&.{ .simd128, .relaxed_simd }),
    });

    // ORT C++ compiled as a static library
    const ort_lib = b.addLibrary(.{
        .name = "ort",
        .root_module = b.createModule(.{
            .target = llm_target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        }),
    });

    // C++ compile flags
    const cpp_flags: []const []const u8 = &.{
        "-std=c++17",
        "-DONNX_ML",
        "-DONNX_NAMESPACE=onnx",
        "-D__wasm__",
        "-DORT_API_MANUAL_INIT",
        "-DDISABLE_FLOAT8_T",
        "-DUSE_JSEP",
        "-DMLAS_NO_ONNXRUNTIME_THREADPOOL",
        "-Wno-deprecated-declarations",
    };

    // Force-include shims
    const force_includes: []const []const u8 = &.{
        "-include", "vendor/ort-shims/fstream",
        "-include", "vendor/ort-shims/mutex",
        "-include", "vendor/ort-shims/shared_mutex",
        "-include", "vendor/ort-shims/condition_variable",
        "-include", "vendor/ort-shims/thread",
        "-include", "vendor/ort-shims/wasm_compat.h",
        "-include", "vendor/ort-shims/thread_stream.h",
    };

    // Include paths
    const ort_root = "vendor/onnxruntime";
    const include_paths: []const []const u8 = &.{
        "vendor/ort-shims",
        ort_root ++ "/include/onnxruntime",
        ort_root ++ "/include",
        ort_root ++ "/onnxruntime",
        ort_root ++ "/include/onnxruntime/core/session",
        ort_root ++ "/cmake/external/abseil-cpp",
        ort_root ++ "/cmake/external/microsoft_gsl/include",
        ort_root ++ "/cmake/external/flatbuffers/include",
        ort_root ++ "/cmake/external/safeint",
        ort_root ++ "/cmake/external/onnx",
        ort_root ++ "/cmake/external/protobuf/src",
        ort_root ++ "/cmake/external/mp11/include",
        ort_root ++ "/cmake/external/date/include",
        ort_root ++ "/cmake/external/json/include",
        ort_root ++ "/cmake/external/eigen",
        ort_root ++ "/onnxruntime/core/mlas/inc",
        ort_root ++ "/onnxruntime/core/mlas/lib",
    };

    for (include_paths) |p| {
        ort_lib.addIncludePath(b.path(p));
    }

    const flags = force_includes ++ cpp_flags;

    // Add ORT core source files
    const core_dirs: []const []const u8 = &.{
        ort_root ++ "/onnxruntime/core/common",
        ort_root ++ "/onnxruntime/core/framework",
        ort_root ++ "/onnxruntime/core/graph",
        ort_root ++ "/onnxruntime/core/session",
        ort_root ++ "/onnxruntime/core/providers/js",
    };
    for (core_dirs) |dir| {
        var walker = std.fs.cwd().openDir(dir, .{ .iterate = true }) catch continue;
        defer walker.close();
        var it = walker.iterate();
        while (it.next() catch null) |entry| {
            if (entry.kind != .file) continue;
            const name = entry.name;
            if (!std.mem.endsWith(u8, name, ".cc")) continue;
            // Skip test files and vitisai
            if (std.mem.indexOf(u8, name, "test") != null) continue;
            if (std.mem.indexOf(u8, name, "vitisai") != null) continue;
            const full_path = std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ dir, name }) catch unreachable;
            ort_lib.addCSourceFile(.{ .file = b.path(full_path), .flags = flags });
        }
    }

    // WASM API
    ort_lib.addCSourceFile(.{ .file = b.path(ort_root ++ "/onnxruntime/wasm/api.cc"), .flags = flags });

    // Build the unified WASM (ORT lib + TQ Zig)
    const llm_tq_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_exports.zig"),
        .target = llm_target,
    });
    llm_tq_mod.addImport("turboquant", wasm_turboquant);

    const llm_exe = b.addExecutable(.{
        .name = "turboquant-llm",
        .root_module = llm_tq_mod,
    });
    llm_exe.entry = .disabled;
    llm_exe.rdynamic = true;
    llm_exe.linkLibrary(ort_lib);

    const install_llm = b.addInstallArtifact(llm_exe, .{
        .dest_dir = .{ .override = .{ .custom = "../dist" } },
    });
    llm_step.dependOn(&install_llm.step);
}
