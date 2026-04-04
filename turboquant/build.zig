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
        .os_tag = .wasi,
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
}
