// Convert f16 buffer to f32 buffer on GPU.
// Input: array of u32 where each u32 packs two f16 values.
// Output: array of f32 with twice as many elements.

struct Params { count: u32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;   // packed f16 pairs
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn convert(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  let packed = input[idx];
  let lo = unpack2x16float(packed);
  output[idx * 2u] = lo.x;
  output[idx * 2u + 1u] = lo.y;
}
