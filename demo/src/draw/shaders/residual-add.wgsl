// Element-wise residual add: output[i] = a[i] + b[i]
// Used for skip connections in transformer layers.

struct Params {
  count: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn residual_add(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= params.count) { return; }
  output[idx] = a[idx] + b[idx];
}
