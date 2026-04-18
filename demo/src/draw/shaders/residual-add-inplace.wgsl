// In-place residual add: accum[i] += addend[i]
// Single read_write binding for the accum, so the read and write share
// a binding (well-defined by WGSL). Used to eliminate a post-residual
// copyBufferToBuffer round-trip in the PLE tail of runLayer.

struct Params {
  count: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> accum: array<f32>;
@group(0) @binding(2) var<storage, read> addend: array<f32>;

@compute @workgroup_size(256)
fn residual_add_inplace(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= params.count) { return; }
  accum[idx] = accum[idx] + addend[idx];
}
