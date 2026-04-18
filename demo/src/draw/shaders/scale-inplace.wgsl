// In-place scalar multiply. Used for the Gemma embedding scale: h = h * sqrt(hidden_size).
//
// Dispatch: ceil(count / 256) workgroups (wgX), optionally tiled over wgY when
// count / 256 exceeds the workgroup dispatch limit (65535).

struct Params {
  count: u32,
  scale: f32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn scale_inplace(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= params.count) { return; }
  data[idx] = data[idx] * params.scale;
}
