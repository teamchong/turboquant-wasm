// In-place scalar multiply where the scale factor comes from a GPU buffer (not a uniform).
// Used for Gemma 4's blk.N.layer_output_scale tensor, which is a single f32 stored in GGUF.
//
// Dispatch: ceil(count / 256) workgroups.

struct Params {
  count: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scale: array<f32>;  // reads scale[0]
@group(0) @binding(2) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn scale_by_buf(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= params.count) { return; }
  data[idx] = data[idx] * scale[0];
}
