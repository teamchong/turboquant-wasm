// In-place residual add + scalar scale: accum[i] = (accum[i] + addend[i]) * scale[0]
//
// Fuses the standalone residualAddInplace + scaleByBuf pair used at the tail
// of each transformer layer. scale[0] comes from Gemma 4's per-layer
// blk.N.layer_output_scale tensor (single f32 stored in a GPU buffer).
// Collapsing these into one kernel removes one dispatch per layer per
// decode step (×35).

struct Params {
  count: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> accum: array<f32>;
@group(0) @binding(2) var<storage, read> addend: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;  // scale[0] broadcast

@compute @workgroup_size(256)
fn residual_add_scale_inplace(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= params.count) { return; }
  accum[idx] = (accum[idx] + addend[idx]) * scale[0];
}
