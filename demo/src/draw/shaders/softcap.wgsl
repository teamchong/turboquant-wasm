// Final logit softcapping: output = cap * tanh(x / cap)
// Gemma 4 uses cap = 30.0 on final logits only (no attention softcap).

struct Params {
  count: u32,
  cap: f32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn softcap(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  data[idx] = params.cap * tanh(data[idx] / params.cap);
}
