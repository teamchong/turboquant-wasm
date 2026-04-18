// GELU with tanh approximation (used by Gemma 4):
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
//
// Applied element-wise. Used in the gated MLP: output = gelu(gate) * up

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

struct Params {
  count: u32,  // total number of elements
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;   // gate_proj output
@group(0) @binding(2) var<storage, read> up: array<f32>;     // up_proj output
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn gelu_gate(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (idx >= params.count) { return; }

  let x = gate[idx];
  // Clamp tanh argument: tanh(15) = 1.0 in f32, avoids Metal GPU NaN for large args
  let inner = clamp(SQRT_2_OVER_PI * (x + COEFF * x * x * x), -15.0, 15.0);
  let g = 0.5 * x * (1.0 + tanh(inner));
  output[idx] = g * up[idx];
}
