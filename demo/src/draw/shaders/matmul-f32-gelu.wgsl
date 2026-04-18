// F32 matrix-vector multiply with GELU-gated activation fused into the
// input load. Used for the PLE proj matmul when we want to eliminate the
// standalone gelu dispatch + the N_EMBD_PER_LAYER round-trip of
// s.pleActivated through global memory.
//
//     input[i] = gelu(gate[i]) * slice[i]
//
// GELU tanh approximation bit-identical to gelu.wgsl so per-layer
// embedding outputs stay numerically reproducible.

enable subgroups;

struct Params {
  n_rows: u32,
  n_cols: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;    // pleGated (inp_gate output)
@group(0) @binding(3) var<storage, read> slice: array<f32>;   // pleLayerSlice (per-layer embedding slice)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

fn gelu_gated(g: f32, u: f32) -> f32 {
  let inner = clamp(SQRT_2_OVER_PI * (g + COEFF * g * g * g), -15.0, 15.0);
  let gelu = 0.5 * g * (1.0 + tanh(inner));
  return gelu * u;
}

var<workgroup> sact: array<f32, 1536>;

@compute @workgroup_size(WG_SIZE)
fn matmul_f32_gelu(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let tid = lid.x;
  let base_row = (wid.y * nwg.x + wid.x) * ROWS_PER_WG;
  let row_in_wg = tid / THREADS_PER_ROW;
  let lane = tid % THREADS_PER_ROW;
  let row = base_row + row_in_wg;
  let n_cols = params.n_cols;

  // Cooperative fused load: compute gelu_gated(gate[i], slice[i]) directly
  // into shared memory instead of materializing pleActivated in global.
  var i = tid;
  while (i < n_cols) {
    sact[i] = gelu_gated(gate[i], slice[i]);
    i += WG_SIZE;
  }
  workgroupBarrier();

  var sum = 0.0f;
  if (row < params.n_rows) {
    let row_base = row * n_cols;
    var c = lane;
    while (c < n_cols) {
      sum += weights[row_base + c] * sact[c];
      c += THREADS_PER_ROW;
    }
  }

  sum = sum + subgroupShuffleXor(sum, 1u);
  sum = sum + subgroupShuffleXor(sum, 2u);
  sum = sum + subgroupShuffleXor(sum, 4u);

  if (lane == 0u && row < params.n_rows) {
    output[row] = sum;
  }
}
