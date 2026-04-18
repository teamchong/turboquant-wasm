// F32 matrix-vector multiply: out = W @ x.
// Used for PLE layers (inp_gate, proj) whose weights are stored as F32.
//
// Optimized layout: 8 threads per row cooperate on the column loop
// (each handles n_cols/8 columns), then reduce via subgroupShuffleXor.
// Shared memory caches the input vector so all rows share one load.

enable subgroups;

struct Params {
  n_rows: u32,
  n_cols: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

var<workgroup> sact: array<f32, 1536>;

@compute @workgroup_size(WG_SIZE)
fn matmul_f32(
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

  // Cooperative input load: 64 threads load n_cols values.
  var i = tid;
  while (i < n_cols) {
    sact[i] = input[i];
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

  // 8-thread butterfly reduction.
  sum = sum + subgroupShuffleXor(sum, 1u);
  sum = sum + subgroupShuffleXor(sum, 2u);
  sum = sum + subgroupShuffleXor(sum, 4u);

  if (lane == 0u && row < params.n_rows) {
    output[row] = sum;
  }
}
