// BF16 matrix-vector multiply: out = W @ x where W is BF16, x is F32.
// W stored as 2 bf16 values per u32 (little-endian). Used for PLE model_proj.
//
// Optimized: 8 threads per row cooperate on columns, shared memory for
// input vector, subgroupShuffleXor reduction. Replaces the naive
// 1-thread-per-row version that ran at 25% of bandwidth ceiling.

enable subgroups;

struct Params {
  n_rows: u32,
  n_cols: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

fn bf16_to_f32(bits: u32) -> f32 {
  return bitcast<f32>(bits << 16u);
}

var<workgroup> sact: array<f32, 1536>;

@compute @workgroup_size(WG_SIZE)
fn matmul_bf16(
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

  // Cooperative input load.
  var i = tid;
  while (i < n_cols) {
    sact[i] = input[i];
    i += WG_SIZE;
  }
  workgroupBarrier();

  var sum = 0.0f;
  if (row < params.n_rows) {
    let row_base_u32 = (row * n_cols) / 2u;
    // Each thread handles columns `lane, lane+8, lane+16, ...` stepping
    // by 2 (bf16 pair) per iteration. Adjacent lanes read adjacent u32
    // words → coalesced.
    var c = lane * 2u;
    while (c + 2u <= n_cols) {
      let packed = weights[row_base_u32 + c / 2u];
      let w0 = bf16_to_f32(packed & 0xFFFFu);
      let w1 = bf16_to_f32(packed >> 16u);
      sum += w0 * sact[c] + w1 * sact[c + 1u];
      c += THREADS_PER_ROW * 2u;
    }
    if (c < n_cols) {
      let packed = weights[row_base_u32 + c / 2u];
      sum += bf16_to_f32(packed & 0xFFFFu) * sact[c];
    }
  }

  sum = sum + subgroupShuffleXor(sum, 1u);
  sum = sum + subgroupShuffleXor(sum, 2u);
  sum = sum + subgroupShuffleXor(sum, 4u);

  if (lane == 0u && row < params.n_rows) {
    output[row] = sum;
  }
}
