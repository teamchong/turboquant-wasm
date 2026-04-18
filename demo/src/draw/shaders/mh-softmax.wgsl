// Multi-head softmax over a single query position. Each head softmaxes its own
// row of `data` (NUM_HEADS rows of length n_cols, contiguous as
// data[head * n_cols + pos]). Dispatch NUM_HEADS workgroups; each runs 256
// threads in parallel via strided loops + tree reductions.
//
// Replaces the old workgroup_size(1) implementation, which was a sequential
// loop in a single GPU thread per head — for a 2400-position cache that was
// burning ~20 ms/token doing nothing while every other lane sat idle.

struct Params {
  n_cols: u32,        // packed stride (usually the cache length)
  window_start: u32,  // positions < ws are masked to -inf (sliding window floor)
  query_pos: u32,     // shared query position; positions > query_pos are masked
  _p: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> s_reduce: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn mh_softmax(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let head = wid.x;
  let tid = lid.x;
  let n = params.n_cols;
  let row_start = head * n;
  let max_pos = params.query_pos + 1u;
  let win_start = params.window_start;

  // Phase 1: causal + sliding-window mask. Each thread strides across `n`,
  // setting out-of-window positions to -inf. Operates directly on `data`
  // (no shared scratch) since positions are independent.
  var i = tid;
  while (i < n) {
    if (i >= max_pos || i < win_start) {
      data[row_start + i] = -1e30;
    }
    i += WG_SIZE;
  }
  workgroupBarrier();

  // Phase 2: parallel max for numerical stability. Subgroup butterfly
  // reduction — max is order-invariant so no FP-reorder risk (unlike
  // phase 3's sum which stays as a shared-memory tree reduction below).
  var local_max = -1e30f;
  i = tid;
  while (i < n) {
    let v = data[row_start + i];
    if (v > local_max) { local_max = v; }
    i += WG_SIZE;
  }
  // Intra-subgroup butterfly (5 strides for 32-wide subgroup).
  local_max = max(local_max, subgroupShuffleXor(local_max, 1u));
  local_max = max(local_max, subgroupShuffleXor(local_max, 2u));
  local_max = max(local_max, subgroupShuffleXor(local_max, 4u));
  local_max = max(local_max, subgroupShuffleXor(local_max, 8u));
  local_max = max(local_max, subgroupShuffleXor(local_max, 16u));
  if ((tid & 31u) == 0u) { s_reduce[tid >> 5u] = local_max; }
  workgroupBarrier();
  // Cross-subgroup butterfly over 8 stashed maxes. Lanes ≥ 8 carry -inf;
  // only lane 0's result is read. Called at top level for uniform flow.
  var cross_max = -1e30f;
  if (tid < 8u) { cross_max = s_reduce[tid]; }
  cross_max = max(cross_max, subgroupShuffleXor(cross_max, 1u));
  cross_max = max(cross_max, subgroupShuffleXor(cross_max, 2u));
  cross_max = max(cross_max, subgroupShuffleXor(cross_max, 4u));
  if (tid == 0u) { s_reduce[0] = cross_max; }
  workgroupBarrier();
  let max_val = s_reduce[0];

  // Phase 3: parallel exp + per-thread sum.
  var local_sum = 0.0f;
  i = tid;
  while (i < n) {
    let e = exp(data[row_start + i] - max_val);
    data[row_start + i] = e;
    local_sum += e;
    i += WG_SIZE;
  }
  s_reduce[tid] = local_sum;
  workgroupBarrier();

  var stride = WG_SIZE / 2u;
  while (stride > 0u) {
    if (tid < stride) {
      s_reduce[tid] += s_reduce[tid + stride];
    }
    stride /= 2u;
    workgroupBarrier();
  }
  let total = s_reduce[0];

  // Phase 4: parallel normalize.
  let inv = 1.0f / total;
  i = tid;
  while (i < n) {
    data[row_start + i] = data[row_start + i] * inv;
    i += WG_SIZE;
  }
}
