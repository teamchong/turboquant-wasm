// Multi-head softmax over a single query position. Each head softmaxes its own
// row of `data` (NUM_HEADS rows of length n_cols, contiguous as
// data[head * n_cols + pos]). Dispatch NUM_HEADS workgroups; each runs 256
// threads in parallel via strided loops + tree reductions.
//
// Replaces the old workgroup_size(1) implementation, which was a sequential
// loop in a single GPU thread per head — for a 2400-position cache that was
// burning ~20 ms/token doing nothing while every other lane sat idle.

enable subgroups;

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
  // Hi-Z loop bound: phases 2-4 only need positions in [win_start, scan_end).
  // Prior phase 1 wrote -1e30 to out-of-window slots so downstream reads
  // were harmless; with tighter bounds those slots are never read.
  // Sliding-window layers (28/35) have window=512 but n grows to 3000+,
  // so the old 0..n scan did ~6× the needed work here.
  let scan_end = min(max_pos, n);

  // Phase 2: parallel max for numerical stability. Subgroup butterfly
  // reduction — max is order-invariant so no FP-reorder risk (unlike
  // phase 3's sum which stays as a shared-memory tree reduction below).
  var local_max = -1e30f;
  var i = win_start + tid;
  while (i < scan_end) {
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

  // Phase 3: parallel exp + per-thread sum. Same window-bounded scan.
  var local_sum = 0.0f;
  i = win_start + tid;
  while (i < scan_end) {
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

  // Phase 4: parallel normalize. Window-bounded so out-of-window slots
  // keep stale values (never read by wsum_p1's compact-list path below).
  let inv = 1.0f / total;
  i = win_start + tid;
  while (i < scan_end) {
    data[row_start + i] = data[row_start + i] * inv;
    i += WG_SIZE;
  }
}
