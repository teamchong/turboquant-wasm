// GPU argmax: find the index of the maximum value in a f32 array.
// Two-pass reduction: each workgroup finds its local max, then workgroup 0 reduces across all.
// Output: result[0] = argmax index (as u32 reinterpreted in f32 buffer).
//
// Reduction uses subgroupShuffleXor butterfly inside each 32-wide subgroup,
// then a cross-subgroup step through 8 shared slots. Max is order-invariant
// so there's no FP-reorder quality risk here (unlike sum reductions).

enable subgroups;

struct Params {
  count: u32,     // total number of elements
  n_groups: u32,  // number of workgroups in first pass
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>; // [n_groups * 2]: (index, value_bits) pairs

// Per-subgroup stash for the cross-subgroup phase. 8 slots = 8 subgroups
// (WG=256, subgroup_size=32).
var<workgroup> sg_val: array<f32, 8>;
var<workgroup> sg_idx: array<u32, 8>;

// Fused (val,idx) butterfly reduce: at each stride, compare with the
// XOR-paired lane; the winner's pair replaces this lane's. After the 5
// strides inside a 32-wide subgroup, every lane holds the subgroup max.
fn subgroup_max_pair(val: f32, idx: u32) -> vec2<u32> {
  var v = val;
  var i = idx;
  var o_v = subgroupShuffleXor(v, 1u);
  var o_i = subgroupShuffleXor(i, 1u);
  if (o_v > v) { v = o_v; i = o_i; }
  o_v = subgroupShuffleXor(v, 2u);
  o_i = subgroupShuffleXor(i, 2u);
  if (o_v > v) { v = o_v; i = o_i; }
  o_v = subgroupShuffleXor(v, 4u);
  o_i = subgroupShuffleXor(i, 4u);
  if (o_v > v) { v = o_v; i = o_i; }
  o_v = subgroupShuffleXor(v, 8u);
  o_i = subgroupShuffleXor(i, 8u);
  if (o_v > v) { v = o_v; i = o_i; }
  o_v = subgroupShuffleXor(v, 16u);
  o_i = subgroupShuffleXor(i, 16u);
  if (o_v > v) { v = o_v; i = o_i; }
  // Pack (val, idx) into vec2<u32>: x = bitcast(val), y = idx.
  return vec2<u32>(bitcast<u32>(v), i);
}

// Cross-subgroup: reduce 8 stashed (val,idx) pairs to one using 3
// additional butterflies. Lanes ≥ 8 contribute -inf; only lane 0's result
// is read. Must be called at the function top level (subgroup ops require
// subgroup-uniform control flow).
fn crossgroup_max_pair(tid: u32) -> vec2<u32> {
  var v = -1e30f;
  var i = 0u;
  if (tid < 8u) { v = sg_val[tid]; i = sg_idx[tid]; }
  var o_v = subgroupShuffleXor(v, 1u);
  var o_i = subgroupShuffleXor(i, 1u);
  if (o_v > v) { v = o_v; i = o_i; }
  o_v = subgroupShuffleXor(v, 2u);
  o_i = subgroupShuffleXor(i, 2u);
  if (o_v > v) { v = o_v; i = o_i; }
  o_v = subgroupShuffleXor(v, 4u);
  o_i = subgroupShuffleXor(i, 4u);
  if (o_v > v) { v = o_v; i = o_i; }
  return vec2<u32>(bitcast<u32>(v), i);
}

@compute @workgroup_size(256)
fn argmax_pass1(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let gid = wid.x;
  let count = params.count;

  // Each thread finds max in its stride
  var best_val = -1e30f;
  var best_idx = 0u;
  var i = gid * 256u + tid;
  while (i < count) {
    let v = data[i];
    if (v > best_val) { best_val = v; best_idx = i; }
    i += params.n_groups * 256u;
  }

  // Intra-subgroup butterfly reduce.
  let sg = subgroup_max_pair(best_val, best_idx);
  if ((tid & 31u) == 0u) {
    sg_val[tid >> 5u] = bitcast<f32>(sg.x);
    sg_idx[tid >> 5u] = sg.y;
  }
  workgroupBarrier();

  // Cross-subgroup butterfly (lanes beyond first 8 harmless).
  let wg_max = crossgroup_max_pair(tid);
  if (tid == 0u) {
    result[gid * 2u] = wg_max.y;
    result[gid * 2u + 1u] = wg_max.x;
  }
}

@compute @workgroup_size(256)
fn argmax_pass2(
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let n = params.n_groups;

  // Load group results
  var best_val = -1e30f;
  var best_idx = 0u;
  if (tid < n) {
    best_idx = result[tid * 2u];
    best_val = bitcast<f32>(result[tid * 2u + 1u]);
  }

  // Intra-subgroup reduce.
  let sg = subgroup_max_pair(best_val, best_idx);
  if ((tid & 31u) == 0u) {
    sg_val[tid >> 5u] = bitcast<f32>(sg.x);
    sg_idx[tid >> 5u] = sg.y;
  }
  workgroupBarrier();

  // Cross-subgroup reduce.
  let wg_max = crossgroup_max_pair(tid);
  if (tid == 0u) { result[0] = wg_max.y; }
}
