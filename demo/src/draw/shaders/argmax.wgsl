// GPU argmax: find the index of the maximum value in a f32 array.
// Two-pass reduction: each workgroup finds its local max, then workgroup 0 reduces across all.
// Output: result[0] = argmax index (as u32 reinterpreted in f32 buffer).

struct Params {
  count: u32,     // total number of elements
  n_groups: u32,  // number of workgroups in first pass
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>; // [n_groups * 2]: (index, value_bits) pairs

var<workgroup> smax: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

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

  smax[tid] = best_val;
  sidx[tid] = best_idx;
  workgroupBarrier();

  // Parallel reduction within workgroup
  var stride = 128u;
  while (stride > 0u) {
    if (tid < stride && smax[tid + stride] > smax[tid]) {
      smax[tid] = smax[tid + stride];
      sidx[tid] = sidx[tid + stride];
    }
    stride /= 2u;
    workgroupBarrier();
  }

  // Write workgroup result
  if (tid == 0u) {
    result[gid * 2u] = sidx[0];
    result[gid * 2u + 1u] = bitcast<u32>(smax[0]);
  }
}

@compute @workgroup_size(256)
fn argmax_pass2(
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let n = params.n_groups;

  // Load group results into shared memory
  if (tid < n) {
    sidx[tid] = result[tid * 2u];
    smax[tid] = bitcast<f32>(result[tid * 2u + 1u]);
  } else {
    smax[tid] = -1e30;
    sidx[tid] = 0u;
  }
  workgroupBarrier();

  // Reduce
  var stride = 128u;
  while (stride > 0u) {
    if (tid < stride && smax[tid + stride] > smax[tid]) {
      smax[tid] = smax[tid + stride];
      sidx[tid] = sidx[tid + stride];
    }
    stride /= 2u;
    workgroupBarrier();
  }

  // Final result: index in result[0]
  if (tid == 0u) {
    result[0] = sidx[0];
  }
}
