// TQ Weighted Sum — Phase 2: output = polar_in + R^T × qjl_sum.
// Uses FWHT: R^T = D × H. One workgroup per head applies FWHT to
// the per-head qjl_sum, then adds polar_in.

struct Params {
  dim: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> polar_in: array<f32>;
@group(0) @binding(2) var<storage, read> qjl_sum_in: array<f32>;
@group(0) @binding(3) var<storage, read> signs: array<i32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> sh_qjl: array<f32, 512>;

@compute @workgroup_size(256)
fn weighted_sum_p2(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let tid = lid.x;
  let head = wid.x;
  let dim = params.dim;
  let qjl_base = head * dim;
  let out_base = head * dim;

  // Load qjl_sum into shared memory.
  var i = tid;
  while (i < dim) {
    sh_qjl[i] = qjl_sum_in[qjl_base + i];
    i += 256u;
  }
  workgroupBarrier();

  // FWHT butterfly stages on sh_qjl (computes H × qjl_sum).
  var stride = 1u;
  while (stride < dim) {
    i = tid;
    while (i < dim) {
      if ((i & stride) == 0u) {
        let j = i | stride;
        let a = sh_qjl[i];
        let b = sh_qjl[j];
        sh_qjl[i] = a + b;
        sh_qjl[j] = a - b;
      }
      i += 256u;
    }
    stride *= 2u;
    workgroupBarrier();
  }

  // R^T = D × H. Scale and apply D signs, then add polar_in.
  let scale = 1.0 / sqrt(f32(dim));
  i = tid;
  while (i < dim) {
    let qjl_rotated = sh_qjl[i] * scale * f32(signs[i]);
    output[out_base + i] = polar_in[out_base + i] + qjl_rotated;
    i += 256u;
  }
}
