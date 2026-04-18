// Multi-head inverse rotation via FWHT.
// R^T = (H × D)^T = D × H. So: FWHT(x) × scale, then multiply by D.

struct Params { dim: u32, num_heads: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<i32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

var<workgroup> s_data: array<f32, 512>;

@compute @workgroup_size(256)
fn inverse_rotate(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let head = wid.x;
  let tid = lid.x;
  let dim = params.dim;
  let offset = head * dim;

  // Load input (no sign flip yet — D is applied AFTER H for R^T).
  var i = tid;
  while (i < dim) {
    s_data[i] = input_vec[offset + i];
    i += 256u;
  }
  workgroupBarrier();

  // FWHT butterfly stages.
  var stride = 1u;
  while (stride < dim) {
    i = tid;
    while (i < dim) {
      if ((i & stride) == 0u) {
        let j = i | stride;
        let a = s_data[i];
        let b = s_data[j];
        s_data[i] = a + b;
        s_data[j] = a - b;
      }
      i += 256u;
    }
    stride *= 2u;
    workgroupBarrier();
  }

  // Scale and apply D (sign flip AFTER H for inverse: R^T = D × H).
  let scale = 1.0 / sqrt(f32(dim));
  i = tid;
  while (i < dim) {
    output_vec[offset + i] = s_data[i] * scale * f32(signs[i]);
    i += 256u;
  }
}
