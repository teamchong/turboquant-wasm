// RMSNorm: output = x * rsqrt(mean(x²) + eps) * weight
//
// Gemma3 uses `x * (1 + learned_w)` but convert_hf_to_gguf.py bakes the +1 into the
// saved norm weights, so the GGUF weight = (learned_w + 1) and we use it directly.
// Adding another +1 here was a latent bug that doubled the offset.
//
// One workgroup per row. Threads cooperate on reduction via sdata memory.

struct Params {
  n: u32,      // row width (hidden_size)
  eps: f32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wid.x;
  let tid = lid.x;
  let n = params.n;
  let row_offset = row * n;

  // Phase 1: Each thread accumulates sum of squares for its elements
  var sum_sq = 0.0f;
  var col = tid;
  while (col < n) {
    let v = input[row_offset + col];
    sum_sq += v * v;
    col += 256u;
  }

  // Phase 2: Parallel reduction in sdata memory
  sdata[tid] = sum_sq;
  workgroupBarrier();

  var stride = 128u;
  while (stride > 0u) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    stride /= 2u;
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(n) + params.eps);

  // Phase 3: Apply normalization and weight
  col = tid;
  while (col < n) {
    output[row_offset + col] = input[row_offset + col] * rms * weight[col];
    col += 256u;
  }
}
