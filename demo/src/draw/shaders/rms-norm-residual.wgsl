// Fused RMSNorm + Residual Add: output = residual + input * rsqrt(mean(x²) + eps) * weight
// Combines post-attention-norm + residual, and post-FFN-norm + residual into one pass.
// Same workgroup structure as rms-norm.wgsl but adds residual in the normalize phase.
// Weight already has Gemma3's +1 baked in by convert_hf_to_gguf.py.

struct Params {
  n: u32,      // row width (hidden_size)
  eps: f32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;     // value to normalize
@group(0) @binding(2) var<storage, read> weight: array<f32>;    // norm weight
@group(0) @binding(3) var<storage, read> residual: array<f32>;  // skip connection
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm_residual(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wid.x;
  let tid = lid.x;
  let n = params.n;
  let row_offset = row * n;

  // Phase 1: Sum of squares
  var sum_sq = 0.0f;
  var col = tid;
  while (col < n) {
    let v = input[row_offset + col];
    sum_sq += v * v;
    col += 256u;
  }

  // Phase 2: Parallel reduction
  sdata[tid] = sum_sq;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    stride /= 2u;
    workgroupBarrier();
  }
  let rms = 1.0 / sqrt(sdata[0] / f32(n) + params.eps);

  // Phase 3: Normalize + add residual
  col = tid;
  while (col < n) {
    let idx = row_offset + col;
    output[idx] = residual[idx] + input[idx] * rms * weight[col];
    col += 256u;
  }
}
