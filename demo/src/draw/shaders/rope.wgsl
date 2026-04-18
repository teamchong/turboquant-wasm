// Rotary Position Embeddings (RoPE) — multi-head, batch-aware
//
// Gemma 4 uses the NEOX pairing convention (LLAMA_ROPE_TYPE_NEOX): within each
// head's `rope_dims`-sized rotation block, element i is paired with element
// i + rope_dims/2 (not with i+1). Only the first `rope_dims` entries rotate;
// any tail (rope_dims < dim) passes through unchanged.
//
// Gemma 4 variants (both rotate the FULL head dimension):
// - Sliding layers: θ=10000,   head_dim=256, rope_dims=256
// - Global  layers: θ=1000000, head_dim=512, rope_dims=512, PLUS proportional
//                   /YaRN scaling via `rope_freqs.weight` (ggml `freq_factors`,
//                   divides the per-pair theta_i).
//
// Data layout: [n_batch, n_heads, dim] contiguous.
// Dispatch: ceil(n_batch * n_heads * rope_dims / 2 / 256) workgroups in X.

struct Params {
  dim: u32,             // head dimension (256 or 512)
  position_start: u32,  // position of first token in batch
  theta: f32,           // base frequency (10000 or 1000000)
  rope_dims: u32,       // dims to apply RoPE to (dim for sliding, dim/4 for global)
  n_heads: u32,         // heads per batch item
  n_batch: u32,         // number of tokens in batch (1 for decode)
  use_freq_factors: u32,// 1 = divide theta_i by freq_factors[i] (global layers)
  _p0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
// Always bound so the pipeline layout stays fixed; sliding layers pass a dummy
// buffer of 1.0s and set use_freq_factors=0 so the read is a no-op.
@group(0) @binding(2) var<storage, read> freq_factors: array<f32>;

@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) gid: vec3u) {
  let half = params.rope_dims / 2u;
  let pairs_per_token = params.n_heads * half;
  let total_pairs = params.n_batch * pairs_per_token;
  if (gid.x >= total_pairs) { return; }

  let batch = gid.x / pairs_per_token;
  let within_token = gid.x % pairs_per_token;
  let head = within_token / half;
  let pair = within_token % half;

  let position = params.position_start + batch;
  // Frequency uses the pair's index within the rotation block; matches llama.cpp
  // ggml_rope_ext / neox convention freq_i = theta^(-2i / rope_dims) / ff_i.
  var base_freq = 1.0 / pow(params.theta, f32(2u * pair) / f32(params.rope_dims));
  if (params.use_freq_factors == 1u) {
    base_freq = base_freq / freq_factors[pair];
  }
  let angle = f32(position) * base_freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let head_base = batch * params.n_heads * params.dim + head * params.dim;
  let lo = head_base + pair;
  let hi = head_base + pair + half;
  let x0 = data[lo];
  let x1 = data[hi];
  data[lo] = x0 * cos_a - x1 * sin_a;
  data[hi] = x0 * sin_a + x1 * cos_a;
}
