// TurboQuant KV attention — scaled dot-product attention on compressed KV cache.
//
// Dispatch: (dim, num_q_heads, 1). Each thread handles one output dimension
// for one query head. Uses online softmax to compute the weighted V sum in a
// single pass without storing all attention scores.
//
// This is the core of the TQ KV cache: Q @ compressed_K^T + softmax @ compressed_V,
// all on GPU, no CPU round-trip. K and V are never decompressed to full float32.

struct AttentionConfig {
  num_q_heads: u32,
  num_kv_heads: u32,
  num_kv_positions: u32,
  dim: u32,
  blob_u32s_per_vec: u32,
  header_u32s: u32,
  polar_byte_offset: u32,
  qjl_byte_offset: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> config: AttentionConfig;
@group(0) @binding(6) var<storage, read> polar_lut: array<vec2<f32>, 128>;

const SQRT_PI_OVER_2: f32 = 1.2533141;
const NEG_INF: f32 = -1e30;

fn read_cache_byte(cache: ptr<storage, array<u32>, read>, blob_offset: u32, byte_off: u32) -> u32 {
  let word_idx = blob_offset + byte_off / 4u;
  let shift = (byte_off % 4u) * 8u;
  return ((*cache)[word_idx] >> shift) & 0xFFu;
}

fn extract7_from(cache: ptr<storage, array<u32>, read>, blob_offset: u32, polar_start: u32, bit_pos: u32) -> u32 {
  let byte_idx = polar_start + bit_pos / 8u;
  let b0 = read_cache_byte(cache, blob_offset, byte_idx);
  let b1 = read_cache_byte(cache, blob_offset, byte_idx + 1u);
  let window = b0 | (b1 << 8u);
  return (window >> (bit_pos % 8u)) & 0x7Fu;
}

// TQ dot product: query @ compressed_vector
// Combines polar dot (7-bit LUT decode) + QJL dot (sign-bit fast path)
fn tq_dot(
  q_offset: u32,
  cache: ptr<storage, array<u32>, read>,
  blob_offset: u32,
  dim: u32,
) -> f32 {
  let max_r = bitcast<f32>((*cache)[blob_offset]);
  let gamma = bitcast<f32>((*cache)[blob_offset + 1u]);
  let num_pairs = dim / 2u;
  let polar_start = config.polar_byte_offset;
  let qjl_start = config.qjl_byte_offset;

  // Polar dot product: sum_i q[2i]*lut[c].x*max_r + q[2i+1]*lut[c].y*max_r
  var polar_sum: f32 = 0.0;
  for (var i: u32 = 0u; i < num_pairs; i += 1u) {
    let combined = extract7_from(cache, blob_offset, polar_start, i * 7u);
    let lv = polar_lut[combined];
    let d = i * 2u;
    polar_sum += query[q_offset + d] * lv.x * max_r + query[q_offset + d + 1u] * lv.y * max_r;
  }

  // QJL dot product: (2*pos_sum - q_sum) * sqrt(pi/2) / dim * gamma
  var pos_sum: f32 = 0.0;
  var q_sum: f32 = 0.0;
  for (var d: u32 = 0u; d < dim; d += 1u) {
    let byte_idx = qjl_start + d / 8u;
    let word_idx = blob_offset + byte_idx / 4u;
    let shift = (byte_idx % 4u) * 8u + (d % 8u);
    let bit = ((*cache)[word_idx] >> shift) & 1u;
    let qv = query[q_offset + d];
    pos_sum += qv * f32(bit);
    q_sum += qv;
  }
  let qjl_scale = SQRT_PI_OVER_2 / f32(dim) * gamma;

  return polar_sum + (2.0 * pos_sum - q_sum) * qjl_scale;
}

// Decode one dimension of a compressed vector (polar + QJL reconstruction)
fn tq_decode_dim(
  cache: ptr<storage, array<u32>, read>,
  blob_offset: u32,
  dim_idx: u32,
  dim: u32,
) -> f32 {
  let max_r = bitcast<f32>((*cache)[blob_offset]);
  let gamma = bitcast<f32>((*cache)[blob_offset + 1u]);
  let polar_start = config.polar_byte_offset;
  let qjl_start = config.qjl_byte_offset;

  // Polar: decode the pair containing this dimension
  let pair_idx = dim_idx / 2u;
  let combined = extract7_from(cache, blob_offset, polar_start, pair_idx * 7u);
  let lv = polar_lut[combined];
  let polar_val = select(lv.x, lv.y, dim_idx % 2u == 1u) * max_r;

  // QJL: decode sign bit for this dimension
  let byte_idx = qjl_start + dim_idx / 8u;
  let word_idx = blob_offset + byte_idx / 4u;
  let shift = (byte_idx % 4u) * 8u + (dim_idx % 8u);
  let bit = ((*cache)[word_idx] >> shift) & 1u;
  let sign = f32(bit) * 2.0 - 1.0;
  let qjl_val = sign * SQRT_PI_OVER_2 / f32(dim) * gamma;

  return polar_val + qjl_val;
}

// Main: dispatch (dim, num_q_heads, 1)
// Each thread computes output[head][dim_id] using online softmax over all
// KV positions. No intermediate score buffer needed.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dim_id = gid.x;
  let head_id = gid.y;

  if (head_id >= config.num_q_heads || dim_id >= config.dim) { return; }

  let kv_head = head_id / (config.num_q_heads / config.num_kv_heads);
  let q_offset = head_id * config.dim;
  let num_pos = config.num_kv_positions;
  let dim = config.dim;

  // Online softmax weighted sum:
  // Maintains running max, sum of exp(score - max), and weighted V sum.
  // Single pass over all KV positions — O(num_pos) time, O(1) memory.
  var running_max: f32 = NEG_INF;
  var running_sum: f32 = 0.0;
  var weighted_v: f32 = 0.0;

  for (var pos: u32 = 0u; pos < num_pos; pos += 1u) {
    let k_blob = pos * config.blob_u32s_per_vec;
    let raw_score = tq_dot(q_offset, &k_cache, k_blob, dim) * config.scale + mask[pos];

    // Online softmax update: when we see a new max, rescale the running totals
    let new_max = max(running_max, raw_score);
    let correction = exp(running_max - new_max);
    weighted_v *= correction;
    running_sum *= correction;

    let w = exp(raw_score - new_max);
    let v_blob = pos * config.blob_u32s_per_vec;
    weighted_v += w * tq_decode_dim(&v_cache, v_blob, dim_id, dim);
    running_sum += w;
    running_max = new_max;
  }

  output[head_id * dim + dim_id] = select(0.0, weighted_v / running_sum, running_sum > 0.0);
}
