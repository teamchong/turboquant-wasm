// TQ Attention: compute Q @ compressed_K scores + softmax + weights @ compressed_V output.
// All operations read directly from compressed TQ data. No vector is ever decompressed.
//
// Polar dot: for each pair (r_q, bucket), score += q[2i]*r*cos(θ) + q[2i+1]*r*sin(θ)
// QJL dot:   for each dim, score += rotated_q[i] * sign(bit[i]) * γ * √(π/2) / D
// Weighted V sum: accumulate polar pairs + QJL signs weighted by softmax scores

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
const SQRT_PI_OVER_2: f32 = 1.2533141373155;
const THETA_LEVELS: f32 = 7.0;

// Precomputed cos/sin for 8 angle buckets: θ = bucket/7 * 2π - π
const BUCKET_COS = array<f32, 8>(
  -0.9009688679, -0.6234898019, 0.0, 0.6234898019,
   0.9009688679,  0.9009688679, 0.6234898019, 0.0
);
const BUCKET_SIN = array<f32, 8>(
  -0.4338837391,  0.7818314825,  1.0, 0.7818314825,
  -0.4338837391, -0.4338837391, -0.7818314825, -1.0
);

struct Params {
  dim: u32,               // head dimension (must be even)
  num_positions: u32,     // current cache length
  polar_words_per_pos: u32, // ceil(dim/2 * 7 / 32) u32s per position
  qjl_words_per_pos: u32,  // dim / 32 u32s per position
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rotated_q: array<f32>;       // [dim] pre-rotated query
@group(0) @binding(2) var<storage, read> k_polar: array<u32>;         // [num_positions * polar_words]
@group(0) @binding(3) var<storage, read> k_qjl: array<u32>;          // [num_positions * qjl_words]
@group(0) @binding(4) var<storage, read> k_max_r: array<f32>;        // [num_positions]
@group(0) @binding(5) var<storage, read> k_gamma: array<f32>;        // [num_positions]
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;   // [num_positions] output

// Extract a 7-bit polar pair from packed u32 buffer at the given bit position.
// Returns (radius_quantized: 0-15, angle_bucket: 0-7).
fn unpack_polar_pair(pos_offset: u32, pair_idx: u32) -> vec2u {
  let bit_pos = pair_idx * 7u;
  let word_idx = pos_offset + bit_pos / 32u;
  let bit_off = bit_pos % 32u;
  var combined: u32;
  if (bit_off + 7u <= 32u) {
    combined = (k_polar[word_idx] >> bit_off) & 0x7Fu;
  } else {
    combined = ((k_polar[word_idx] >> bit_off) | (k_polar[word_idx + 1u] << (32u - bit_off))) & 0x7Fu;
  }
  return vec2u(combined >> 3u, combined & 7u); // (r_q, bucket)
}

// Extract a single QJL sign bit.
fn get_qjl_sign(pos_offset: u32, dim_idx: u32) -> f32 {
  let word_idx = pos_offset + dim_idx / 32u;
  let bit_off = dim_idx % 32u;
  let bit = (k_qjl[word_idx] >> bit_off) & 1u;
  return select(-1.0, 1.0, bit == 1u);
}

// Compute dot product of rotated query with one compressed K position.
fn tq_dot(pos_idx: u32) -> f32 {
  let dim = params.dim;
  let num_pairs = dim / 2u;
  let max_r = k_max_r[pos_idx];
  let gamma = k_gamma[pos_idx];
  let polar_offset = pos_idx * params.polar_words_per_pos;
  let qjl_offset = pos_idx * params.qjl_words_per_pos;

  // Polar dot product: Σ q[2i]*r*cos(θ) + q[2i+1]*r*sin(θ)
  var polar_sum = 0.0f;
  for (var i = 0u; i < num_pairs; i++) {
    let pair = unpack_polar_pair(polar_offset, i);
    let r_norm = f32(pair.x) / 15.0;
    let r_scaled = r_norm * max_r;
    let bucket = pair.y;
    let dx = r_scaled * BUCKET_COS[bucket];
    let dy = r_scaled * BUCKET_SIN[bucket];
    polar_sum += rotated_q[i * 2u] * dx + rotated_q[i * 2u + 1u] * dy;
  }

  // QJL dot product: Σ rotated_q[i] * sign(bit[i]) * γ * √(π/2) / D
  var qjl_sum = 0.0f;
  for (var i = 0u; i < dim; i++) {
    qjl_sum += rotated_q[i] * get_qjl_sign(qjl_offset, i);
  }
  let qjl_scale = SQRT_PI_OVER_2 / f32(dim) * gamma;
  qjl_sum *= qjl_scale;

  return polar_sum + qjl_sum;
}

@compute @workgroup_size(256)
fn compute_scores(@builtin(global_invocation_id) gid: vec3u) {
  let pos = gid.x;
  if (pos >= params.num_positions) { return; }
  scores[pos] = tq_dot(pos);
}
