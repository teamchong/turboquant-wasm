// TurboQuant WebGPU compute shader — batch dot product on compressed vectors.
//
// One workgroup (64 threads) per compressed vector.
// Each thread processes a subset of polar pairs and QJL bits,
// then a tree reduction produces the final score.
//
// Polar: 7 bits per pair (4 radius + 3 angle), LSB-first.
//   score_polar = Σ (q[2i]*dx + q[2i+1]*dy) where (dx,dy) = lut[bits] * max_r
//
// QJL: 1 bit per dimension, LSB-first.
//   score_qjl = (2 * Σ q[i] where bit=1 - q_sum) * sqrt(π/2) / dim * gamma

struct Config {
  num_vectors: u32,
  dim: u32,
  polar_u32s_per_vec: u32,
  qjl_u32s_per_vec: u32,
  q_sum: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

// Database (uploaded once)
@group(0) @binding(0) var<storage, read> polar_data: array<u32>;
@group(0) @binding(1) var<storage, read> qjl_data: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<vec2<f32>>;  // (max_r, gamma) per vector

// Per-query (updated each search)
@group(1) @binding(0) var<storage, read> query: array<f32>;  // rotated query, dim floats
@group(1) @binding(1) var<uniform> config: Config;
@group(1) @binding(2) var<storage, read_write> scores: array<f32>;

// Static LUT (128 entries of vec2<f32>)
@group(2) @binding(0) var<storage, read> polar_lut: array<vec2<f32>, 128>;

const WG_SIZE: u32 = 64u;
const SQRT_PI_OVER_2: f32 = 1.2533141;

var<workgroup> polar_partial: array<f32, 64>;
var<workgroup> qjl_partial: array<f32, 64>;

// Extract 7 bits from the polar bitstream at the given bit position.
// polar_data is u32-packed, LSB-first.
fn extract7(polar_offset: u32, bit_pos: u32) -> u32 {
  let word_idx = polar_offset + bit_pos / 32u;
  let bit_off = bit_pos % 32u;
  let w0 = polar_data[word_idx];
  if (bit_off <= 25u) {
    return (w0 >> bit_off) & 0x7Fu;
  }
  // 7 bits span two u32 words
  let w1 = polar_data[word_idx + 1u];
  return ((w0 >> bit_off) | (w1 << (32u - bit_off))) & 0x7Fu;
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let vec_id = wid.x;
  let tid = lid.x;

  if (vec_id >= config.num_vectors) {
    return;
  }

  let dim = config.dim;
  let p = params[vec_id];
  let max_r = p.x;
  let gamma = p.y;

  let num_pairs = dim / 2u;
  let polar_offset = vec_id * config.polar_u32s_per_vec;
  let qjl_offset = vec_id * config.qjl_u32s_per_vec;

  // Each thread handles ceil(num_pairs / WG_SIZE) polar pairs
  var p_sum: f32 = 0.0;
  var pair_idx = tid;
  while (pair_idx < num_pairs) {
    let bit_pos = pair_idx * 7u;
    let combined = extract7(polar_offset, bit_pos);
    let lut_val = polar_lut[combined];
    let dx = lut_val.x * max_r;
    let dy = lut_val.y * max_r;

    let d = pair_idx * 2u;
    p_sum += query[d] * dx + query[d + 1u] * dy;

    pair_idx += WG_SIZE;
  }

  // Each thread handles ceil(dim / WG_SIZE) QJL bits
  // Fast path: accumulate q[i] where bit=1
  var pos_sum: f32 = 0.0;
  var d_idx = tid;
  while (d_idx < dim) {
    let word_idx = qjl_offset + d_idx / 32u;
    let bit_off = d_idx % 32u;
    let bit = (qjl_data[word_idx] >> bit_off) & 1u;
    pos_sum += query[d_idx] * f32(bit);

    d_idx += WG_SIZE;
  }

  // Store partial sums for reduction
  polar_partial[tid] = p_sum;
  qjl_partial[tid] = pos_sum;
  workgroupBarrier();

  // Tree reduction (log2(64) = 6 steps)
  for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      polar_partial[tid] += polar_partial[tid + stride];
      qjl_partial[tid] += qjl_partial[tid + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the final score
  if (tid == 0u) {
    let polar_total = polar_partial[0];
    let pos_total = qjl_partial[0];
    let qjl_scale = SQRT_PI_OVER_2 / f32(dim) * gamma;
    let qjl_total = (2.0 * pos_total - config.q_sum) * qjl_scale;
    scores[vec_id] = polar_total + qjl_total;
  }
}
