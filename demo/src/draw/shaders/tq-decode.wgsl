// TQ Decode: reconstruct an f32 vector from polar + QJL compressed format.
// Inverse of tq-encode.wgsl. Outputs in original (unrotated) space.
//
// decoded = R^T * (polar_approx + √(π/2)/D * R^T_proj * (γ * sign_vec))

const SQRT_PI_OVER_2: f32 = 1.2533141373155;

const BUCKET_COS = array<f32, 8>(
  -0.9009688679, -0.6234898019, 0.0, 0.6234898019,
   0.9009688679,  0.9009688679, 0.6234898019, 0.0
);
const BUCKET_SIN = array<f32, 8>(
  -0.4338837391,  0.7818314825,  1.0, 0.7818314825,
  -0.4338837391, -0.4338837391, -0.7818314825, -1.0
);

struct DecodeParams {
  dim: u32,
  num_vectors: u32,
  polar_words_per_pos: u32,
  qjl_words_per_pos: u32,
  read_pos: u32,          // starting position in cache buffers
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: DecodeParams;
@group(0) @binding(1) var<storage, read> polar_in: array<u32>;
@group(0) @binding(2) var<storage, read> qjl_in: array<u32>;
@group(0) @binding(3) var<storage, read> max_r_in: array<f32>;
@group(0) @binding(4) var<storage, read> gamma_in: array<f32>;
@group(0) @binding(5) var<storage, read> rotation_matrix: array<f32>; // [dim * dim]
@group(0) @binding(6) var<storage, read_write> output: array<f32>;    // [num_vectors * dim]

fn unpack_polar_pair(base: u32, pair_idx: u32) -> vec2u {
  let bit_pos = pair_idx * 7u;
  let word_idx = base + bit_pos / 32u;
  let bit_off = bit_pos % 32u;
  var combined: u32;
  if (bit_off + 7u <= 32u) {
    combined = (polar_in[word_idx] >> bit_off) & 0x7Fu;
  } else {
    combined = ((polar_in[word_idx] >> bit_off) | (polar_in[word_idx + 1u] << (32u - bit_off))) & 0x7Fu;
  }
  return vec2u(combined >> 3u, combined & 7u);
}

fn get_qjl_sign(base: u32, dim_idx: u32) -> f32 {
  let word_idx = base + dim_idx / 32u;
  let bit_off = dim_idx % 32u;
  let bit = (qjl_in[word_idx] >> bit_off) & 1u;
  return select(-1.0, 1.0, bit == 1u);
}

@compute @workgroup_size(1)
fn decode(@builtin(global_invocation_id) gid: vec3u) {
  let vec_idx = gid.x;
  if (vec_idx >= params.num_vectors) { return; }

  let dim = params.dim;
  let num_pairs = dim / 2u;
  let cache_idx = params.read_pos + vec_idx;
  let polar_base = cache_idx * params.polar_words_per_pos;
  let qjl_base = cache_idx * params.qjl_words_per_pos;
  let max_r = max_r_in[cache_idx];
  let gamma = gamma_in[cache_idx];
  let out_offset = vec_idx * dim;

  // Step 1: Reconstruct polar approximation in rotated space
  var rotated: array<f32, 512>; // max dim 512
  for (var i = 0u; i < num_pairs; i++) {
    let pair = unpack_polar_pair(polar_base, i);
    let r_scaled = f32(pair.x) / 15.0 * max_r;
    let bucket = pair.y;
    rotated[i * 2u] = r_scaled * BUCKET_COS[bucket];
    rotated[i * 2u + 1u] = r_scaled * BUCKET_SIN[bucket];
  }

  // Step 2: QJL residual reconstruction
  // qjl_rotated[d] = Σ_k R[k, d] * (γ * sign[k])  (R^T applied)
  let qjl_scale = SQRT_PI_OVER_2 / f32(dim);
  for (var d = 0u; d < dim; d++) {
    var qjl_contrib = 0.0f;
    for (var k = 0u; k < dim; k++) {
      let sign = get_qjl_sign(qjl_base, k);
      qjl_contrib += rotation_matrix[k * dim + d] * sign;
    }
    rotated[d] += gamma * qjl_scale * qjl_contrib;
  }

  // Step 3: Inverse rotation R^T to original space
  for (var d = 0u; d < dim; d++) {
    var sum = 0.0f;
    for (var k = 0u; k < dim; k++) {
      // R^T[d, k] = R[k, d]
      sum += rotation_matrix[k * dim + d] * rotated[k];
    }
    output[out_offset + d] = sum;
  }
}
