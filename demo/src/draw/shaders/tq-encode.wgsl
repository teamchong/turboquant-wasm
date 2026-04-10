// TQ Encode: compress a K or V vector into polar + QJL format on GPU.
// Input:  f32 vector [dim] (already rotated by caller)
// Output: polar pairs (packed 7-bit), QJL sign bits (packed), max_r, gamma
//
// Polar: pairs (x[2i], x[2i+1]) → (4-bit radius, 3-bit angle) = 7 bits/pair
// QJL:   project residual through rotation matrix, take 1 sign bit per dimension

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
const THETA_LEVELS: f32 = 7.0;

// Direction vectors for 8 angle buckets
const DIR_COS = array<f32, 8>(
  -0.9009688679, -0.6234898019, 0.0, 0.6234898019,
   0.9009688679,  0.9009688679, 0.6234898019, 0.0
);
const DIR_SIN = array<f32, 8>(
  -0.4338837391,  0.7818314825,  1.0, 0.7818314825,
  -0.4338837391, -0.4338837391, -0.7818314825, -1.0
);

struct EncodeParams {
  dim: u32,
  num_vectors: u32,         // vectors to encode this dispatch
  polar_words_per_pos: u32, // ceil(dim/2 * 7 / 32)
  qjl_words_per_pos: u32,  // dim / 32
  write_pos: u32,           // starting position index in the cache buffers
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: EncodeParams;
@group(0) @binding(1) var<storage, read> raw_input: array<f32>;       // [num_vectors * dim] unrotated
@group(0) @binding(2) var<storage, read> rotation_matrix: array<f32>; // [dim * dim] rotation + QJL projection
@group(0) @binding(3) var<storage, read_write> polar_out: array<u32>; // packed 7-bit pairs
@group(0) @binding(4) var<storage, read_write> qjl_out: array<u32>;  // packed sign bits
@group(0) @binding(5) var<storage, read_write> max_r_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> gamma_out: array<f32>;

fn nearest_bucket(x: f32, y: f32) -> u32 {
  var best = 0u;
  var best_dot = -1e30f;
  for (var i = 0u; i < 8u; i++) {
    let d = x * DIR_COS[i] + y * DIR_SIN[i];
    if (d > best_dot) { best_dot = d; best = i; }
  }
  return best;
}

// Pack a 7-bit value at the given bit position into the polar output buffer.
fn pack_polar_pair(base_word: u32, pair_idx: u32, value: u32) {
  let bit_pos = pair_idx * 7u;
  let word_idx = base_word + bit_pos / 32u;
  let bit_off = bit_pos % 32u;
  // Atomic OR to handle cross-word boundaries safely
  let lo = value << bit_off;
  polar_out[word_idx] = polar_out[word_idx] | lo;
  if (bit_off + 7u > 32u) {
    let hi = value >> (32u - bit_off);
    polar_out[word_idx + 1u] = polar_out[word_idx + 1u] | hi;
  }
}

@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) gid: vec3u) {
  let vec_idx = gid.x;
  if (vec_idx >= params.num_vectors) { return; }

  let dim = params.dim;
  let num_pairs = dim / 2u;
  let vec_offset = vec_idx * dim;

  // Step 0: Rotate input on GPU — rotated = R * raw_input
  var rotated: array<f32, 512>; // max dim 512
  for (var r = 0u; r < dim; r++) {
    var sum = 0.0f;
    for (var c = 0u; c < dim; c++) {
      sum += rotation_matrix[r * dim + c] * raw_input[vec_offset + c];
    }
    rotated[r] = sum;
  }

  // Step 1: Find max_r across all pairs
  var max_r = 0.0f;
  for (var i = 0u; i < num_pairs; i++) {
    let x = rotated[i * 2u];
    let y = rotated[i * 2u + 1u];
    let r = sqrt(x * x + y * y);
    max_r = max(max_r, r);
  }
  if (max_r < 1e-10) { max_r = 1.0; }
  let out_idx = params.write_pos + vec_idx;
  max_r_out[out_idx] = max_r;

  // Step 2: Polar encode each pair + compute residual
  let polar_base = out_idx * params.polar_words_per_pos;

  // Zero the polar output region
  for (var w = 0u; w < params.polar_words_per_pos; w++) {
    polar_out[polar_base + w] = 0u;
  }

  // We'll accumulate the residual in a second pass, but first encode polar
  for (var i = 0u; i < num_pairs; i++) {
    let x = rotated[i * 2u];
    let y = rotated[i * 2u + 1u];
    let r = sqrt(x * x + y * y);
    let r_q = u32(clamp(round(r / max_r * 15.0), 0.0, 15.0));
    let bucket = nearest_bucket(x, y);
    let combined = (r_q << 3u) | bucket;
    pack_polar_pair(polar_base, i, combined);
  }

  // Step 3: Compute residual (rotated - polar_decoded) and project for QJL
  let qjl_base = out_idx * params.qjl_words_per_pos;
  for (var w = 0u; w < params.qjl_words_per_pos; w++) {
    qjl_out[qjl_base + w] = 0u;
  }

  var residual_norm_sq = 0.0f;

  // For each projected dimension, compute R * residual and take sign
  for (var proj_dim = 0u; proj_dim < dim; proj_dim++) {
    var projected = 0.0f;
    for (var i = 0u; i < num_pairs; i++) {
      let x = rotated[i * 2u];
      let y = rotated[i * 2u + 1u];

      // Reconstruct polar approximation for this pair
      let bit_pos = i * 7u;
      let word_idx = polar_base + bit_pos / 32u;
      let bit_off = bit_pos % 32u;
      var combined: u32;
      if (bit_off + 7u <= 32u) {
        combined = (polar_out[word_idx] >> bit_off) & 0x7Fu;
      } else {
        combined = ((polar_out[word_idx] >> bit_off) | (polar_out[word_idx + 1u] << (32u - bit_off))) & 0x7Fu;
      }
      let r_q = combined >> 3u;
      let bucket = combined & 7u;
      let r_decoded = f32(r_q) / 15.0 * max_r;
      let dx_approx = r_decoded * DIR_COS[bucket];
      let dy_approx = r_decoded * DIR_SIN[bucket];

      let res_x = x - dx_approx;
      let res_y = y - dy_approx;

      // Accumulate residual norm (only on first proj_dim iteration)
      if (proj_dim == 0u) {
        residual_norm_sq += res_x * res_x + res_y * res_y;
      }

      // Project: R[proj_dim, 2i] * res_x + R[proj_dim, 2i+1] * res_y
      projected += rotation_matrix[proj_dim * dim + i * 2u] * res_x;
      projected += rotation_matrix[proj_dim * dim + i * 2u + 1u] * res_y;
    }

    // Take sign bit
    if (projected > 0.0) {
      let word_idx = qjl_base + proj_dim / 32u;
      qjl_out[word_idx] = qjl_out[word_idx] | (1u << (proj_dim % 32u));
    }
  }

  gamma_out[out_idx] = sqrt(residual_norm_sq);
}
