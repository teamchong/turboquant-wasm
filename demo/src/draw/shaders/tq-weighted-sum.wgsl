// TQ Weighted Sum: compute attention_weights @ compressed_V directly on compressed data.
// This is the second half of attention: softmax(Q@K^T) @ V.
//
// Operates on compressed V without decompressing any individual vector.
// Accumulates the weighted polar reconstruction + weighted QJL signs,
// then applies inverse rotation to produce the final output in original space.
//
// rotated_output = Σ(w[i] * polar_decoded[i]) + √(π/2)/D * R^T * Σ(w[i] * γ[i] * sign_vec[i])
// output = R^T * rotated_output

const SQRT_PI_OVER_2: f32 = 1.2533141373155;

const BUCKET_COS = array<f32, 8>(
  -0.9009688679, -0.6234898019, 0.0, 0.6234898019,
   0.9009688679,  0.9009688679, 0.6234898019, 0.0
);
const BUCKET_SIN = array<f32, 8>(
  -0.4338837391,  0.7818314825,  1.0, 0.7818314825,
  -0.4338837391, -0.4338837391, -0.7818314825, -1.0
);

struct Params {
  dim: u32,
  num_positions: u32,
  polar_words_per_pos: u32,
  qjl_words_per_pos: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;         // [num_positions] softmax scores
@group(0) @binding(2) var<storage, read> v_polar: array<u32>;         // compressed V polar data
@group(0) @binding(3) var<storage, read> v_qjl: array<u32>;          // compressed V QJL bits
@group(0) @binding(4) var<storage, read> v_max_r: array<f32>;        // [num_positions]
@group(0) @binding(5) var<storage, read> v_gamma: array<f32>;        // [num_positions]
@group(0) @binding(6) var<storage, read> rotation_matrix: array<f32>; // [dim * dim] for inverse rotate
@group(0) @binding(7) var<storage, read_write> output: array<f32>;   // [dim] final output

fn unpack_polar_pair(pos_offset: u32, pair_idx: u32) -> vec2u {
  let bit_pos = pair_idx * 7u;
  let word_idx = pos_offset + bit_pos / 32u;
  let bit_off = bit_pos % 32u;
  var combined: u32;
  if (bit_off + 7u <= 32u) {
    combined = (v_polar[word_idx] >> bit_off) & 0x7Fu;
  } else {
    combined = ((v_polar[word_idx] >> bit_off) | (v_polar[word_idx + 1u] << (32u - bit_off))) & 0x7Fu;
  }
  return vec2u(combined >> 3u, combined & 7u);
}

fn get_qjl_sign(pos_offset: u32, dim_idx: u32) -> f32 {
  let word_idx = pos_offset + dim_idx / 32u;
  let bit_off = dim_idx % 32u;
  let bit = (v_qjl[word_idx] >> bit_off) & 1u;
  return select(-1.0, 1.0, bit == 1u);
}

// Each workgroup computes one output dimension.
// Thread 0 accumulates across all positions (sequential, but positions are few during generation).
@compute @workgroup_size(1)
fn weighted_sum(@builtin(global_invocation_id) gid: vec3u) {
  let out_dim = gid.x;
  let dim = params.dim;
  if (out_dim >= dim) { return; }

  let num_pairs = dim / 2u;
  let num_pos = params.num_positions;

  // Accumulate weighted polar + QJL in rotated space for this dimension
  var rotated_sum = 0.0f;

  // Polar contribution: for the pair containing this dimension
  let pair_idx = out_dim / 2u;
  let is_cos_component = (out_dim % 2u) == 0u;

  var polar_accum = 0.0f;
  for (var pos = 0u; pos < num_pos; pos++) {
    let w = weights[pos];
    if (abs(w) < 1e-8) { continue; }

    let polar_offset = pos * params.polar_words_per_pos;
    let max_r = v_max_r[pos];

    // This dimension's polar pair
    let pair = unpack_polar_pair(polar_offset, pair_idx);
    let r_norm = f32(pair.x) / 15.0;
    let r_scaled = r_norm * max_r;
    let bucket = pair.y;

    if (is_cos_component) {
      polar_accum += w * r_scaled * BUCKET_COS[bucket];
    } else {
      polar_accum += w * r_scaled * BUCKET_SIN[bucket];
    }
  }

  // QJL contribution: Σ(w[i] * γ[i] * sign[i, out_dim])
  // This gives us the weighted QJL component in the projected (doubly-rotated) space.
  // We need R^T to bring it back to rotated space.
  var qjl_projected_sum = 0.0f;
  for (var pos = 0u; pos < num_pos; pos++) {
    let w = weights[pos];
    if (abs(w) < 1e-8) { continue; }

    let qjl_offset = pos * params.qjl_words_per_pos;
    let gamma = v_gamma[pos];
    let sign = get_qjl_sign(qjl_offset, out_dim);
    qjl_projected_sum += w * gamma * sign;
  }

  // The polar_accum is already in rotated space for this dimension.
  // The qjl_projected_sum is in doubly-rotated space; apply R^T to get to rotated space.
  // But we need the FULL vector to apply R^T, and we only have one dimension here.
  // Instead, accumulate per-dimension and let the inverse rotation handle it.

  // For the QJL part, we accumulate the weighted-sign vector across all dims,
  // then apply R^T as a separate step. Store intermediate.
  // Since each thread handles one output dim, we compute R^T row by row:
  // qjl_rotated[out_dim] = Σ_k R^T[out_dim, k] * qjl_projected[k]
  //                       = Σ_k R[k, out_dim] * qjl_projected[k]

  // Recompute qjl_projected for ALL dims to apply R^T for this output dim
  var qjl_rotated = 0.0f;
  for (var k = 0u; k < dim; k++) {
    // qjl_projected[k] = Σ_pos(w[pos] * γ[pos] * sign[pos, k])
    var proj_k = 0.0f;
    for (var pos = 0u; pos < num_pos; pos++) {
      let w = weights[pos];
      if (abs(w) < 1e-8) { continue; }
      let qjl_offset = pos * params.qjl_words_per_pos;
      let gamma = v_gamma[pos];
      let sign = get_qjl_sign(qjl_offset, k);
      proj_k += w * gamma * sign;
    }
    // R^T[out_dim, k] = R[k, out_dim]
    qjl_rotated += rotation_matrix[k * dim + out_dim] * proj_k;
  }
  let qjl_scale = SQRT_PI_OVER_2 / f32(dim);
  let rotated_val = polar_accum + qjl_rotated * qjl_scale;

  // Final step: inverse rotation R^T to go from rotated space to original space.
  // output[out_dim] = Σ_k R^T[out_dim, k] * rotated[k]
  // But we only have rotated[out_dim], not the full rotated vector.
  // Each thread computes one element of the rotated vector; we need a reduction.
  // Store the rotated vector first, then do the matrix multiply.
  output[out_dim] = rotated_val;
}

// Second pass: inverse rotation. output_final = R^T * output_rotated.
// Called after weighted_sum writes the rotated vector.
@compute @workgroup_size(1)
fn inverse_rotate(@builtin(global_invocation_id) gid: vec3u) {
  let out_dim = gid.x;
  let dim = params.dim;
  if (out_dim >= dim) { return; }

  // Read the rotated vector (written by weighted_sum)
  // We need a separate input/output to avoid read-write hazard.
  // The caller should ping-pong buffers or use a barrier.
  var sum = 0.0f;
  for (var k = 0u; k < dim; k++) {
    // R^T[out_dim, k] = R[k, out_dim]
    sum += rotation_matrix[k * dim + out_dim] * output[k];
  }
  // Write back (caller must ensure weighted_sum is complete before dispatching this)
  output[out_dim] = sum;
}
