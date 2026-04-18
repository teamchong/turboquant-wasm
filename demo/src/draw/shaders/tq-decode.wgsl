// TQ Decode (debug): reconstruct a single vector from the compressed cache
// back into f32. Not used in the forward pass — solely for diagnostic probes
// that measure round-trip reconstruction error.
//
// Output = R^T * (polar_approx + gamma * QJL_level_vec)
// where polar_approx is the per-pair polar decode, gamma is the stored
// optimal scale, and QJL_level_vec is the 1- or 2-bit level sequence.

/*@POLAR_CONFIG@*/

struct DecodeParams {
  dim: u32,
  num_vectors: u32,
  polar_words_per_pos: u32,   // iteration count (unused for addressing)
  qjl_words_per_pos: u32,     // iteration count (unused for addressing)
  read_pos: u32,
  pos_stride: u32,            // position-slot stride for [word][pos] layout
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: DecodeParams;
@group(0) @binding(1) var<storage, read> polar_in: array<u32>;
@group(0) @binding(2) var<storage, read> qjl_in: array<u32>;
@group(0) @binding(3) var<storage, read> max_r_in: array<f32>;
@group(0) @binding(4) var<storage, read> gamma_in: array<f32>;
@group(0) @binding(5) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

var<workgroup> s_rotated: array<f32, 512>;  // holds polar approx in rotated space, then residual in rotated space
var<workgroup> s_levels: array<f32, 512>;   // QJL level vector before R^T applications

@compute @workgroup_size(256)
fn decode(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let vec_idx = wid.x;
  if (vec_idx >= params.num_vectors) { return; }

  let tid = lid.x;
  let dim = params.dim;
  let num_pairs = dim / 2u;
  let cache_idx = params.read_pos + vec_idx;
  let pos_stride = params.pos_stride;
  let max_r = max_r_in[cache_idx];
  let gamma = gamma_in[cache_idx];

  // Phase 1a: polar decode into s_rotated. V cache layout is [word][pos],
  // so word W at position P lives at `W * pos_stride + P`.
  var p = tid;
  while (p < num_pairs) {
    let bit_pos = p * POLAR_PAIR_BITS;
    let word_off = bit_pos / 32u;
    let b_off = bit_pos % 32u;
    let w_idx = word_off * pos_stride + cache_idx;
    var comb: u32;
    if (b_off + POLAR_PAIR_BITS <= 32u) {
      comb = (polar_in[w_idx] >> b_off) & POLAR_PAIR_MASK;
    } else {
      comb = ((polar_in[w_idx] >> b_off) | (polar_in[w_idx + pos_stride] << (32u - b_off))) & POLAR_PAIR_MASK;
    }
    let r_val = f32(comb >> POLAR_ANGLE_BITS) / POLAR_RADIUS_LEVELS_F * max_r;
    let bucket = comb & POLAR_ANGLE_MASK;
    s_rotated[p * 2u] = r_val * BUCKET_COS[bucket];
    s_rotated[p * 2u + 1u] = r_val * BUCKET_SIN[bucket];
    p += 256u;
  }

  // Phase 1b: unpack QJL level vector (level[k] ∈ QJL_LEVELS).
  var d = tid;
  while (d < dim) {
    let word_off = d / QJL_PER_WORD;
    let bit_off = (d % QJL_PER_WORD) * QJL_BITS;
    let word_idx = word_off * pos_stride + cache_idx;
    let idx = (qjl_in[word_idx] >> bit_off) & QJL_MASK;
    s_levels[d] = QJL_LEVELS[idx];
    d += 256u;
  }
  workgroupBarrier();

  // Phase 2: residual_rot = gamma * R^T @ level_vec. The encoder stored
  // sign(R @ residual_rot), so JL reconstruction of residual_rot uses R^T.
  // Add into s_rotated so it now holds (polar_rot + residual_rot) = V_rot.
  d = tid;
  while (d < dim) {
    var residual_rot = 0.0f;
    for (var k = 0u; k < dim; k++) {
      residual_rot += rotation_matrix[k * dim + d] * s_levels[k];
    }
    s_rotated[d] += gamma * residual_rot;
    d += 256u;
  }
  workgroupBarrier();

  // Phase 3: inverse rotate V_rot → V_original. output = R^T @ s_rotated.
  let out_offset = vec_idx * dim;
  d = tid;
  while (d < dim) {
    var sum = 0.0f;
    for (var k = 0u; k < dim; k++) {
      sum += rotation_matrix[k * dim + d] * s_rotated[k];
    }
    output[out_offset + d] = sum;
    d += 256u;
  }
}
