// TQ Attention: compute Q @ compressed_K scores for each (head, position).
// Each thread handles one (head, pos) and writes one score.
//
// Layout:
//   dispatch X = ceil(num_positions / 256)
//   dispatch Y = num_heads
//   workgroup_size X = 256, each thread = one position slot
//
// Hot-path optimization: the 256 threads of a workgroup all share the same
// head (wid.y), so they all read the SAME `rotated_q` slice. We cooperatively
// preload that slice into shared memory once, then the inner polar + QJL
// loops fetch Q values from shared memory instead of global memory — roughly
// dim-many saved global loads per thread per position.

const SQRT_PI_OVER_2: f32 = 1.2533141373155;

/*@POLAR_CONFIG@*/

struct Params {
  dim: u32,                 // head dimension (must be even)
  num_positions: u32,       // current cache length
  polar_words_per_pos: u32, // iteration count for polar pairs (not stride)
  qjl_words_per_pos: u32,   // iteration count for qjl words (not stride)
  window_start: u32,        // skip positions before this (sliding window)
  num_heads: u32,           // dispatch Y = num_heads, wid.y = head index
  pos_stride: u32,          // position-slot stride for [word][pos] K cache layout
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rotated_q: array<f32>;       // [num_heads * dim]
@group(0) @binding(2) var<storage, read> k_polar: array<u32>;
@group(0) @binding(3) var<storage, read> k_qjl: array<u32>;
@group(0) @binding(4) var<storage, read> k_max_r: array<f32>;
@group(0) @binding(5) var<storage, read> k_gamma: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;    // [num_heads * num_positions]

const WG_SIZE: u32 = 256u;
var<workgroup> s_rotated_q: array<f32, 512>;  // max dim=512 (global layers)

@compute @workgroup_size(WG_SIZE)
fn compute_scores(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let tid = lid.x;
  let head = wid.y;
  let dim = params.dim;
  let num_pairs = dim / 2u;
  let q_off = head * dim;
  let score_offset = head * params.num_positions;

  // Cooperatively load this head's rotated_q slice into shared memory.
  // Each thread loads dim/WG_SIZE consecutive values.
  var i = tid;
  while (i < dim) {
    s_rotated_q[i] = rotated_q[q_off + i];
    i += WG_SIZE;
  }
  workgroupBarrier();

  // Dispatch is sized to cover only (num_positions - window_start) slots so
  // out-of-window workgroups never launch — saves ~85% of WG dispatches on
  // sliding-window layers. gid.x therefore indexes the in-window suffix;
  // add window_start to get the absolute position.
  let pos = gid.x + params.window_start;
  if (pos >= params.num_positions) { return; }

  let max_r = k_max_r[pos];
  let gamma = k_gamma[pos];
  let pos_stride = params.pos_stride;

  // Polar dot: iterate pairs, unpack (r_q, bucket) from packed storage, add
  // to running polar_sum. Packed K is laid out [word][pos] so different
  // warp threads (which handle different `pos`) read adjacent u32s for the
  // same polar word — a single coalesced transaction instead of 32.
  var polar_sum = 0.0f;
  for (var p = 0u; p < num_pairs; p++) {
    let bit_pos = p * POLAR_PAIR_BITS;
    let word_off = bit_pos / 32u;
    let b_off = bit_pos % 32u;
    let w_idx = word_off * pos_stride + pos;
    var comb: u32;
    if (b_off + POLAR_PAIR_BITS <= 32u) {
      comb = (k_polar[w_idx] >> b_off) & POLAR_PAIR_MASK;
    } else {
      comb = ((k_polar[w_idx] >> b_off) | (k_polar[w_idx + pos_stride] << (32u - b_off))) & POLAR_PAIR_MASK;
    }
    let r_scaled = f32(comb >> POLAR_ANGLE_BITS) / POLAR_RADIUS_LEVELS_F * max_r;
    let bucket = comb & POLAR_ANGLE_MASK;
    let qx = s_rotated_q[p * 2u];
    let qy = s_rotated_q[p * 2u + 1u];
    polar_sum += qx * r_scaled * BUCKET_COS[bucket]
               + qy * r_scaled * BUCKET_SIN[bucket];
  }

  // QJL dot: K is always 1-bit (K_POLAR_CONFIG.qjlBits = 1). Each bit of
  // k_qjl picks the sign of q[d]: bit=1 → +q[d], bit=0 → -q[d].
  var qjl_sum = 0.0f;
  let qjl_words = dim / 32u;
  for (var w = 0u; w < qjl_words; w++) {
    let k_word = k_qjl[w * pos_stride + pos];
    let base_d = w * 32u;
    for (var b = 0u; b < 32u; b++) {
      let bit = (k_word >> b) & 1u;
      qjl_sum += s_rotated_q[base_d + b] * f32(i32(bit) * 2 - 1);
    }
  }

  // gamma already bakes in the optimal per-vector least-squares scale
  // (see tq-encode Phase 4), so no sqrt(pi/2)/dim factor here.
  scores[score_offset + pos] = polar_sum + qjl_sum * gamma;
}
