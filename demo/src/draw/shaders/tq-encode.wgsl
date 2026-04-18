// TQ Encode: compress a K or V vector into polar + QJL format on GPU.
// 256 threads per vector. Both Phase 1 (rotation) and Phase 4a (QJL
// projection) use FWHT instead of explicit dim×dim matrix multiply.

/*@POLAR_CONFIG@*/

struct EncodeParams {
  dim: u32,
  num_vectors: u32,
  polar_words_per_pos: u32,
  qjl_words_per_pos: u32,
  write_pos: u32,
  pos_stride: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: EncodeParams;
@group(0) @binding(1) var<storage, read> raw_input: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<i32>;
@group(0) @binding(3) var<storage, read_write> polar_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> qjl_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> max_r_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> gamma_out: array<f32>;

fn nearest_bucket(x: f32, y: f32) -> u32 {
  var best = 0u;
  var best_dot = -1e30f;
  for (var i = 0u; i < POLAR_BUCKET_COUNT; i++) {
    let d = x * BUCKET_COS[i] + y * BUCKET_SIN[i];
    if (d > best_dot) { best_dot = d; best = i; }
  }
  return best;
}

var<workgroup> s_rotated: array<f32, 512>;
var<workgroup> s_projected: array<f32, 512>;
var<workgroup> s_reduce: array<f32, 256>;
var<workgroup> s_max_r: f32;
var<workgroup> s_sigma: f32;
var<workgroup> s_num: f32;
var<workgroup> s_polar: array<atomic<u32>, 96>;
var<workgroup> s_qjl: array<atomic<u32>, 32>;

// In-place FWHT on s_rotated[0..dim). Caller must barrier before and after.
fn fwht_inplace(tid: u32, dim: u32) {
  var stride = 1u;
  while (stride < dim) {
    var i = tid;
    while (i < dim) {
      if ((i & stride) == 0u) {
        let j = i | stride;
        let a = s_rotated[i];
        let b = s_rotated[j];
        s_rotated[i] = a + b;
        s_rotated[j] = a - b;
      }
      i += 256u;
    }
    stride *= 2u;
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn encode(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let vec_idx = wid.x;
  if (vec_idx >= params.num_vectors) { return; }

  let tid = lid.x;
  let dim = params.dim;
  let num_pairs = dim / 2u;
  let vec_offset = vec_idx * dim;
  let out_idx = params.write_pos + vec_idx;
  let scale = 1.0 / sqrt(f32(dim));

  // Phase 1: FWHT rotation — D × input then butterfly.
  var r = tid;
  while (r < dim) {
    s_rotated[r] = raw_input[vec_offset + r] * f32(signs[r]);
    r += 256u;
  }
  workgroupBarrier();
  fwht_inplace(tid, dim);
  // Scale result.
  r = tid;
  while (r < dim) {
    s_rotated[r] = s_rotated[r] * scale;
    r += 256u;
  }
  workgroupBarrier();

  // Phase 2: max_r reduction.
  var local_max = 0.0f;
  var p = tid;
  while (p < num_pairs) {
    let x = s_rotated[p * 2u];
    let y = s_rotated[p * 2u + 1u];
    local_max = max(local_max, sqrt(x * x + y * y));
    p += 256u;
  }
  s_reduce[tid] = local_max;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (tid < stride) { s_reduce[tid] = max(s_reduce[tid], s_reduce[tid + stride]); }
    stride /= 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    s_max_r = select(s_reduce[0], 1.0, s_reduce[0] < 1e-10);
    max_r_out[out_idx] = s_max_r;
  }
  workgroupBarrier();
  let max_r = s_max_r;

  // Phase 3: Polar encode.
  if (tid < params.polar_words_per_pos) {
    atomicStore(&s_polar[tid], 0u);
  }
  workgroupBarrier();

  p = tid;
  while (p < num_pairs) {
    let x = s_rotated[p * 2u];
    let y = s_rotated[p * 2u + 1u];
    let r_val = sqrt(x * x + y * y);
    let r_q = u32(clamp(round(r_val / max_r * POLAR_RADIUS_LEVELS_F), 0.0, POLAR_RADIUS_LEVELS_F));
    let bucket = nearest_bucket(x, y);
    let combined = (r_q << POLAR_ANGLE_BITS) | bucket;
    let bit_pos = p * POLAR_PAIR_BITS;
    let word_idx = bit_pos / 32u;
    let bit_off = bit_pos % 32u;
    atomicOr(&s_polar[word_idx], combined << bit_off);
    if (bit_off + POLAR_PAIR_BITS > 32u) {
      atomicOr(&s_polar[word_idx + 1u], combined >> (32u - bit_off));
    }
    p += 256u;
  }
  workgroupBarrier();

  // Write polar output.
  if (tid < params.polar_words_per_pos) {
    polar_out[tid * params.pos_stride + out_idx] = atomicLoad(&s_polar[tid]);
  }

  // Compute residual: s_rotated = s_rotated - polar_decoded.
  // This overwrites s_rotated with the residual for Phase 4's FWHT.
  p = tid;
  while (p < num_pairs) {
    let bit_pos_p = p * POLAR_PAIR_BITS;
    let w_idx = bit_pos_p / 32u;
    let b_off = bit_pos_p % 32u;
    var comb: u32;
    if (b_off + POLAR_PAIR_BITS <= 32u) {
      comb = (atomicLoad(&s_polar[w_idx]) >> b_off) & POLAR_PAIR_MASK;
    } else {
      comb = ((atomicLoad(&s_polar[w_idx]) >> b_off) | (atomicLoad(&s_polar[w_idx + 1u]) << (32u - b_off))) & POLAR_PAIR_MASK;
    }
    let r_decoded = f32(comb >> POLAR_ANGLE_BITS) / POLAR_RADIUS_LEVELS_F * max_r;
    let bucket = comb & POLAR_ANGLE_MASK;
    s_rotated[p * 2u] -= r_decoded * BUCKET_COS[bucket];
    s_rotated[p * 2u + 1u] -= r_decoded * BUCKET_SIN[bucket];
    p += 256u;
  }
  workgroupBarrier();

  // Phase 4: QJL projection via FWHT on the residual.
  if (tid < params.qjl_words_per_pos) {
    atomicStore(&s_qjl[tid], 0u);
  }

  // Apply D signs to residual, then FWHT, then scale → s_projected.
  // Reuse s_rotated for the transform (sign flip in place).
  r = tid;
  while (r < dim) {
    s_rotated[r] = s_rotated[r] * f32(signs[r]);
    r += 256u;
  }
  workgroupBarrier();
  fwht_inplace(tid, dim);
  // Scale and copy to s_projected, accumulating sumsq for sigma.
  var local_sumsq = 0.0f;
  var proj_dim = tid;
  while (proj_dim < dim) {
    let projected = s_rotated[proj_dim] * scale;
    s_projected[proj_dim] = projected;
    local_sumsq += projected * projected;
    proj_dim += 256u;
  }

  // Reduce sumsq → sigma.
  s_reduce[tid] = local_sumsq;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (tid < stride) { s_reduce[tid] += s_reduce[tid + stride]; }
    stride /= 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    s_sigma = sqrt(s_reduce[0] / f32(dim));
  }
  workgroupBarrier();
  let sigma = s_sigma;

  // Phase 4b: Quantize + pack + gamma fit.
  var local_num = 0.0f;
  var local_den = 0.0f;
  proj_dim = tid;
  while (proj_dim < dim) {
    let projected = s_projected[proj_dim];
    let idx = qjl_quantize_idx(projected, sigma);
    let level = QJL_LEVELS[idx];

    let word_idx = proj_dim / QJL_PER_WORD;
    let bit_off = (proj_dim % QJL_PER_WORD) * QJL_BITS;
    atomicOr(&s_qjl[word_idx], idx << bit_off);

    local_num += projected * level;
    local_den += level * level;
    proj_dim += 256u;
  }

  // Reduce num then den.
  s_reduce[tid] = local_num;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (tid < stride) { s_reduce[tid] += s_reduce[tid + stride]; }
    stride /= 2u;
    workgroupBarrier();
  }
  if (tid == 0u) { s_num = s_reduce[0]; }
  workgroupBarrier();

  s_reduce[tid] = local_den;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (tid < stride) { s_reduce[tid] += s_reduce[tid + stride]; }
    stride /= 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    gamma_out[out_idx] = s_num / s_reduce[0];
  }
  workgroupBarrier();

  // Write QJL output.
  if (tid < params.qjl_words_per_pos) {
    qjl_out[tid * params.pos_stride + out_idx] = atomicLoad(&s_qjl[tid]);
  }
}
