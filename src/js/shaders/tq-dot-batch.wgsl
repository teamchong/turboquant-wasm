// TurboQuant WebGPU compute shader — batch dot product on compressed vectors.
//
// Each vector is stored as a full blob: [32-byte header][polar][qjl][padding]
// One workgroup (64 threads) per vector.
// Polar: 7 bits per pair, LSB-first. QJL: 1 bit per dimension, LSB-first.

struct Config {
  num_vectors: u32,
  dim: u32,
  blob_u32s_per_vec: u32,
  polar_byte_offset: u32,
  qjl_byte_offset: u32,
  q_sum: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> blob_data: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<vec2<f32>>;

@group(1) @binding(0) var<storage, read> query: array<f32>;
@group(1) @binding(1) var<uniform> config: Config;
@group(1) @binding(2) var<storage, read_write> scores: array<f32>;

@group(2) @binding(0) var<storage, read> polar_lut: array<vec2<f32>, 128>;

const WG_SIZE: u32 = 64u;
const SQRT_PI_OVER_2: f32 = 1.2533141;

var<workgroup> polar_partial: array<f32, 64>;
var<workgroup> qjl_partial: array<f32, 64>;

fn read_byte(blob_offset: u32, byte_off: u32) -> u32 {
  let word_idx = blob_offset + byte_off / 4u;
  let shift = (byte_off % 4u) * 8u;
  return (blob_data[word_idx] >> shift) & 0xFFu;
}

fn extract7(blob_offset: u32, polar_start_byte: u32, bit_pos: u32) -> u32 {
  let byte_idx = polar_start_byte + bit_pos / 8u;
  let bit_off = bit_pos % 8u;
  let b0 = read_byte(blob_offset, byte_idx);
  let b1 = read_byte(blob_offset, byte_idx + 1u);
  let window = b0 | (b1 << 8u);
  return (window >> bit_off) & 0x7Fu;
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
  let blob_offset = vec_id * config.blob_u32s_per_vec;
  let polar_start = config.polar_byte_offset;
  let qjl_start = config.qjl_byte_offset;

  var p_sum: f32 = 0.0;
  var pair_idx = tid;
  while (pair_idx < num_pairs) {
    let bit_pos = pair_idx * 7u;
    let combined = extract7(blob_offset, polar_start, bit_pos);
    let lut_val = polar_lut[combined];
    let dx = lut_val.x * max_r;
    let dy = lut_val.y * max_r;

    let d = pair_idx * 2u;
    p_sum += query[d] * dx + query[d + 1u] * dy;

    pair_idx += WG_SIZE;
  }

  var pos_sum: f32 = 0.0;
  var d_idx = tid;
  while (d_idx < dim) {
    let byte_off = qjl_start + d_idx / 8u;
    let bit_off = d_idx % 8u;
    let byte_val = read_byte(blob_offset, byte_off);
    let bit = (byte_val >> bit_off) & 1u;
    pos_sum += query[d_idx] * f32(bit);

    d_idx += WG_SIZE;
  }

  polar_partial[tid] = p_sum;
  qjl_partial[tid] = pos_sum;
  workgroupBarrier();

  for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      polar_partial[tid] += polar_partial[tid + stride];
      qjl_partial[tid] += qjl_partial[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let polar_total = polar_partial[0];
    let pos_total = qjl_partial[0];
    let qjl_scale = SQRT_PI_OVER_2 / f32(dim) * gamma;
    let qjl_total = (2.0 * pos_total - config.q_sum) * qjl_scale;
    scores[vec_id] = polar_total + qjl_total;
  }
}
