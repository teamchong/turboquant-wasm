// TurboQuant WebGPU compute shader — batch dot product on compressed vectors.
// One thread per vector, 256 threads per workgroup.

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

const SQRT_PI_OVER_2: f32 = 1.2533141;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vec_id = gid.x;
  if (vec_id >= config.num_vectors) { return; }

  let dim = config.dim;
  let p = params[vec_id];
  let max_r = p.x;
  let gamma = p.y;
  let num_pairs = dim / 2u;
  let blob_offset = vec_id * config.blob_u32s_per_vec;
  let polar_start = config.polar_byte_offset;
  let qjl_start = config.qjl_byte_offset;

  var polar_sum: f32 = 0.0;
  for (var i: u32 = 0u; i < num_pairs; i += 1u) {
    let combined = extract7(blob_offset, polar_start, i * 7u);
    let lv = polar_lut[combined];
    let d = i * 2u;
    polar_sum += query[d] * lv.x * max_r + query[d + 1u] * lv.y * max_r;
  }

  // QJL bit scan: read one byte at a time, but reuse the u32 word for 4 consecutive bytes.
  // This reduces storage reads by 4x compared to calling read_byte per dimension.
  var pos_sum: f32 = 0.0;
  var prev_word_idx: u32 = 0xFFFFFFFFu;
  var cached_word: u32 = 0u;
  for (var d: u32 = 0u; d < dim; d += 1u) {
    let byte_idx = qjl_start + d / 8u;
    let word_idx = blob_offset + byte_idx / 4u;
    if (word_idx != prev_word_idx) {
      cached_word = blob_data[word_idx];
      prev_word_idx = word_idx;
    }
    let shift = (byte_idx % 4u) * 8u + (d % 8u);
    let bit = (cached_word >> shift) & 1u;
    pos_sum += query[d] * f32(bit);
  }

  let qjl_scale = SQRT_PI_OVER_2 / f32(dim) * gamma;
  scores[vec_id] = polar_sum + (2.0 * pos_sum - config.q_sum) * qjl_scale;
}
