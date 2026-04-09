// TurboQuant KV cache encoder — compress a float32 vector to ~3 bits/dim.
// Runs as a WebGPU compute shader during KV cache update.
//
// Input: float32 vector (head_dim dimensions, already rotated by model)
// Output: compressed blob (polar 7-bit pairs + QJL sign bits)
//
// One workgroup encodes one vector. Threads cooperate on the encoding.

struct EncodeConfig {
  dim: u32,                // head_dim (e.g. 256)
  num_vectors: u32,        // number of vectors to encode this step
  polar_bytes: u32,        // bytes for polar data per vector
  qjl_bytes: u32,          // bytes for QJL data per vector
  blob_u32s_per_vec: u32,  // total u32s per compressed vector (header + polar + qjl)
  header_u32s: u32,        // header size in u32s (max_r, gamma stored here)
  _pad0: u32,
  _pad1: u32,
}

// Input: raw float32 K or V vectors [num_vectors][dim]
@group(0) @binding(0) var<storage, read> input_vectors: array<f32>;
// Output: compressed blobs [num_vectors][blob_u32s_per_vec]
@group(0) @binding(1) var<storage, read_write> compressed: array<u32>;
// Config
@group(0) @binding(2) var<uniform> config: EncodeConfig;

const R_BITS: u32 = 4u;
const THETA_BITS: u32 = 3u;
const BITS_PER_PAIR: u32 = 7u;
const R_LEVELS: f32 = 15.0;
const PI: f32 = 3.14159265;
const TWO_PI: f32 = 6.28318530;
const THETA_LEVELS: f32 = 7.0;
const SQRT_PI_OVER_2: f32 = 1.2533141;

// Precomputed direction vectors for 8 angle buckets
const DIR_COS = array<f32, 8>(
  -1.0, -0.62349, 0.22252, 0.90097, 0.90097, 0.22252, -0.62349, -1.0
);
const DIR_SIN = array<f32, 8>(
  0.0, 0.78183, 0.97493, 0.43388, -0.43388, -0.97493, -0.78183, 0.0
);

fn find_nearest_angle(x: f32, y: f32) -> u32 {
  var best_bucket: u32 = 0u;
  var best_dot: f32 = -1e30;
  let len = sqrt(x * x + y * y);
  let nx = select(0.0, x / len, len > 1e-10);
  let ny = select(0.0, y / len, len > 1e-10);

  for (var b: u32 = 0u; b < 8u; b += 1u) {
    let d = nx * DIR_COS[b] + ny * DIR_SIN[b];
    if (d > best_dot) {
      best_dot = d;
      best_bucket = b;
    }
  }
  return best_bucket;
}

fn write_bit(blob_offset: u32, bit_pos: u32, bit_val: u32) {
  if (bit_val == 0u) { return; }
  let byte_idx = bit_pos / 8u;
  let word_idx = blob_offset + byte_idx / 4u;
  let shift = (byte_idx % 4u) * 8u + (bit_pos % 8u);
  // Atomic OR to avoid race conditions between threads in same workgroup
  atomicOr(&compressed[word_idx], 1u << shift);
}

// Each workgroup processes one vector. Thread 0 does all the work
// (the encoding is sequential due to bit-packing dependencies).
// TODO: parallelize across pairs with workgroup-level bit assembly.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vec_id = gid.x;
  if (vec_id >= config.num_vectors) { return; }

  let dim = config.dim;
  let num_pairs = dim / 2u;
  let input_offset = vec_id * dim;
  let blob_offset = vec_id * config.blob_u32s_per_vec;

  // Zero the output
  for (var i: u32 = 0u; i < config.blob_u32s_per_vec; i += 1u) {
    compressed[blob_offset + i] = 0u;
  }

  // Step 1: Compute max_r (maximum radius across all pairs)
  var max_r: f32 = 0.0;
  for (var i: u32 = 0u; i < num_pairs; i += 1u) {
    let x = input_vectors[input_offset + i * 2u];
    let y = input_vectors[input_offset + i * 2u + 1u];
    let r = sqrt(x * x + y * y);
    max_r = max(max_r, r);
  }
  if (max_r < 1e-10) { max_r = 1.0; }

  // Step 2: Compute gamma (norm of the vector, for QJL scaling)
  var norm_sq: f32 = 0.0;
  for (var d: u32 = 0u; d < dim; d += 1u) {
    let v = input_vectors[input_offset + d];
    norm_sq += v * v;
  }
  let gamma = sqrt(norm_sq);

  // Step 3: Write header (max_r, gamma as f32)
  compressed[blob_offset] = bitcast<u32>(max_r);
  compressed[blob_offset + 1u] = bitcast<u32>(gamma);

  // Step 4: Polar encoding — 7 bits per pair (4 radius + 3 angle)
  let polar_bit_start = config.header_u32s * 32u;
  for (var i: u32 = 0u; i < num_pairs; i += 1u) {
    let x = input_vectors[input_offset + i * 2u];
    let y = input_vectors[input_offset + i * 2u + 1u];
    let r = sqrt(x * x + y * y);
    let r_quant = u32(min(r / max_r * R_LEVELS, R_LEVELS));
    let theta_bucket = find_nearest_angle(x, y);
    let combined = (r_quant << THETA_BITS) | theta_bucket;

    let bit_pos = polar_bit_start + i * BITS_PER_PAIR;
    for (var j: u32 = 0u; j < BITS_PER_PAIR; j += 1u) {
      write_bit(blob_offset, bit_pos + j, (combined >> j) & 1u);
    }
  }

  // Step 5: QJL encoding — 1 sign bit per dimension
  // QJL encodes sign(R * residual) where R is a random rotation.
  // For KV cache, we skip the rotation and encode sign(vector) directly.
  // This is the "fast" QJL variant — same as estimateDotFast in the Zig code.
  let qjl_bit_start = polar_bit_start + num_pairs * BITS_PER_PAIR;
  for (var d: u32 = 0u; d < dim; d += 1u) {
    // Compute residual: vector - polar_reconstruction
    // For the fast path, just use the sign of each dimension
    let v = input_vectors[input_offset + d];
    let bit_val = select(0u, 1u, v > 0.0);
    write_bit(blob_offset, qjl_bit_start + d, bit_val);
  }
}
