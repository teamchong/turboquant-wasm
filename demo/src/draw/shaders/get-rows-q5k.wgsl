// Extract and dequantize one row from a Q5_K weight matrix.
// Q5_K block format (QK_K=256 values per block):
//   struct {
//     f16 d;         // super-block scale for quantized scales
//     f16 dmin;      // super-block scale for quantized mins
//     u8  scales[12];// 8 sub-blocks of (scale, min) packed with 6 bits each
//     u8  qh[32];    // high bits (see below)
//     u8  qs[128];   // low 4 bits for each of 256 values
//   }; // 176 bytes
//
// Reference: llama.cpp dequantize_row_q5_K. The outer loop iterates j = 0, 64, 128, 192
// (4 passes). Each pass processes 64 output values: first 32 using low nibbles of
// ql[0..31], then 32 using high nibbles of the same ql bytes. `ql` advances by 32 each
// pass, and the qh masks `u1`, `u2` shift left by 2 each pass:
//   j = 0:   u1 = 0x01, u2 = 0x02
//   j = 64:  u1 = 0x04, u2 = 0x08
//   j = 128: u1 = 0x10, u2 = 0x20
//   j = 192: u1 = 0x40, u2 = 0x80
// so qh[l] bit (2*(j/64))   provides the high bit for the low-nibble half and
//    qh[l] bit (2*(j/64)+1) provides it for the high-nibble half.
//
// For within-block index `within_block` (0..255):
//   j_block       = within_block / 64           (0..3)
//   k             = within_block % 64           (0..63)
//   l             = k % 32                       (0..31)  -- byte offset inside qh and within the j_block's ql group
//   is_high_nibble= k >= 32                       (bool)   -- whether we use the high nibble of ql[j_block*32 + l]
//   ql_byte_idx   = j_block * 32 + l              (0..127) -- position in qs array
//   qh_byte_idx   = l                              (0..31)
//   qh_bit        = 2 * j_block + (is_high_nibble ? 1 : 0) -- bit within qh[l]
//
// Sub-block scale index is still `sub_block = within_block / 32` (0..7), identical to Q4_K.

enable f16;

struct q5_k {
  d:      f16,
  dmin:   f16,
  scales: array<u32, 3>,  // 12 bytes
  qh:     array<u32, 8>,  // 32 bytes
  qs:     array<u32, 32>, // 128 bytes
}

struct Params {
  n_cols:    u32,
  batch_pos: u32,
  _p1:       u32,
  _p2:       u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<q5_k>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Row comes from a GPU-resident storage buffer so the decode loop can chain
// encoders without a CPU roundtrip on the token id. Batch prefill writes CHUNK
// token ids at once and each dispatch reads its own slot via `batch_pos`.
@group(0) @binding(3) var<storage, read> token_ids: array<u32>;

fn get_byte(value: u32, index: u32) -> u32 {
  return (value >> (index * 8u)) & 0xFFu;
}

// Same 6-bit scale/min packing as Q4_K — reused verbatim.
fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
  if (is < 4u) {
    let sc_byte = get_byte(scales[is / 4u], is % 4u);
    let min_byte = get_byte(scales[(is + 4u) / 4u], is % 4u);
    return vec2(f32(sc_byte & 63u), f32(min_byte & 63u));
  } else {
    let sc_min_lo = get_byte(scales[(is + 4u) / 4u], (is + 4u) % 4u);
    let sc_hi = get_byte(scales[(is - 4u) / 4u], (is - 4u) % 4u);
    let min_hi = get_byte(scales[is / 4u], is % 4u);
    let sc = (sc_min_lo & 0xFu) | ((sc_hi >> 6u) << 4u);
    let m = (sc_min_lo >> 4u) | ((min_hi >> 6u) << 4u);
    return vec2(f32(sc), f32(m));
  }
}

@compute @workgroup_size(256)
fn get_rows_q5k(@builtin(global_invocation_id) gid: vec3u) {
  let col = gid.x;
  if (col >= params.n_cols) { return; }

  let row = token_ids[params.batch_pos];
  let blocks_per_row = params.n_cols / 256u;
  let row_start = row * blocks_per_row;
  let block_idx = col / 256u;
  let within_block = col % 256u;
  let block = weights[row_start + block_idx];
  let d = f32(block.d);
  let m = f32(block.dmin);

  let sub_block = within_block / 32u;
  let scale_min = get_scale_min(sub_block, block.scales);
  let dl = d * scale_min.x;
  let ml = m * scale_min.y;

  let j_block = within_block / 64u;
  let k = within_block % 64u;
  let l = k % 32u;
  let is_high_nibble = k >= 32u;

  // Low 4 bits from qs[j_block*32 + l], low or high nibble depending on is_high_nibble.
  let ql_byte_idx = j_block * 32u + l;
  let ql_byte = get_byte(block.qs[ql_byte_idx / 4u], ql_byte_idx % 4u);
  let ql_val = select(ql_byte & 0xFu, (ql_byte >> 4u) & 0xFu, is_high_nibble);

  // High bit from qh[l], at bit position 2*j_block (+1 for high nibble).
  let qh_byte_idx = l;
  let qh_byte = get_byte(block.qh[qh_byte_idx / 4u], qh_byte_idx % 4u);
  let qh_bit_pos = 2u * j_block + select(0u, 1u, is_high_nibble);
  let qh_bit = (qh_byte >> qh_bit_pos) & 1u;

  let q = f32(ql_val | (qh_bit << 4u));
  output[col] = q * dl - ml;
}
