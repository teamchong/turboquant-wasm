// Extract and dequantize one row from Q4_K weight matrix.
// Used for embedding lookup: output[:] = dequant(weights[row, :])
// Same dequant logic as matmul-q4k but without the dot product.

enable f16;

struct q4_k {
  d: f16,
  dmin: f16,
  scales: array<u32, 3>,
  qs: array<u32, 32>,
}

struct Params {
  n_cols: u32,
  batch_pos: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<q4_k>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Row comes from a GPU-resident storage buffer so that back-to-back token
// generations can chain: encoder N's argmax → tokenIdBuf, encoder N+1's embed
// reads tokenIdBuf[0] without a CPU roundtrip. For batch prefill, each token
// in an encoder uses its own slot (`batch_pos`) into the same array.
@group(0) @binding(3) var<storage, read> token_ids: array<u32>;

fn get_byte(value: u32, index: u32) -> u32 {
  return (value >> (index * 8u)) & 0xFFu;
}

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
fn get_rows_q4k(@builtin(global_invocation_id) gid: vec3u) {
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
  let sub_offset = within_block % 32u;
  let scale_min = get_scale_min(sub_block, block.scales);
  let dl = d * scale_min.x;
  let ml = m * scale_min.y;

  let qs_group = sub_block / 2u;
  let is_high = (sub_block & 1u) != 0u;
  let q_idx = qs_group * 32u + sub_offset;
  let q_byte = get_byte(block.qs[q_idx / 4u], q_idx % 4u);
  let shift = select(0u, 4u, is_high);
  let qs_val = (q_byte >> shift) & 0xFu;

  output[col] = f32(qs_val) * dl - ml;
}
