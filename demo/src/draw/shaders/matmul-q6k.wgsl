// Q6_K matrix-vector multiply — 64-thread workgroup, 8 rows per
// workgroup, 8 threads per output row (one per 32-weight group).
//
// Q6_K block layout (210 bytes, 256 weights):
//   ql[128]: lower 4 bits (2 quants per byte, low/high nibble)
//   qh[64]:  upper 2 bits (4 quants per byte)
//   scales[16]: int8 per-sub-block scale
//   d: f16 super-block scale
//
// Dequant: weight = d * scales[is] * (q6 - 32)
// where q6 = (ql_nibble | (qh_bits << 4)), 6-bit range 0..63, centered at 32.
//
// Thread layout mirrors matmul-q4k:
//   - 64 threads total per workgroup
//   - 8 output rows per workgroup (8 threads cooperate per row)
//   - Each thread owns 1 of the 8 × 32-weight groups in a Q6_K block
//     (half = lane/4, group = lane%4; half in {0,1}, group in {0..3}).
//   - Reduction via `subgroupShuffleXor` butterfly (strides 1, 2, 4). The
//     8 lanes of a row fit inside one 32-wide Metal subgroup, so the
//     masks permute within the 8-group; drops 3 workgroupBarriers and
//     the sred scratch vs the old tree reduction.

enable subgroups;

struct Params {
  n_rows: u32,
  n_cols: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights_raw: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BLOCK_SIZE: u32 = 256u;
const BYTES_PER_BLOCK: u32 = 210u;
const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

fn read_u8(byte_offset: u32) -> u32 {
  let word = weights_raw[byte_offset / 4u];
  return (word >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

fn read_i8(byte_offset: u32) -> i32 {
  let u = read_u8(byte_offset);
  return i32(u) - select(0, 256, u >= 128u);
}

fn read_f16(byte_offset: u32) -> f32 {
  let word = weights_raw[byte_offset / 4u];
  let pair = unpack2x16float(word);
  return select(pair.x, pair.y, (byte_offset & 3u) >= 2u);
}

// Shared memory: one activation tile (256 f32). Reduction uses subgroup
// ops so no scratch buffer needed.
var<workgroup> sact: array<f32, 256>;

@compute @workgroup_size(WG_SIZE)
fn matmul_q6k(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let tid = lid.x;
  let base_row = (wid.x + wid.y * nwg.x) * ROWS_PER_WG;
  let row_in_wg = tid / THREADS_PER_ROW;  // 0..7
  let lane = tid % THREADS_PER_ROW;       // 0..7 — one 32-weight group per lane
  let row = base_row + row_in_wg;

  let blocks_per_row = params.n_cols / BLOCK_SIZE;
  var sum = 0.0f;

  // Lane to (half, group) mapping — one 32-weight group per lane.
  // Precompute group-dependent constants to eliminate the switch statements.
  let half = lane / 4u;
  let group = lane % 4u;
  let ql_off = (group & 1u) * 32u;
  let qh_shift = group * 2u;
  let is_high_nibble = group >= 2u;
  let col_base_lane = half * 128u + group * 32u;
  let sc_offset = group * 2u;

  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    let col_base = block_k * BLOCK_SIZE;
    sact[tid]        = input[col_base + tid];
    sact[tid +  64u] = input[col_base + tid +  64u];
    sact[tid + 128u] = input[col_base + tid + 128u];
    sact[tid + 192u] = input[col_base + tid + 192u];
    workgroupBarrier();

    if (row < params.n_rows) {
      let block_byte = row * blocks_per_row * BYTES_PER_BLOCK + block_k * BYTES_PER_BLOCK;
      let d = read_f16(block_byte + 208u);

      let ql_base = block_byte + half * 64u + ql_off;
      let qh_base = block_byte + 128u + half * 32u;
      let sc_base = block_byte + 192u + half * 8u + sc_offset;

      // sc only changes every 16 iters (l / 16); split into two straight
      // loops with the scale×d premultiplied so the inner FMA chain has
      // one fewer f32 mul per step. Classic loop-invariant code motion.
      let d_sc0 = d * f32(read_i8(sc_base));
      for (var l = 0u; l < 16u; l++) {
        let ql_byte = read_u8(ql_base + l);
        let ql_nibble = select(ql_byte & 0xFu, ql_byte >> 4u, is_high_nibble);
        let qh_bits = (read_u8(qh_base + l) >> qh_shift) & 3u;
        let q6 = ql_nibble | (qh_bits << 4u);
        sum += d_sc0 * f32(i32(q6) - 32) * sact[col_base_lane + l];
      }
      let d_sc1 = d * f32(read_i8(sc_base + 1u));
      for (var l = 16u; l < 32u; l++) {
        let ql_byte = read_u8(ql_base + l);
        let ql_nibble = select(ql_byte & 0xFu, ql_byte >> 4u, is_high_nibble);
        let qh_bits = (read_u8(qh_base + l) >> qh_shift) & 3u;
        let q6 = ql_nibble | (qh_bits << 4u);
        sum += d_sc1 * f32(i32(q6) - 32) * sact[col_base_lane + l];
      }
    }
    workgroupBarrier();
  }

  // Butterfly reduction within each 8-thread row-group.
  sum = sum + subgroupShuffleXor(sum, 1u);
  sum = sum + subgroupShuffleXor(sum, 2u);
  sum = sum + subgroupShuffleXor(sum, 4u);

  if (lane == 0u && row < params.n_rows) {
    output[row] = sum;
  }
}
