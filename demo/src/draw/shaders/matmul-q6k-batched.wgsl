// Q6_K matrix × N activation vectors → N output vectors.
//
// Same layout as matmul-q6k.wgsl (64-thread workgroup, 8 rows per WG,
// 8 threads per row — one 32-weight group per lane). The batched
// variant holds MAX_BATCH accumulators per thread and amortizes the
// Q6_K dequant cost (which includes the nibble/qh/scale reads that
// are the actual bottleneck — see the `33882cd` commit in the perf
// notes where fixing this Q6K kernel gave the session's biggest win)
// across all N activation vectors.
//
// For decode (N=1) the engine still dispatches the unbatched kernel;
// this shader is only called for prefill or speculative decoding where
// the weight bandwidth is the dominant cost and can be shared.
//
// Max batch: 8. Shared memory: 8 × 256 × 4 B (sact) = 8 KB; reduction
// via subgroupShuffleXor so no sred scratch. Well under the 16 KB
// default workgroup storage limit.

enable subgroups;

struct Params {
  n_rows: u32,
  n_cols: u32,
  batch_size: u32,
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
const MAX_BATCH: u32 = 8u;

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

var<workgroup> sact: array<f32, 2048>;  // MAX_BATCH * 256

@compute @workgroup_size(WG_SIZE)
fn matmul_q6k_batched(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let tid = lid.x;
  let base_row = (wid.x + wid.y * nwg.x) * ROWS_PER_WG;
  let row_in_wg = tid / THREADS_PER_ROW;
  let lane = tid % THREADS_PER_ROW;
  let row = base_row + row_in_wg;
  let batch_size = params.batch_size;

  let blocks_per_row = params.n_cols / BLOCK_SIZE;

  // Lane to (half, group) mapping — one 32-weight group per lane.
  let half = lane / 4u;
  let group = lane % 4u;

  // Per-thread accumulators.
  var sums: array<f32, 8>;
  for (var b = 0u; b < MAX_BATCH; b++) { sums[b] = 0.0; }

  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    // Cooperative load of N activation tiles.
    let col_base = block_k * BLOCK_SIZE;
    for (var b = 0u; b < batch_size; b++) {
      let src_base = b * params.n_cols + col_base;
      let dst_base = b * 256u;
      sact[dst_base + tid]        = input[src_base + tid];
      sact[dst_base + tid +  64u] = input[src_base + tid +  64u];
      sact[dst_base + tid + 128u] = input[src_base + tid + 128u];
      sact[dst_base + tid + 192u] = input[src_base + tid + 192u];
    }
    workgroupBarrier();

    if (row < params.n_rows) {
      let block_byte = row * blocks_per_row * BYTES_PER_BLOCK + block_k * BYTES_PER_BLOCK;
      let d = read_f16(block_byte + 208u);

      let ql_off = (group & 1u) * 32u;
      let qh_shift = group * 2u;
      let is_high_nibble = group >= 2u;
      let col_base_lane = half * 128u + group * 32u;
      let sc_offset = group * 2u;
      let ql_base = block_byte + half * 64u + ql_off;
      let qh_base = block_byte + 128u + half * 32u;
      let sc_base = block_byte + 192u + half * 8u + sc_offset;

      // sc only changes every 16 iters; split into two loops with scale×d
      // premultiplied so the inner FMA chain saves one f32 mul per step.
      let d_sc0 = d * f32(read_i8(sc_base));
      for (var l = 0u; l < 16u; l++) {
        let ql_byte = read_u8(ql_base + l);
        let ql_nibble = select(ql_byte & 0xFu, ql_byte >> 4u, is_high_nibble);
        let qh_bits = (read_u8(qh_base + l) >> qh_shift) & 3u;
        let q6 = ql_nibble | (qh_bits << 4u);
        let w = d_sc0 * f32(i32(q6) - 32);
        let col_in_block = col_base_lane + l;
        for (var b = 0u; b < batch_size; b++) {
          sums[b] = sums[b] + w * sact[b * 256u + col_in_block];
        }
      }
      let d_sc1 = d * f32(read_i8(sc_base + 1u));
      for (var l = 16u; l < 32u; l++) {
        let ql_byte = read_u8(ql_base + l);
        let ql_nibble = select(ql_byte & 0xFu, ql_byte >> 4u, is_high_nibble);
        let qh_bits = (read_u8(qh_base + l) >> qh_shift) & 3u;
        let q6 = ql_nibble | (qh_bits << 4u);
        let w = d_sc1 * f32(i32(q6) - 32);
        let col_in_block = col_base_lane + l;
        for (var b = 0u; b < batch_size; b++) {
          sums[b] = sums[b] + w * sact[b * 256u + col_in_block];
        }
      }
    }
    workgroupBarrier();
  }

  // Butterfly reduction per slot via subgroupShuffleXor. Matches the
  // unbatched Q6K kernel's reduction structure for bit-identity.
  for (var b = 0u; b < batch_size; b++) {
    var s = sums[b];
    s = s + subgroupShuffleXor(s, 1u);
    s = s + subgroupShuffleXor(s, 2u);
    s = s + subgroupShuffleXor(s, 4u);
    if (lane == 0u && row < params.n_rows) {
      output[b * params.n_rows + row] = s;
    }
  }
}
