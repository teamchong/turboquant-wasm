// Q4_K matrix-vector multiply — 8 rows per workgroup, one thread per
// Q4_K sub-block, subgroup-butterfly per-row reduction.
//
// Structure follows the MLC-generated WebGPU kernels: each output row is
// handled by 8 threads instead of 4, with each thread owning exactly one
// Q4_K sub-block (32 of the 256 weights per block). This halves the
// per-thread serial work in the K loop and doubles the number of threads
// cooperating on each output.
//
// Sub-block layout:
//   Q4_K packs 256 weights into 8 sub-blocks of 32 weights each.
//   Each sub-block has its own 6-bit scale + 6-bit min (unpacked via
//   get_scale_min). With 8 threads per row = 8 sub-blocks per row, the
//   lane-to-sub-block mapping is the identity: lane `l` handles sb `l`.
//
// A 16-threads-per-row variant was tried (each thread handles half a
// sub-block) but regressed — the 4-level reduction tree plus 2× dispatch
// count ate the per-thread work savings. 8 threads per row is the sweet
// spot for Q4_K_M on M1 with our current quant format.
//
// Per-row reduction uses `subgroupShuffleXor` butterfly (strides 1, 2, 4)
// instead of a shared-memory tree. The 8 threads of each row live at
// consecutive tids (`row_in_wg * 8 + 0..7`) which fit inside one 32-wide
// subgroup on Metal, so the XOR masks 1/2/4 permute only within the
// 8-group. After 3 steps every thread in the group holds the full sum;
// lane 0 writes the output. Drops 3 workgroupBarriers + the sred scratch.
//
// Decode (N=1): same weight matrix × 1 input vector → output vector.
// Shared memory caches 256 input values (one Q4_K block), reused across
// all 8 rows.

enable f16;
enable subgroups;

struct q4_k {
  d: f16,
  dmin: f16,
  scales: array<u32, 3>,
  qs: array<u32, 32>,
}

struct Params {
  n_rows: u32,
  n_cols: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<q4_k>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;  // 64 / 8 — matches Q4_K's 8 sub-blocks per block

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

// Shared memory: one activation tile (256 f32, cooperative load from
// 64 threads × 4 loads each). Reduction uses subgroup ops so no scratch.
var<workgroup> sact: array<f32, 256>;

@compute @workgroup_size(WG_SIZE)
fn matmul_q4k(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let tid = lid.x;
  let base_row = (wid.x + wid.y * nwg.x) * ROWS_PER_WG;
  let row_in_wg = tid / THREADS_PER_ROW;  // 0..31
  let lane = tid % THREADS_PER_ROW;       // 0..7 — one sub-block per lane
  let row = base_row + row_in_wg;

  let blocks_per_row = params.n_cols / 256u;
  var sum = 0.0f;

  // Outer K loop — one tile per Q4_K block.
  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    // Cooperative activation load: 64 threads load 4 f32 each (strided
    // by WG_SIZE) to cover a full 256-value Q4_K block.
    let col_base = block_k * 256u;
    sact[tid] = input[col_base + tid];
    sact[tid + 64u] = input[col_base + tid + 64u];
    sact[tid + 128u] = input[col_base + tid + 128u];
    sact[tid + 192u] = input[col_base + tid + 192u];
    workgroupBarrier();

    if (row < params.n_rows) {
      let block = weights[row * blocks_per_row + block_k];
      let d = f32(block.d);
      let m = f32(block.dmin);

      // Lane == sub-block index in 0..7. Each thread handles 32 weights.
      let sb = lane;
      let scale_min = get_scale_min(sb, block.scales);
      let dl = d * scale_min.x;
      let ml = m * scale_min.y;
      let qs_group = sb / 2u;
      let is_high = (sb & 1u) != 0u;
      let shift = select(0u, 4u, is_high);
      let col_base_sb = sb * 32u;

      // 32 weights per sub-block, processed 4 at a time. Each u32 word of
      // qs holds 4 bytes; each byte's low or high nibble (selected by shift)
      // is one weight. Reading the word once per 4 iterations avoids
      // per-iteration get_byte(qs[j/4], j%4) division/modulo that can block
      // the Metal shader compiler's pipeline optimizer.
      let qs_base = qs_group * 8u;
      for (var w = 0u; w < 8u; w++) {
        let qs_word = block.qs[qs_base + w];
        let j = w * 4u;
        sum += (f32((qs_word >> (0u + shift)) & 0xFu) * dl - ml) * sact[col_base_sb + j + 0u];
        sum += (f32((qs_word >> (8u + shift)) & 0xFu) * dl - ml) * sact[col_base_sb + j + 1u];
        sum += (f32((qs_word >> (16u + shift)) & 0xFu) * dl - ml) * sact[col_base_sb + j + 2u];
        sum += (f32((qs_word >> (24u + shift)) & 0xFu) * dl - ml) * sact[col_base_sb + j + 3u];
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
