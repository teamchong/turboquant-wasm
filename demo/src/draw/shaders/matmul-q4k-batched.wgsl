// Q4_K matrix × N activation vectors → N output vectors.
//
// Same thread/row/lane layout as matmul-q4k.wgsl (64-thread workgroup,
// 8 output rows per workgroup, 8 threads per row, one lane per Q4_K
// sub-block). The difference: each thread holds MAX_BATCH accumulators
// and the inner K loop loads the quantized weight byte ONCE, then
// multiplies it by MAX_BATCH different activation values before moving
// on to the next weight. Weight bandwidth — the dominant cost on
// decode — is shared across all N activation vectors in a single
// dispatch, which is the whole point of batched matmul for multi-token
// paths (batched prefill, speculative decoding).
//
// This shader does NOT replace matmul-q4k.wgsl. Decode (N=1) still
// uses the unbatched kernel to avoid any per-token overhead from the
// extra batch loop and from cooperatively loading N tiles into shared
// memory. The batched variant only runs when the engine asks for N>1.
//
// Max batch: 8. Shared memory footprint: 8 × 256 × 4 B (sact) + 8 × 64
// × 4 B (sred) = 10 KB, well under the 16 KB default workgroup
// storage limit. The speculative-decoding measurement in PERF_NOTES
// picked lookahead=8 as the sweet spot, so 8 is the natural cap.

enable f16;
enable subgroups;

const MAX_BATCH: u32 = 8u;

struct q4_k {
  d: f16,
  dmin: f16,
  scales: array<u32, 3>,
  qs: array<u32, 32>,
}

struct Params {
  n_rows: u32,
  n_cols: u32,
  batch_size: u32,   // 1..MAX_BATCH
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<q4_k>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

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

// Shared memory: MAX_BATCH activation tiles stacked. Layout:
// sact[b * 256 + j] for batch b, column j within the current K=256 tile.
// Reduction uses subgroupShuffleXor so no sred scratch buffer.
var<workgroup> sact: array<f32, 2048>;  // MAX_BATCH * 256

@compute @workgroup_size(WG_SIZE)
fn matmul_q4k_batched(
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

  let blocks_per_row = params.n_cols / 256u;

  // Per-thread accumulators — one f32 per batch entry. WGSL requires a
  // compile-time array size, so we reserve MAX_BATCH slots and only
  // touch the first batch_size of them.
  var sums: array<f32, 8>;
  for (var b = 0u; b < MAX_BATCH; b++) { sums[b] = 0.0; }

  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    // Cooperative load of N activation tiles into shared memory. Each
    // tile covers 256 contiguous f32 values; 64 threads × 4 strided
    // loads covers one tile; we repeat that for each batch entry.
    let col_base = block_k * 256u;
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
      let block = weights[row * blocks_per_row + block_k];
      let d = f32(block.d);
      let m = f32(block.dmin);

      let sb = lane;
      let scale_min = get_scale_min(sb, block.scales);
      let dl = d * scale_min.x;
      let ml = m * scale_min.y;
      let qs_group = sb / 2u;
      let is_high = (sb & 1u) != 0u;
      let shift = select(0u, 4u, is_high);
      let col_base_sb = sb * 32u;

      // K loop — 32 weights per sub-block. The weight value `w` is
      // computed ONCE per iteration and multiplied into every batch's
      // accumulator. That's the whole reason this kernel exists: weight
      // load cost amortized across N activation vectors.
      for (var j = 0u; j < 32u; j++) {
        let q_idx = qs_group * 32u + j;
        let q_byte = get_byte(block.qs[q_idx / 4u], q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;
        let w = f32(qs_val) * dl - ml;
        for (var b = 0u; b < batch_size; b++) {
          sums[b] = sums[b] + w * sact[b * 256u + col_base_sb + j];
        }
      }
    }
    workgroupBarrier();
  }

  // Butterfly reduction per slot via subgroupShuffleXor. Matches the
  // unbatched kernel's reduction structure exactly for bit-identity.
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
