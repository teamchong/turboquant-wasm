// Q4_K matrix × N activation vectors (fixed compile-time N).
//
// Variant of matmul-q4k-batched.wgsl where BATCH_N is injected at
// pipeline-compile time so Naga can fully unroll the inner batch loop
// and the per-thread accumulators become scalar registers rather than
// an array<f32, MAX_BATCH> (which was being indirectly indexed in the
// runtime-N version and almost certainly spilling to memory on Metal —
// the microbenchmark showed the runtime-N kernel was 1.5-5.5× slower
// than N sequential unbatched dispatches).
//
// Engine code compiles one pipeline per N in {2,4,8} and picks the
// right one at dispatch time based on the actual batch size. N must
// match batch_size exactly — partial batches use the nearest-higher-N
// pipeline and zero-pad the extra slots.

enable f16;
enable subgroups;

// Injected by engine.ts via string replacement. Must be 2, 4, or 8.
const BATCH_N: u32 = /*@BATCH_N@*/;

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

// Shared memory: BATCH_N activation tiles stacked.
var<workgroup> sact: array<f32, 2048>;  // max 8 * 256

@compute @workgroup_size(WG_SIZE)
fn matmul_q4k_batched_fixed(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
) {
  let tid = lid.x;
  let base_row = (wid.x + wid.y * nwg.x) * ROWS_PER_WG;
  let row_in_wg = tid / THREADS_PER_ROW;
  let lane = tid % THREADS_PER_ROW;
  let row = base_row + row_in_wg;

  let blocks_per_row = params.n_cols / 256u;

  // Scalar accumulators — compile-time BATCH_N so Naga register-allocates.
  var sums: array<f32, BATCH_N>;
  for (var b = 0u; b < BATCH_N; b++) { sums[b] = 0.0; }

  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    let col_base = block_k * 256u;
    for (var b = 0u; b < BATCH_N; b++) {
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

      for (var j = 0u; j < 32u; j++) {
        let q_idx = qs_group * 32u + j;
        let q_byte = get_byte(block.qs[q_idx / 4u], q_idx % 4u);
        let qs_val = (q_byte >> shift) & 0xFu;
        let w = f32(qs_val) * dl - ml;
        // Unrolled batch loop — BATCH_N is compile-time so this fully
        // unrolls into BATCH_N scalar FMAs per iteration.
        for (var b = 0u; b < BATCH_N; b++) {
          sums[b] = sums[b] + w * sact[b * 256u + col_base_sb + j];
        }
      }
    }
    workgroupBarrier();
  }

  // Per-slot butterfly reduction.
  for (var b = 0u; b < BATCH_N; b++) {
    var s = sums[b];
    s = s + subgroupShuffleXor(s, 1u);
    s = s + subgroupShuffleXor(s, 2u);
    s = s + subgroupShuffleXor(s, 4u);
    if (lane == 0u && row < params.n_rows) {
      output[b * params.n_rows + row] = s;
    }
  }
}
