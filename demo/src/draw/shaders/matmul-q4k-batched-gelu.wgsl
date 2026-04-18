// Q4_K batched matmul × N activation vectors with GELU-gated activation
// fused into the input load, for the FFN down-projection on the prefill
// path (runLayerBatched + chunked prefill).
//
// Identical to matmul-q4k-batched-fixed.wgsl except that for each batch
// slot, the cooperative load reads gate[] + up[] and applies GELU mixing
// inline:
//
//     input[b, i] = gelu(gate[b, i]) * up[b, i]
//
// BATCH_N is injected at pipeline compile time by the engine (2u/4u/8u)
// so Naga can fully unroll the per-slot batch loop into scalar
// accumulators — the runtime-N version of the non-fused kernel spills
// on Metal and the measurement confirmed 1.5-5.5× regressions, so we
// keep the compile-time specialization.

enable f16;
enable subgroups;

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
@group(0) @binding(2) var<storage, read> gate: array<f32>;    // [BATCH_N × n_cols] ffn_gate output
@group(0) @binding(3) var<storage, read> up: array<f32>;      // [BATCH_N × n_cols] ffn_up output
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

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

fn gelu_gated(g: f32, u: f32) -> f32 {
  let inner = clamp(SQRT_2_OVER_PI * (g + COEFF * g * g * g), -15.0, 15.0);
  let gelu = 0.5 * g * (1.0 + tanh(inner));
  return gelu * u;
}

var<workgroup> sact: array<f32, 2048>;  // max 8 × 256

@compute @workgroup_size(WG_SIZE)
fn matmul_q4k_batched_gelu(
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

  var sums: array<f32, BATCH_N>;
  for (var b = 0u; b < BATCH_N; b++) { sums[b] = 0.0; }

  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    let col_base = block_k * 256u;
    for (var b = 0u; b < BATCH_N; b++) {
      let src_base = b * params.n_cols + col_base;
      let dst_base = b * 256u;
      sact[dst_base + tid]        = gelu_gated(gate[src_base + tid],        up[src_base + tid]);
      sact[dst_base + tid +  64u] = gelu_gated(gate[src_base + tid +  64u], up[src_base + tid +  64u]);
      sact[dst_base + tid + 128u] = gelu_gated(gate[src_base + tid + 128u], up[src_base + tid + 128u]);
      sact[dst_base + tid + 192u] = gelu_gated(gate[src_base + tid + 192u], up[src_base + tid + 192u]);
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
        for (var b = 0u; b < BATCH_N; b++) {
          sums[b] = sums[b] + w * sact[b * 256u + col_base_sb + j];
        }
      }
    }
    workgroupBarrier();
  }

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
