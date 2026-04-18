// Q4_K matrix-vector multiply with INLINED GELU-gated activation.
//
// Identical to matmul-q4k.wgsl except the activation input is derived on
// the fly from two buffers (gate, up) via GELU-gated mixing instead of a
// pre-reduced geluOut buffer:
//
//     input[i] = gelu(gate[i]) * up[i]
//
// This fuses what used to be two dispatches (gelu_gate + matmul(ffn_down))
// into one. Saves the ffnSize-element round-trip of geluOut through global
// memory on every FFN per layer (35× per decode).
//
// Structure follows matmul-q4k.wgsl verbatim for the Q4_K dequant +
// subgroup reduction; only the cooperative activation load path changed.

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
@group(0) @binding(2) var<storage, read> gate: array<f32>;    // ffn_gate output
@group(0) @binding(3) var<storage, read> up: array<f32>;      // ffn_up output
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 64u;
const ROWS_PER_WG: u32 = 8u;
const THREADS_PER_ROW: u32 = 8u;

// GELU tanh-approximation constants (match gelu.wgsl).
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
  // Clamp the tanh arg exactly as gelu.wgsl does — tanh(15)=1 in f32 and
  // Metal NaN-checks large args. Must stay bit-identical with the
  // unfused path to preserve conformance.
  let inner = clamp(SQRT_2_OVER_PI * (g + COEFF * g * g * g), -15.0, 15.0);
  let gelu = 0.5 * g * (1.0 + tanh(inner));
  return gelu * u;
}

var<workgroup> sact: array<f32, 256>;

@compute @workgroup_size(WG_SIZE)
fn matmul_q4k_gelu(
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
  var sum = 0.0f;

  for (var block_k = 0u; block_k < blocks_per_row; block_k++) {
    let col_base = block_k * 256u;
    // Fused load: pull gate[i] and up[i], GELU-gate them, store in shared.
    // Exactly the same access pattern as matmul-q4k.wgsl so M1's coalesced
    // read behaviour is preserved — we just do the ALU work inline.
    sact[tid]        = gelu_gated(gate[col_base + tid],        up[col_base + tid]);
    sact[tid + 64u]  = gelu_gated(gate[col_base + tid + 64u],  up[col_base + tid + 64u]);
    sact[tid + 128u] = gelu_gated(gate[col_base + tid + 128u], up[col_base + tid + 128u]);
    sact[tid + 192u] = gelu_gated(gate[col_base + tid + 192u], up[col_base + tid + 192u]);
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

  sum = sum + subgroupShuffleXor(sum, 1u);
  sum = sum + subgroupShuffleXor(sum, 2u);
  sum = sum + subgroupShuffleXor(sum, 4u);

  if (lane == 0u && row < params.n_rows) {
    output[row] = sum;
  }
}
