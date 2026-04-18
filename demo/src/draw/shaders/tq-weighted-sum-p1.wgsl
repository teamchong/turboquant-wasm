// TQ Weighted Sum — Phase 1.
//
// Computes per-(head, dim) polar contribution and per-(head, dim) qjl_sum
// (pre-rotation) by scanning the V cache in the position axis.
//
// The OLD weighted_sum shader did Phase 1 + Phase 2 (R^T mat-vec) in one
// dispatch with only `num_heads` workgroups. At long context the position
// loop is the hot path and `num_heads` (typically 2 for GQA) is nowhere
// near enough parallelism to saturate the GPU — tq_wsum went from 5 ms at
// 1752 positions to 22 ms at 5588 positions (roughly linear scaling with
// cache length).
//
// This version unlocks two parallelism axes:
//   * DIM_TILE (64): split the head dim across multiple workgroups. Each
//     workgroup handles one (head, dim_tile) pair, so dim=256 becomes
//     `num_heads × 4` workgroups and dim=512 becomes `num_heads × 8`.
//   * LANES_PER_DIM (4): within a workgroup, 4 threads cooperate on each
//     dim, striding the position loop by 4 and reducing via shared memory
//     at the end. Each thread does 1/4 of the position work.
//
// Together: ~4-8× more workgroups, ~4× less per-thread work → ~4× speedup
// expected on parallelism-limited hardware (which M1 clearly was —
// fewer than 2048 threads in flight per pass, vs the thousands it wants).

/*@POLAR_CONFIG@*/

const DIM_TILE: u32 = 64u;
const LANES_PER_DIM: u32 = 4u;
const WG_SIZE: u32 = 256u;  // DIM_TILE × LANES_PER_DIM

struct Params {
  dim: u32,
  num_positions: u32,
  polar_words_per_pos: u32,
  qjl_words_per_pos: u32,
  window_start: u32,
  pos_stride: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> v_polar: array<u32>;
@group(0) @binding(3) var<storage, read> v_qjl: array<u32>;
@group(0) @binding(4) var<storage, read> v_max_r: array<f32>;
@group(0) @binding(5) var<storage, read> v_gamma: array<f32>;
// Temp outputs for Phase 2. Both [head × dim] f32 arrays.
@group(0) @binding(6) var<storage, read_write> polar_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> qjl_sum_out: array<f32>;

// Reduction scratch: 256 slots, one per thread. The 4 lanes of each dim
// write into tid, tid+1, tid+2, tid+3 (since their tid values are
// consecutive — see index math in the kernel).
var<workgroup> sh_red: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn weighted_sum_p1(
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let tid = lid.x;
  let head = wid.x;
  let dim_tile = wid.y;
  let dim = params.dim;
  let num_pos = params.num_positions;

  // Thread layout inside the workgroup: 64 dims × 4 lanes per dim.
  // Using lane-major order so a dim's 4 lanes sit at consecutive tids,
  // which makes the sh_red reduction a trivial `sh_red[base..base+3]` sum.
  let dim_in_tile = tid / LANES_PER_DIM;  // 0..63
  let lane = tid % LANES_PER_DIM;           // 0..3
  let d = dim_tile * DIM_TILE + dim_in_tile;

  // Thread-local accumulators. If this thread's dim is out of range the
  // accumulators stay zero — the reduction still runs harmlessly.
  var polar = 0.0f;
  var qjl_sum = 0.0f;

  if (d < dim) {
    let pair_idx = d / 2u;
    let is_cos = (d % 2u) == 0u;
    let bit_pos = pair_idx * POLAR_PAIR_BITS;
    let polar_word_off = bit_pos / 32u;
    let polar_bit_off = bit_pos % 32u;
    let polar_spans_word = polar_bit_off + POLAR_PAIR_BITS > 32u;
    let shift_hi = 32u - polar_bit_off;
    let qjl_word_off = d / QJL_PER_WORD;
    let qjl_bit_off = (d % QJL_PER_WORD) * QJL_BITS;

    let pos_stride = params.pos_stride;
    let polar_row0 = polar_word_off * pos_stride;
    let polar_row1 = (polar_word_off + 1u) * pos_stride;
    let qjl_row = qjl_word_off * pos_stride;

    let score_base = head * num_pos;

    // Stride the position loop by LANES_PER_DIM: lane 0 hits positions
    // window_start, window_start+4, ...; lane 1 hits window_start+1,
    // window_start+5, ...; etc. 1/4 the iterations per thread.
    for (var pos = params.window_start + lane; pos < num_pos; pos += LANES_PER_DIM) {
      let w = weights[score_base + pos];
      if (abs(w) >= 1e-3) {
        let max_r = v_max_r[pos];
        let gamma = v_gamma[pos];

        var comb: u32;
        if (polar_spans_word) {
          comb = ((v_polar[polar_row0 + pos] >> polar_bit_off) | (v_polar[polar_row1 + pos] << shift_hi)) & POLAR_PAIR_MASK;
        } else {
          comb = (v_polar[polar_row0 + pos] >> polar_bit_off) & POLAR_PAIR_MASK;
        }
        let r_scaled = f32(comb >> POLAR_ANGLE_BITS) / POLAR_RADIUS_LEVELS_F * max_r;
        let bucket = comb & POLAR_ANGLE_MASK;
        let wr = w * r_scaled;
        if (is_cos) { polar += wr * BUCKET_COS[bucket]; }
        else { polar += wr * BUCKET_SIN[bucket]; }

        let q_idx = (v_qjl[qjl_row + pos] >> qjl_bit_off) & QJL_MASK;
        qjl_sum += w * gamma * QJL_LEVELS[q_idx];
      }
    }
  }

  // Reduce 4 lanes per dim via shared memory. Lane 0 of each dim writes
  // the final value to polar_out / qjl_sum_out.
  let base_tid = dim_in_tile * LANES_PER_DIM;

  sh_red[tid] = polar;
  workgroupBarrier();
  if (lane == 0u && d < dim) {
    let total = sh_red[base_tid] + sh_red[base_tid + 1u] + sh_red[base_tid + 2u] + sh_red[base_tid + 3u];
    polar_out[head * dim + d] = total;
  }
  workgroupBarrier();

  sh_red[tid] = qjl_sum;
  workgroupBarrier();
  if (lane == 0u && d < dim) {
    let total = sh_red[base_tid] + sh_red[base_tid + 1u] + sh_red[base_tid + 2u] + sh_red[base_tid + 3u];
    qjl_sum_out[head * dim + d] = total;
  }
}
