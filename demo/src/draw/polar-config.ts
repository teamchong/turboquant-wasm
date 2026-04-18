/**
 * TQ polar-format configuration — centralizes bit layout for encode, attention,
 * and weighted-sum shaders so variants can be swapped in one place.
 *
 * `radiusBits + angleBits` = bits per polar pair. `qjlBits` is bits per dim
 * used for the residual JL sketch: 1 = original sign-only QJL, 2 = Lloyd-Max
 * 4-level quantization of the projected residual (multi-level JL).
 */

export interface PolarConfig {
  radiusBits: number;
  angleBits: number;
  qjlBits: 1 | 2;
}

// Active configurations — K_POLAR_CONFIG is used for K encoding and score
// computation; V_POLAR_CONFIG is used for V encoding and the weighted sum.
// V errors flow linearly into the attention output (kqv_out) while K errors
// flow through softmax, so V benefits disproportionately from extra bits.
export const K_POLAR_CONFIG: PolarConfig = { radiusBits: 4, angleBits: 3, qjlBits: 1 };
export const V_POLAR_CONFIG: PolarConfig = { radiusBits: 6, angleBits: 7, qjlBits: 2 };

// Legacy alias for code paths that don't distinguish K from V.
export const POLAR_CONFIG: PolarConfig = K_POLAR_CONFIG;

export function polarPairBits(cfg: PolarConfig = POLAR_CONFIG): number {
  return cfg.radiusBits + cfg.angleBits;
}

/** u32 words needed to store one position's polar payload for a head of `dim`. */
export function polarWordsPerPos(dim: number, cfg: PolarConfig = POLAR_CONFIG): number {
  return Math.ceil((dim / 2) * polarPairBits(cfg) / 32);
}

/** u32 words needed to store one position's QJL payload for a head of `dim`. */
export function qjlWordsPerPos(dim: number, cfg: PolarConfig = POLAR_CONFIG): number {
  return Math.ceil(dim * cfg.qjlBits / 32);
}

/**
 * WGSL const block that shaders inject at the `/*@POLAR_CONFIG@*\/` marker.
 *
 * For the QJL sketch we also emit a `QJL_LEVELS` array and thresholds:
 *   qjlBits=1 → 2 levels {-1, +1}, threshold 0 (sign-only JL, unchanged math)
 *   qjlBits=2 → 4 Lloyd-Max levels for Gaussian, thresholds at ±0.9816σ and 0
 *
 * Both modes share one encoder/decoder path: the encoder computes `projected`,
 * picks an index via `qjl_quantize_idx`, and accumulates `num += projected * LEVEL`
 * and `den += LEVEL * LEVEL`. The decoder looks up `LEVEL[idx]` and multiplies
 * by `gamma = num/den` (for 1-bit, ||LEVEL||^2 = dim by construction, so this
 * collapses to the old sign-only optimal scale).
 */
export function polarConfigWgsl(cfg: PolarConfig = POLAR_CONFIG): string {
  const bucketCount = 1 << cfg.angleBits;
  const radiusLevels = (1 << cfg.radiusBits) - 1;
  const pairBits = polarPairBits(cfg);
  const pairMask = (1 << pairBits) - 1;
  const angleMask = bucketCount - 1;

  // Bucket angles: theta_i = i / (bucketCount - 1) * 2π - π for i in 0..bucketCount.
  // Matches the legacy 8-bucket table (i=0 and i=bucketCount-1 collapse to the same
  // direction). Keeps the refactor bit-similar to the previous 4r+3a baseline.
  const cos: string[] = [];
  const sin: string[] = [];
  for (let i = 0; i < bucketCount; i++) {
    const theta = (i / (bucketCount - 1)) * 2 * Math.PI - Math.PI;
    cos.push(Math.cos(theta).toFixed(10));
    sin.push(Math.sin(theta).toFixed(10));
  }

  // QJL tables. For 1-bit mode the two "levels" are the usual ±1 and the
  // threshold is 0 (sigma unused). For 2-bit mode we use Lloyd-Max optimal
  // constants for a unit-variance Gaussian: thresholds at ±0.9816σ and 0,
  // reconstruction levels at ±0.4528, ±1.5104 (unscaled — σ is folded into
  // the per-vector gamma).
  const qjlLevelCount = 1 << cfg.qjlBits;
  let qjlLevels: number[];
  if (cfg.qjlBits === 1) {
    qjlLevels = [-1.0, 1.0];
  } else {
    qjlLevels = [-1.5104, -0.4528, 0.4528, 1.5104];
  }
  const qjlMask = qjlLevelCount - 1;

  return `
const POLAR_RADIUS_BITS: u32 = ${cfg.radiusBits}u;
const POLAR_ANGLE_BITS: u32 = ${cfg.angleBits}u;
const POLAR_PAIR_BITS: u32 = ${pairBits}u;
const POLAR_RADIUS_LEVELS: u32 = ${radiusLevels}u;
const POLAR_RADIUS_LEVELS_F: f32 = ${radiusLevels}.0;
const POLAR_PAIR_MASK: u32 = ${pairMask}u;
const POLAR_ANGLE_MASK: u32 = ${angleMask}u;
const POLAR_BUCKET_COUNT: u32 = ${bucketCount}u;
const BUCKET_COS = array<f32, ${bucketCount}>(${cos.join(", ")});
const BUCKET_SIN = array<f32, ${bucketCount}>(${sin.join(", ")});

const QJL_BITS: u32 = ${cfg.qjlBits}u;
const QJL_LEVEL_COUNT: u32 = ${qjlLevelCount}u;
const QJL_MASK: u32 = ${qjlMask}u;
const QJL_PER_WORD: u32 = ${32 / cfg.qjlBits}u;
const QJL_LEVELS = array<f32, ${qjlLevelCount}>(${qjlLevels.map(v => v.toFixed(10)).join(", ")});

// Returns the QJL bin index for projected residual value 'p' given per-vector
// sigma (stddev of projected). For 1-bit, sigma is ignored.
fn qjl_quantize_idx(p: f32, sigma: f32) -> u32 {
${
  cfg.qjlBits === 1
    ? "  return select(0u, 1u, p > 0.0);"
    : "  let t = 0.9816 * sigma;\n  if (p < -t) { return 0u; }\n  if (p < 0.0) { return 1u; }\n  if (p < t)  { return 2u; }\n  return 3u;"
}
}
`;
}

export function injectPolarConfig(src: string, cfg: PolarConfig = POLAR_CONFIG): string {
  return src.replace("/*@POLAR_CONFIG@*/", polarConfigWgsl(cfg));
}
