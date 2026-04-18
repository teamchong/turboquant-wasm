// Logit mask for constrained decoding. Zeros out (sets to -1e30) logits whose
// token id is NOT allowed by the current grammar bitmap. Runs between
// projectLogits and gpuArgmax so masked tokens can never win the argmax.
//
// Bitmap layout: one bit per vocab entry, packed 32 tokens per u32.
//   allowed[i] = (mask[i / 32u] >> (i % 32u)) & 1u
// A bit of 1 means "allowed", 0 means "masked".
//
// Three bitmaps are preuploaded at init (FREE, EXPECT_STRING, IN_STRING); the
// `mask_offset` uniform picks which bitmap starts at that u32 offset.

struct Params {
  count: u32,
  mask_offset: u32,   // starting u32 index within `mask` for this bitmap
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> logits: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;

@compute @workgroup_size(256)
fn mask_logits(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  let word = mask[params.mask_offset + (idx >> 5u)];
  let bit = (word >> (idx & 31u)) & 1u;
  if (bit == 0u) {
    logits[idx] = -1e30;
  }
}
