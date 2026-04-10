// Rotate a vector: output = R * input (matrix-vector multiply).
// Used to pre-rotate Q before TQ attention dot product.

struct Params {
  dim: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;       // [dim]
@group(0) @binding(2) var<storage, read> rotation_matrix: array<f32>; // [dim * dim]
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>; // [dim]

@compute @workgroup_size(256)
fn rotate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  let dim = params.dim;
  if (row >= dim) { return; }

  var sum = 0.0f;
  for (var c = 0u; c < dim; c++) {
    sum += rotation_matrix[row * dim + c] * input_vec[c];
  }
  output_vec[row] = sum;
}
