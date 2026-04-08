// Brute-force dot product: query × N raw f32 vectors.
// One thread per vector, query in shared memory.

struct Config {
  num_vectors: u32,
  dim: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

const WG_SIZE: u32 = 256u;
const MAX_DIM: u32 = 1024u;

var<workgroup> sq: array<f32, MAX_DIM>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let dim = config.dim;
  let tid = lid.x;

  var qi = tid;
  while (qi < dim) {
    sq[qi] = query[qi];
    qi += WG_SIZE;
  }
  workgroupBarrier();

  let vec_id = gid.x;
  if (vec_id >= config.num_vectors) { return; }

  let base = vec_id * dim;
  var dot: f32 = 0.0;
  for (var d: u32 = 0u; d < dim; d += 1u) {
    dot += sq[d] * vectors[base + d];
  }

  scores[vec_id] = dot;
}
