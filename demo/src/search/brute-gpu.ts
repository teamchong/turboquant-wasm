/**
 * BruteGpuIndex — WebGPU brute-force dot product on raw f32 vectors.
 * Fair baseline comparison: same GPU dispatch overhead as TQGpuIndex.
 * Demo-only — not part of the npm package.
 */

const SHADER_SRC = /* wgsl */ `
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
`;

export class BruteGpuIndex {
  #device: GPUDevice;
  #pipeline: GPUComputePipeline;
  #bindGroup: GPUBindGroup;
  #queryBuf: GPUBuffer;
  #configBuf: GPUBuffer;
  #scoresBuf: GPUBuffer;
  #stagingBuf: GPUBuffer;
  #vecBuf: GPUBuffer;
  #numVectors: number;
  #resultBuf: Float32Array;

  private constructor(
    device: GPUDevice, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup,
    queryBuf: GPUBuffer, configBuf: GPUBuffer,
    scoresBuf: GPUBuffer, stagingBuf: GPUBuffer,
    vecBuf: GPUBuffer,
    numVectors: number,
  ) {
    this.#device = device;
    this.#pipeline = pipeline;
    this.#bindGroup = bindGroup;
    this.#queryBuf = queryBuf;
    this.#configBuf = configBuf;
    this.#scoresBuf = scoresBuf;
    this.#stagingBuf = stagingBuf;
    this.#vecBuf = vecBuf;
    this.#numVectors = numVectors;
    this.#resultBuf = new Float32Array(numVectors);
  }

  static async create(
    device: GPUDevice, rawVectors: Float32Array, dim: number,
  ): Promise<BruteGpuIndex> {
    if (dim > 1024) throw new Error(`BruteGpuIndex: dim ${dim} exceeds shader MAX_DIM (1024)`);
    const numVectors = rawVectors.length / dim;

    const vecBuf = device.createBuffer({
      size: rawVectors.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vecBuf, 0, rawVectors.buffer);

    const queryBuf = device.createBuffer({
      size: dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const configBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const scoresBuf = device.createBuffer({
      size: numVectors * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stagingBuf = device.createBuffer({
      size: numVectors * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const shaderModule = device.createShaderModule({ code: SHADER_SRC });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: vecBuf } },
        { binding: 1, resource: { buffer: queryBuf } },
        { binding: 2, resource: { buffer: configBuf } },
        { binding: 3, resource: { buffer: scoresBuf } },
      ],
    });

    const configData = new Uint32Array([numVectors, dim, 0, 0]);
    device.queue.writeBuffer(configBuf, 0, configData.buffer);

    return new BruteGpuIndex(
      device, pipeline, bindGroup, queryBuf, configBuf, scoresBuf, stagingBuf, vecBuf, numVectors,
    );
  }

  async dotBatch(query: Float32Array): Promise<Float32Array> {
    this.#device.queue.writeBuffer(this.#queryBuf, 0, query.buffer);

    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.#numVectors / 256));
    pass.end();

    encoder.copyBufferToBuffer(this.#scoresBuf, 0, this.#stagingBuf, 0, this.#numVectors * 4);
    this.#device.queue.submit([encoder.finish()]);

    await this.#stagingBuf.mapAsync(GPUMapMode.READ);
    this.#resultBuf.set(new Float32Array(this.#stagingBuf.getMappedRange()));
    this.#stagingBuf.unmap();

    return this.#resultBuf;
  }

  destroy(): void {
    this.#queryBuf.destroy();
    this.#configBuf.destroy();
    this.#scoresBuf.destroy();
    this.#stagingBuf.destroy();
    this.#vecBuf.destroy();
    this.#device.destroy();
  }
}
