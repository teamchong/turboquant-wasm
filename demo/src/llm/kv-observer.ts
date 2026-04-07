import { TurboQuant, TQStream } from "turboquant-wasm";
import type { TextGenerationPipeline } from "@huggingface/transformers";

export interface KVStats {
  contextLength: number;
  compressedBytes: number;
  uncompressedBytes: number;
  ratio: number;
  layerCount: number;
  encodeTimeMs: number;
}

export class KVCompressor {
  private streams: Map<string, TQStream> = new Map();
  private tqInstances: Map<number, TurboQuant> = new Map();
  private encodeTimeMs = 0;
  private appendTimeMs = 0;
  private replaceTimeMs = 0;
  private callCount = 0;

  async install(gen: TextGenerationPipeline): Promise<boolean> {
    const model = (gen as any).model;
    if (!model?._update_model_kwargs_for_generation) return false;

    const original = model._update_model_kwargs_for_generation.bind(model);
    const self = this;

    model._update_model_kwargs_for_generation = function (args: any) {
      const result = original(args);
      const cache = result.past_key_values;
      if (cache) self.compressAndReplace(cache);
      return result;
    };

    return true;
  }

  private compressAndReplace(cache: Record<string, any>): void {
    const t0 = performance.now();

    for (const [name, tensor] of Object.entries(cache)) {
      if (!name.startsWith("past_key_values") || tensor.location !== "cpu") continue;

      const data = tensor.data as Float32Array;
      const dims = tensor.dims as number[];
      if (dims.length !== 4) continue;

      const [, , seqLen, headDim] = dims;

      let stream = this.streams.get(name);
      if (!stream) {
        const tq = this.getOrCreateTQ(headDim);
        stream = tq.createStream(128);
        this.streams.set(name, stream);
      }

      // Encode new positions and replace tensor data with TQ round-tripped values
      while (stream.length < seqLen) {
        const pos = stream.length;
        const offset = pos * headDim;
        const vector = data.subarray(offset, offset + headDim);

        const ta = performance.now();
        stream.append(vector);
        this.appendTimeMs += performance.now() - ta;

        const decoded = stream.getDecompressed(pos, pos + 1);
        data.set(decoded, offset);
      }
    }

    this.encodeTimeMs += performance.now() - t0;
    this.callCount++;
    if (this.callCount % 10 === 0) {
      const total = this.appendTimeMs + this.replaceTimeMs;
      console.log(`TQStream step=${this.callCount}: append=${this.appendTimeMs.toFixed(0)}ms replace=${this.replaceTimeMs.toFixed(0)}ms total=${total.toFixed(0)}ms (${(total / this.callCount).toFixed(1)}ms/step)`);
    }
  }

  private getOrCreateTQ(headDim: number): TurboQuant {
    let tq = this.tqInstances.get(headDim);
    if (!tq) throw new Error(`TQ for dim=${headDim} not initialized. Call waitForInit() first.`);
    return tq;
  }

  async waitForInit(): Promise<void> {
    for (const dim of [256, 512]) {
      if (!this.tqInstances.has(dim)) {
        const tq = await TurboQuant.init({ dim, seed: 42 });
        this.tqInstances.set(dim, tq);
      }
    }
  }

  getStats(): KVStats {
    let compressed = 0;
    let uncompressed = 0;
    let maxLen = 0;

    for (const [, stream] of this.streams) {
      compressed += stream.length * stream.bytesPerVector;
      uncompressed += stream.length * stream.dim * 4;
      maxLen = Math.max(maxLen, stream.length);
    }

    return {
      contextLength: maxLen,
      compressedBytes: compressed,
      uncompressedBytes: uncompressed,
      ratio: uncompressed > 0 ? uncompressed / compressed : 0,
      layerCount: this.streams.size,
      encodeTimeMs: this.encodeTimeMs,
    };
  }

  destroy(): void {
    for (const stream of this.streams.values()) stream.destroy();
    this.streams.clear();
    for (const tq of this.tqInstances.values()) tq.destroy();
    this.tqInstances.clear();
  }
}
