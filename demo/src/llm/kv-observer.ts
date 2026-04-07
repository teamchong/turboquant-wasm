import { TurboQuant } from "turboquant-wasm";
import type { TextGenerationPipeline } from "@huggingface/transformers";

export interface KVStats {
  contextLength: number;
  compressedBytes: number;
  uncompressedBytes: number;
  ratio: number;
  layerCount: number;
  encodeTimeMs: number;
}

interface LayerCache {
  compressed: Uint8Array;
  length: number;
  headDim: number;
}

export class KVObserver {
  private tqInstances: Map<number, TurboQuant> = new Map();
  private layers: Map<string, LayerCache> = new Map();
  private bytesPerVector: Map<number, number> = new Map();
  private capacity = 128;
  private encodeTimeMs = 0;

  async install(gen: TextGenerationPipeline): Promise<boolean> {
    const model = (gen as any).model;
    if (!model?._update_model_kwargs_for_generation) return false;

    const original = model._update_model_kwargs_for_generation.bind(model);
    const self = this;

    model._update_model_kwargs_for_generation = function (args: any) {
      const result = original(args);
      const cache = result.past_key_values;
      if (cache) self.onCacheUpdate(cache);
      return result;
    };

    return true;
  }

  private async onCacheUpdate(cache: Record<string, any>): Promise<void> {
    const t0 = performance.now();

    for (const [name, tensor] of Object.entries(cache)) {
      if (!name.startsWith("past_key_values") || tensor.location !== "cpu") continue;

      const data = tensor.data as Float32Array;
      const dims = tensor.dims as number[];
      if (dims.length !== 4) continue;

      const [, , seqLen, headDim] = dims;
      let layer = this.layers.get(name);

      if (!layer) {
        layer = {
          compressed: new Uint8Array(this.capacity * this.estimateBytesPerVector(headDim)),
          length: 0,
          headDim,
        };
        this.layers.set(name, layer);
      }

      // Only encode new positions (delta)
      while (layer.length < seqLen) {
        const pos = layer.length;
        const offset = pos * headDim;
        const vector = data.subarray(offset, offset + headDim);

        let tq = this.tqInstances.get(headDim);
        if (!tq) {
          // Init on demand for unexpected dimensions
          if (!this.tqInitPromises.has(headDim)) {
            this.tqInitPromises.set(
              headDim,
              TurboQuant.init({ dim: headDim, seed: 42 }).then((t) => {
                this.tqInstances.set(headDim, t);
                return t;
              }),
            );
          }
          tq = await this.tqInitPromises.get(headDim)!;
        }

        const encoded = tq.encode(vector);
        const bpv = encoded.byteLength;

        // Grow if needed
        if ((pos + 1) * bpv > layer.compressed.byteLength) {
          const newCap = Math.max(this.capacity * 2, (pos + 1) * 2);
          const newBuf = new Uint8Array(newCap * bpv);
          newBuf.set(layer.compressed.subarray(0, pos * bpv));
          layer.compressed = newBuf;
          this.capacity = newCap;
        }

        layer.compressed.set(encoded, pos * bpv);
        this.bytesPerVector.set(headDim, bpv);
        layer.length = pos + 1;
      }
    }

    this.encodeTimeMs = performance.now() - t0;
  }

  private tqInitPromises: Map<number, Promise<TurboQuant>> = new Map();

  async waitForInit(): Promise<void> {
    // Pre-init TQ for known dimensions
    const dims = [256, 512];
    await Promise.all(dims.map((d) => {
      if (!this.tqInitPromises.has(d)) {
        this.tqInitPromises.set(
          d,
          TurboQuant.init({ dim: d, seed: 42 }).then((tq) => {
            this.tqInstances.set(d, tq);
            return tq;
          }),
        );
      }
      return this.tqInitPromises.get(d)!;
    }));
  }

  private estimateBytesPerVector(headDim: number): number {
    return this.bytesPerVector.get(headDim) ?? Math.ceil(headDim * 4.5 / 8) + 22;
  }

  getStats(): KVStats {
    let compressed = 0;
    let uncompressed = 0;
    let maxLen = 0;

    for (const [, layer] of this.layers) {
      const bpv = this.bytesPerVector.get(layer.headDim) ?? 0;
      compressed += layer.length * bpv;
      uncompressed += layer.length * layer.headDim * 4;
      maxLen = Math.max(maxLen, layer.length);
    }

    return {
      contextLength: maxLen,
      compressedBytes: compressed,
      uncompressedBytes: uncompressed,
      ratio: uncompressed > 0 ? uncompressed / compressed : 0,
      layerCount: this.layers.size,
      encodeTimeMs: this.encodeTimeMs,
    };
  }

  serialize(): Uint8Array {
    const entries: Array<{ name: string; headDim: number; length: number; data: Uint8Array }> = [];
    let totalSize = 4; // entry count

    for (const [name, layer] of this.layers) {
      const bpv = this.bytesPerVector.get(layer.headDim) ?? 0;
      const dataSlice = layer.compressed.subarray(0, layer.length * bpv);
      entries.push({ name, headDim: layer.headDim, length: layer.length, data: dataSlice });
      totalSize += 4 + name.length + 4 + 4 + dataSlice.byteLength; // nameLen + name + headDim + length + data
    }

    const buf = new Uint8Array(totalSize);
    const view = new DataView(buf.buffer);
    let offset = 0;

    view.setUint32(offset, entries.length, true); offset += 4;
    for (const entry of entries) {
      view.setUint32(offset, entry.name.length, true); offset += 4;
      for (let i = 0; i < entry.name.length; i++) buf[offset++] = entry.name.charCodeAt(i);
      view.setUint32(offset, entry.headDim, true); offset += 4;
      view.setUint32(offset, entry.length, true); offset += 4;
      buf.set(entry.data, offset); offset += entry.data.byteLength;
    }

    return buf;
  }

  destroy(): void {
    for (const tq of this.tqInstances.values()) tq.destroy();
    this.tqInstances.clear();
    this.layers.clear();
  }
}
