import { TurboQuant, TQStream } from "turboquant-wasm";
import { Tensor, type TextGenerationPipeline } from "@huggingface/transformers";

export interface KVStats {
  contextLength: number;
  compressedBytes: number;
  uncompressedBytes: number;
  ratio: number;
  layerCount: number;
}

/**
 * TQ-compressed KV cache that replaces transformers.js DynamicCache.
 *
 * Uses a Proxy: when ONNX writes KV tensors, they're compressed into TQStream.
 * When ONNX reads KV tensors, they're decompressed on-demand from TQStream.
 * No full float tensors are stored persistently — only compressed bytes.
 */
function createCompressedCache(
  streams: Map<string, TQStream>,
  tqInstances: Map<number, TurboQuant>,
): Record<string, any> {
  const meta: Map<string, { batch: number; heads: number; headDim: number }> = new Map();

  return new Proxy(Object.create(null), {
    set(_target, prop: string, value: any) {
      if (!prop.startsWith("past_key_values") || !(value instanceof Tensor)) {
        _target[prop] = value;
        return true;
      }

      const tensor = value;
      if (tensor.location !== "cpu") {
        _target[prop] = value;
        return true;
      }

      const dims = tensor.dims as number[];
      if (dims.length !== 4) {
        _target[prop] = value;
        return true;
      }

      const [batch, heads, seqLen, headDim] = dims;
      const data = tensor.data as Float32Array;

      let stream = streams.get(prop);
      if (!stream) {
        const tq = tqInstances.get(headDim);
        if (!tq) { _target[prop] = value; return true; }
        stream = tq.createStream(128);
        streams.set(prop, stream);
      }

      // Streaming compress + decode — O(1) per new position
      while (stream.length < seqLen) {
        const pos = stream.length;
        stream.append(data.subarray(pos * headDim, (pos + 1) * headDim));
      }

      meta.set(prop, { batch, heads, headDim });

      // Don't store the tensor. Compressed data in TQStream is the store.
      return true;
    },

    get(_target, prop: string) {
      // Non-KV properties (methods like get_seq_length, dispose, etc.)
      if (!prop.startsWith("past_key_values")) {
        if (prop === "get_seq_length") {
          return () => {
            for (const stream of streams.values()) {
              if (stream.length > 0) return stream.length;
            }
            return 0;
          };
        }
        if (prop === "dispose") {
          return async () => {}; // Nothing to dispose — no GPU tensors
        }
        return _target[prop];
      }

      const stream = streams.get(prop);
      if (!stream || stream.length === 0) return undefined;

      const m = meta.get(prop);
      if (!m) return undefined;

      // TQStream's decompressed buffer IS the KV store — view directly, no copy
      const seqLen = stream.length;
      const view = stream.getDecompressed(0, seqLen);
      return new Tensor("float32", view, [m.batch, m.heads, seqLen, m.headDim]);
    },

    has(_target, prop: string) {
      if (prop.startsWith("past_key_values")) {
        return streams.has(prop);
      }
      return prop in _target;
    },

    ownKeys(_target) {
      const keys = Object.keys(_target);
      for (const name of streams.keys()) {
        if (!keys.includes(name)) keys.push(name);
      }
      return keys;
    },

    getOwnPropertyDescriptor(_target, prop: string) {
      if (streams.has(prop)) {
        return { configurable: true, enumerable: true, writable: true };
      }
      return Object.getOwnPropertyDescriptor(_target, prop);
    },
  });
}

export class KVCompressor {
  private streams: Map<string, TQStream> = new Map();
  private tqInstances: Map<number, TurboQuant> = new Map();
  private callCount = 0;

  async install(gen: TextGenerationPipeline): Promise<boolean> {
    const model = (gen as any).model;
    if (!model?.getPastKeyValues) return false;

    const originalGetPKV = model.getPastKeyValues.bind(model);
    const self = this;

    model.getPastKeyValues = function (decoderResults: any, pastKeyValues: any, disposeEncoderPKVs: boolean) {
      // Let original extract present_* → past_key_values.* naming
      const cache = originalGetPKV(decoderResults, pastKeyValues, disposeEncoderPKVs);

      // First call: create compressed cache and populate from ONNX's initial output
      if (self.callCount === 0) {
        const compressed = createCompressedCache(self.streams, self.tqInstances);
        for (const [name, tensor] of Object.entries(cache) as [string, any][]) {
          compressed[name] = tensor;
        }
        self.callCount++;
        return compressed;
      }

      // Subsequent calls: cache IS already the proxy. Write new tensors into it.
      // getPastKeyValues creates a new DynamicCache each time, but we need to
      // write the new present_* tensors into our existing proxy.
      for (const [name, tensor] of Object.entries(cache) as [string, any][]) {
        if (name.startsWith("past_key_values")) {
          pastKeyValues[name] = tensor; // Write into the proxy (triggers compress)
        }
      }

      self.callCount++;
      if (self.callCount % 10 === 0) {
        const s = self.getStats();
        console.log(`TQ KV step=${self.callCount}: ${s.compressedBytes} compressed, ${s.uncompressedBytes} uncompressed, ${s.ratio.toFixed(1)}x`);
      }

      return pastKeyValues; // Return the proxy, not the new DynamicCache
    };

    return true;
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
    };
  }

  destroy(): void {
    for (const stream of this.streams.values()) stream.destroy();
    this.streams.clear();
    for (const tq of this.tqInstances.values()) tq.destroy();
    this.tqInstances.clear();
  }
}
