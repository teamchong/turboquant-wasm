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
 * Uses a Proxy: when ONNX writes CPU KV tensors, they're compressed into TQStream.
 * When ONNX reads KV tensors, they're decompressed on-demand from TQStream.
 * GPU tensors pass through unmodified (compression happens on CPU data only).
 */
function createCompressedCache(
  streams: Map<string, TQStream>,
  tqInstances: Map<number, TurboQuant>,
): Record<string, any> {
  const meta: Map<string, { batch: number; heads: number; headDim: number }> = new Map();
  const target = Object.create(null);

  return new Proxy(target, {
    set(_target, prop: string, value: any) {
      if (!prop.startsWith("past_key_values") || !(value instanceof Tensor)) {
        _target[prop] = value;
        return true;
      }

      const tensor = value;
      const dims = tensor.dims as number[];

      // Only compress CPU f32 tensors with 4D shape [batch, heads, seq, headDim]
      if (tensor.location !== "cpu" || dims.length !== 4) {
        _target[prop] = value;
        return true;
      }

      const [batch, heads, seqLen, headDim] = dims;
      const data = tensor.data as Float32Array;

      // Check if TQ supports this headDim
      const tq = tqInstances.get(headDim);
      if (!tq) {
        _target[prop] = value;
        return true;
      }

      let stream = streams.get(prop);
      if (!stream) {
        stream = tq.createStream(128);
        streams.set(prop, stream);
      }

      // Compress new positions only
      while (stream.length < seqLen) {
        const pos = stream.length;
        stream.append(data.subarray(pos * headDim, (pos + 1) * headDim));
      }

      meta.set(prop, { batch, heads, headDim });
      // Remove from _target if we now own this key in streams
      delete _target[prop];
      return true;
    },

    get(_target, prop: string) {
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
          return async () => {};
        }
        return _target[prop];
      }

      // Check streams first (compressed path)
      const stream = streams.get(prop);
      if (stream && stream.length > 0) {
        const m = meta.get(prop);
        if (!m) return undefined;
        const seqLen = stream.length;
        const data = new Float32Array(seqLen * m.headDim);
        for (let i = 0; i < seqLen; i++) {
          data.set(stream.decodePosition(i), i * m.headDim);
        }
        return new Tensor("float32", data, [m.batch, m.heads, seqLen, m.headDim]);
      }

      // Fall back to raw tensor (GPU path or unsupported dim)
      return _target[prop];
    },

    has(_target, prop: string) {
      if (prop.startsWith("past_key_values")) {
        return streams.has(prop) || prop in _target;
      }
      return prop in _target;
    },

    ownKeys(_target) {
      const keys = new Set(Object.keys(_target));
      for (const name of streams.keys()) keys.add(name);
      return [...keys];
    },

    getOwnPropertyDescriptor(_target, prop: string) {
      if (streams.has(prop) || prop in _target) {
        return { configurable: true, enumerable: true, writable: true };
      }
      return undefined;
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

    let proxy: Record<string, any> | null = null;

    model.getPastKeyValues = function (decoderResults: any, pastKeyValues: any, disposeEncoderPKVs: boolean) {
      const cache = originalGetPKV(decoderResults, pastKeyValues, disposeEncoderPKVs);

      // New generation or first call — create fresh proxy
      if (!pastKeyValues || pastKeyValues !== proxy) {
        // Reset streams for new generation
        for (const stream of self.streams.values()) stream.rewind(0);
        self.streams.clear();

        proxy = createCompressedCache(self.streams, self.tqInstances);
        for (const [name, tensor] of Object.entries(cache) as [string, any][]) {
          proxy[name] = tensor;
        }
        self.callCount++;
        return proxy;
      }

      // Continuing generation — write new KV into existing proxy
      for (const [name, tensor] of Object.entries(cache) as [string, any][]) {
        if (name.startsWith("past_key_values")) {
          proxy[name] = tensor;
        }
      }

      self.callCount++;
      return proxy;
    };

    return true;
  }

  async waitForInit(): Promise<void> {
    // Common head dimensions for LLMs
    for (const dim of [64, 128, 256, 512]) {
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
