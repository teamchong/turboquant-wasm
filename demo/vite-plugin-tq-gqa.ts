/**
 * Vite plugin that patches ORT Web's GroupQueryAttention kernel
 * to use TQ compressed KV cache. Model-generic — any model using GQA
 * gets TQ compression automatically.
 *
 * Intercepts the ort.all.mjs bundle during Vite's transform phase and
 * replaces the applyAttention call inside groupQueryAttention with
 * tqApplyAttention, which dispatches TQ WGSL compute shaders.
 */
import type { Plugin } from "vite";
import { resolve } from "path";

export default function tqGqaPlugin(): Plugin {
  const ortAllPath = resolve(__dirname, "node_modules/onnxruntime-web/dist/ort.all.mjs");

  return {
    name: "tq-gqa-patch",
    enforce: "pre",

    // Redirect onnxruntime-web/webgpu to the unminified ort.all.mjs
    // so we can reliably find and replace the GQA function.
    resolveId(source) {
      if (source === "onnxruntime-web/webgpu" || source === "onnxruntime-web") {
        return ortAllPath;
      }
    },

    // Patch the GQA function to call tqApplyAttention instead of applyAttention.
    transform(code, id) {
      if (!id.endsWith("ort.all.mjs")) return;

      // Inject the TQ import at the top of the module
      const tqImport = `import { tqApplyAttention } from "${resolve(__dirname, "src/draw/tq-apply-attention.ts")}";\n`;

      // The applyAttention call in groupQueryAttention looks like:
      //   applyAttention(
      //     context,
      //     Q,
      //     K,
      //     V,
      //     void 0,
      //     void 0,
      //     pastKey,
      //     pastValue,
      //     void 0,
      //     params,
      //     seqLens,
      //     totalSequenceLengthInput
      //   );
      //
      // We replace it with tqApplyAttention which takes the backend from context.

      // Patch both GroupQueryAttention and MultiHeadAttention call sites.
      // GQA signature: applyAttention(context, Q, K, V, void 0, void 0, pastKey, pastValue, void 0, params, seqLens, totalSequenceLengthInput);
      // MHA signature: applyAttention(context, Q, K, V, keyPaddingMask, void 0, pastKey, pastValue, attentionBias, params);
      const gqaPattern = /applyAttention\(\s*context,\s*Q,\s*K,\s*V,\s*void 0,\s*void 0,\s*pastKey,\s*pastValue,\s*void 0,\s*params,\s*seqLens,\s*totalSequenceLengthInput\s*\);/;
      const mhaPattern = /applyAttention\(\s*context,\s*Q,\s*K,\s*V,\s*keyPaddingMask,\s*void 0,\s*pastKey,\s*pastValue,\s*attentionBias,\s*params\s*\);/;

      const replacement = `{
        if (!context.kernelCustomData._tqLayerIdx) {
          context.kernelCustomData._tqLayerIdx = (globalThis.__tqLayerCounter = (globalThis.__tqLayerCounter || 0) + 1);
        }
        tqApplyAttention(context.backend, Q, K, V, context, params, context.kernelCustomData._tqLayerIdx);
      }`;

      let patched = code;
      let patchCount = 0;

      if (gqaPattern.test(patched)) {
        patched = patched.replace(gqaPattern, replacement);
        patchCount++;
        console.log("[tq-gqa-patch] Patched GroupQueryAttention");
      }
      if (mhaPattern.test(patched)) {
        patched = patched.replace(mhaPattern, replacement);
        patchCount++;
        console.log("[tq-gqa-patch] Patched MultiHeadAttention");
      }

      if (patchCount === 0) {
        console.warn("[tq-gqa-patch] Could not find any attention calls to patch — TQ NOT applied");
        return;
      }

      console.log(`[tq-gqa-patch] Replaced ${patchCount} applyAttention call(s) with tqApplyAttention`);
      return tqImport + patched;
    },
  };
}
