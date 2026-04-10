"""
Optimize gemma-4-E2B decoder_model_merged.onnx with ORT graph optimization.
Fuses standard attention ops into GroupQueryAttention/MultiHeadAttention.

Usage:
  python scripts/export_gqa_model.py
"""

import os
import onnxruntime as ort
from huggingface_hub import snapshot_download

MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX"
VARIANT = "q4f16"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-E2B-gqa")


def optimize_onnx(input_path: str, output_path: str):
    """Run ORT graph optimization on a single ONNX file."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # O2
    opts.optimized_model_filepath = output_path

    # Create session just to trigger optimization + save
    print(f"  Optimizing {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    ort.InferenceSession(input_path, opts, providers=["CPUExecutionProvider"])
    print(f"  Saved: {output_path}")


def check_ops(path: str):
    """Print attention-related ops in an ONNX file."""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        ops = set(n.op_type for n in m.graph.node)
        attn = [op for op in ops if "attention" in op.lower()]
        print(f"  Ops in {os.path.basename(path)}: {len(m.graph.node)} nodes, attention ops: {attn or 'none'}")
    except ImportError:
        print("  (install onnx to inspect ops)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download the model files
    print(f"[1/3] Downloading {MODEL_ID} (onnx subfolder)...")
    cache_dir = snapshot_download(
        MODEL_ID,
        allow_patterns=[f"onnx/*{VARIANT}*", "onnx/decoder_model_merged.onnx", "*.json"],
    )
    onnx_dir = os.path.join(cache_dir, "onnx")
    print(f"  Downloaded to: {onnx_dir}")

    # Find the decoder model
    decoder = os.path.join(onnx_dir, f"decoder_model_merged_{VARIANT}.onnx")
    if not os.path.exists(decoder):
        decoder = os.path.join(onnx_dir, "decoder_model_merged.onnx")

    print(f"\n[2/3] Before optimization:")
    check_ops(decoder)

    print(f"\n[3/3] Running ORT O2 graph optimization...")
    output_path = os.path.join(OUTPUT_DIR, f"decoder_model_merged_{VARIANT}_gqa.onnx")
    optimize_onnx(decoder, output_path)

    print(f"\nAfter optimization:")
    check_ops(output_path)


if __name__ == "__main__":
    main()
