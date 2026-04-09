/**
 * TQ KV Cache Integration — shared state between kvcache.cc and sdpa.cc.
 *
 * KVCache creates TQ streams (one per layer × KV head × {key, value}).
 * SDPA looks up streams by layer index to compute attention directly
 * on compressed data: tq_kv_dot_batch for Q@K^T, decode-and-accumulate for V.
 *
 * Memory savings: float32 → ~3 bits/dim ≈ 5x compression.
 * For Gemma 4 E4B (head_dim=256): 1024 bytes/position → ~210 bytes/position.
 */
#pragma once
#include <cstdint>

// TQ bridge C-ABI functions (defined in tq_bridge.zig, compiled to same WASM)
extern "C" {
  int32_t tq_kv_create(uint32_t head_dim, uint32_t max_positions);
  void    tq_kv_destroy(int32_t stream_id);
  int32_t tq_kv_append(int32_t stream_id, const float* data_ptr, uint32_t dim);
  int32_t tq_kv_dot_batch(int32_t stream_id, const float* query_ptr,
                           uint32_t dim, float* out_scores, uint32_t max_scores);
  int32_t tq_kv_decode_position(int32_t stream_id, uint32_t position,
                                 float* out_ptr, uint32_t dim);
  uint32_t tq_kv_length(int32_t stream_id);
  uint32_t tq_kv_compressed_size(int32_t stream_id);
}

// Shared state between KVCache and SDPA ops.
// Single-threaded WASM — global state is safe.
struct TQKVRegistry {
  static constexpr int kMaxLayers = 64;
  static constexpr int kMaxKVHeads = 16;

  // Stream IDs: [layer][kv_head]. -1 = not initialized.
  int32_t k_streams[kMaxLayers][kMaxKVHeads];
  int32_t v_streams[kMaxLayers][kMaxKVHeads];
  int num_kv_heads = 0;
  int head_dim = 0;
  bool initialized = false;

  void reset() {
    for (int l = 0; l < kMaxLayers; ++l) {
      for (int h = 0; h < kMaxKVHeads; ++h) {
        if (k_streams[l][h] >= 0) tq_kv_destroy(k_streams[l][h]);
        if (v_streams[l][h] >= 0) tq_kv_destroy(v_streams[l][h]);
        k_streams[l][h] = -1;
        v_streams[l][h] = -1;
      }
    }
    initialized = false;
  }

  void init() {
    for (int l = 0; l < kMaxLayers; ++l) {
      for (int h = 0; h < kMaxKVHeads; ++h) {
        k_streams[l][h] = -1;
        v_streams[l][h] = -1;
      }
    }
  }
};

// Global instance — accessed by both kvcache.cc and sdpa.cc
inline TQKVRegistry& tq_kv_registry() {
  static TQKVRegistry reg = []() {
    TQKVRegistry r;
    r.init();
    return r;
  }();
  return reg;
}
