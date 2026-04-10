// TurboQuant KV Cache API — called from ORT's attention kernel.
// These functions are implemented in Zig (tq_bridge.zig) and linked
// into the same WASM binary. Zero copy, shared linear memory.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create a compressed KV stream for one attention layer.
// Returns stream_id (>=0) or -1 on failure.
int32_t tq_kv_create(uint32_t head_dim, uint32_t max_positions);

// Destroy a KV stream.
void tq_kv_destroy(int32_t stream_id);

// Append a K or V vector (compress and store). Returns 0 or -1.
int32_t tq_kv_append(int32_t stream_id, const float* data, uint32_t dim);

// Compute dot(query, K[i]) for all stored K vectors.
// Scores written to out_scores. Returns count written.
int32_t tq_kv_dot_batch(int32_t stream_id, const float* query, uint32_t dim,
                        float* out_scores, uint32_t max_scores);

// Decode a single position's vector into out buffer. Returns 0 or -1.
int32_t tq_kv_decode_position(int32_t stream_id, uint32_t position,
                              float* out, uint32_t dim);

// Get number of vectors stored.
uint32_t tq_kv_length(int32_t stream_id);

// Get compressed size in bytes.
uint32_t tq_kv_compressed_size(int32_t stream_id);

#ifdef __cplusplus
}
#endif
