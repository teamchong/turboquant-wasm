/**
 * Minimal implementations for 5 symbols not provided by Zig's libc/libc++.
 *
 * Abseil LowLevelAlloc: ORT uses this for thread-safe arena allocation.
 * In single-threaded WASM there's no contention, so malloc/free is correct.
 *
 * ORT LogRuntimeError: Error logging callback. The WASM fd_write shim
 * already routes output to console.log, so this is a no-op by design —
 * ORT logs errors through its own logging infrastructure before calling this.
 */

#include <stdlib.h>
#include <stdint.h>

/* Abseil LowLevelAlloc — arena-based allocator for thread safety.
 * WASM is single-threaded, so arena is unnecessary. malloc/free is the
 * correct implementation for this target. */
void* _ZN4absl12lts_2025081413base_internal13LowLevelAlloc14AllocWithArenaEmPNS2_5ArenaE(
    size_t size, void* arena) {
  (void)arena;
  return malloc(size);
}

void _ZN4absl12lts_2025081413base_internal13LowLevelAlloc4FreeEPv(void* ptr) {
  free(ptr);
}

/* Arena initialization — returns a sentinel. The arena pointer is only
 * passed back to AllocWithArena which ignores it (above). */
static int _sig_safe_arena;

void* _ZN4absl12lts_2025081413base_internal17InitSigSafeArenaEv(void) {
  return &_sig_safe_arena;
}

void* _ZN4absl12lts_2025081413base_internal12SigSafeArenaEv(void) {
  return &_sig_safe_arena;
}

