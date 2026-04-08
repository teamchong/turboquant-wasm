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

/* Exported allocator for JS glue (JSEP callbacks need WASM heap access). */
__attribute__((visibility("default")))
void* wasm_malloc(size_t size) { return malloc(size); }

__attribute__((visibility("default")))
void wasm_free(void* ptr) { free(ptr); }

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

/* LowLevelAlloc::Alloc — non-arena variant, same rationale as above. */
void* _ZN4absl12lts_2025081413base_internal13LowLevelAlloc5AllocEm(
    size_t size) {
  return malloc(size);
}

/* ---- pthread (single-threaded WASM, all no-ops returning success) ---- */

int pthread_mutex_init(void* m, const void* attr) {
  (void)m; (void)attr; return 0;
}
int pthread_mutex_lock(void* m) { (void)m; return 0; }
int pthread_mutex_unlock(void* m) { (void)m; return 0; }

int pthread_cond_init(void* c, const void* attr) {
  (void)c; (void)attr; return 0;
}
int pthread_cond_signal(void* c) { (void)c; return 0; }
int pthread_cond_wait(void* c, void* m) { (void)c; (void)m; return 0; }
int pthread_cond_timedwait(void* c, void* m, const void* t) {
  (void)c; (void)m; (void)t; return 0;
}

static void* _tls_values[64];
static int _tls_next_key = 0;

int pthread_key_create(unsigned int* key, void (*dtor)(void*)) {
  (void)dtor;
  if (_tls_next_key >= 64) return -1;
  *key = (unsigned int)_tls_next_key++;
  return 0;
}

int pthread_setspecific(unsigned int key, const void* val) {
  if (key >= 64) return -1;
  _tls_values[key] = (void*)val;
  return 0;
}

void* pthread_getspecific(unsigned int key) {
  if (key >= 64) return 0;
  return _tls_values[key];
}

/* Returns a non-zero "thread id" — single thread in WASM. */
unsigned long pthread_self(void) { return 1; }

/* Arena initialization — returns a sentinel. The arena pointer is only
 * passed back to AllocWithArena which ignores it (above). */
static int _sig_safe_arena;

void* _ZN4absl12lts_2025081413base_internal16InitSigSafeArenaEv(void) {
  return &_sig_safe_arena;
}

void* _ZN4absl12lts_2025081413base_internal12SigSafeArenaEv(void) {
  return &_sig_safe_arena;
}

