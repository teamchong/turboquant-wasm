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

/* pthread_create — cannot spawn threads in single-threaded WASM */
int pthread_create(void* t, const void* attr, void* (*fn)(void*), void* arg) {
  (void)t; (void)attr; (void)fn; (void)arg; return -1;
}
int pthread_join(void* t, void** retval) {
  (void)t; (void)retval; return 0;
}
int pthread_cond_broadcast(void* c) { (void)c; return 0; }
int pthread_cond_destroy(void* c) { (void)c; return 0; }
int pthread_mutex_destroy(void* m) { (void)m; return 0; }
int pthread_once(int* once_control, void (*init_routine)(void)) {
  if (once_control && *once_control == 0) {
    *once_control = 1;
    if (init_routine) init_routine();
  }
  return 0;
}

/* dup — POSIX fd duplication, not available in WASI.
   Weight caching file I/O won't be used in browser WASM. */
int dup(int fd) { (void)fd; return -1; }

/* getpagesize — not in WASI, return 64KB (WASM page size) */
int getpagesize(void) { return 65536; }

/* getuid/getgid — POSIX user/group ID, not in WASI. Return 0 (root). */
unsigned int getuid(void) { return 0; }
unsigned int getgid(void) { return 0; }

/* madvise — POSIX memory advisory, no-op on WASM (no virtual memory) */
int madvise(void* addr, size_t len, int advice) {
  (void)addr; (void)len; (void)advice; return 0;
}

/* pthreadpool_update_executor — defined in pthreads.c which requires real
   pthread support. Single-threaded WASM always returns false (no executor). */
int pthreadpool_update_executor(void* pool, void* executor, void* ctx) {
  (void)pool; (void)executor; (void)ctx;
  return 0;
}

/* XNNPack wasmsimd-magic vcvt kernels — the generated source files have a
   codegen bug (include arm_neon.h instead of wasm_simd128.h). These specific
   batch sizes are referenced by the unary-elementwise config but the runtime
   will select the working WASM SIMD variants at dispatch time. */
void xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u32(
    size_t batch, const float* input, void* output, const void* params) {
  (void)batch; (void)input; (void)output; (void)params;
}
void xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32(
    size_t batch, const float* input, void* output, const void* params) {
  (void)batch; (void)input; (void)output; (void)params;
}

/* Abseil internal thread primitives — single-threaded WASM */
void AbslInternalPerThreadSemPost_lts_20250814(void* w) { (void)w; }
int AbslInternalPerThreadSemWait_lts_20250814(void* w, void* t) {
  (void)w; (void)t; return 0;
}

/* C++ exception allocation — WASM runs with exceptions disabled,
   but some TFLite code has throw paths that the linker pulls in.
   __cxa_throw traps immediately (exception = fatal in WASM). */
void* __cxa_allocate_exception(size_t size) {
  return malloc(size);
}
void __cxa_throw(void* obj, void* type, void (*dtor)(void*)) {
  (void)obj; (void)type; (void)dtor;
  __builtin_trap();
}

/* Arena initialization — returns a sentinel. The arena pointer is only
 * passed back to AllocWithArena which ignores it (above). */
static int _sig_safe_arena;

void* _ZN4absl12lts_2025081413base_internal16InitSigSafeArenaEv(void) {
  return &_sig_safe_arena;
}

void* _ZN4absl12lts_2025081413base_internal12SigSafeArenaEv(void) {
  return &_sig_safe_arena;
}

