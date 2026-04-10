/**
 * LoRA adapter API for WASM — returns NOT_IMPLEMENTED.
 *
 * LoRA adapters require filesystem access which is unavailable in
 * browser WASM. These are part of the OrtApis function pointer table
 * and must have correct signatures for indirect call dispatch.
 *
 * Uses mangled C symbol names to avoid pulling in ort_apis.h
 * (which has extensive SAL annotation dependencies).
 */

#include <stddef.h>
#include <stdint.h>

/* OrtApis::CreateLoraAdapter(const char*, OrtAllocator*, OrtLoraAdapter**) -> OrtStatusPtr */
void* _ZN7OrtApis17CreateLoraAdapterEPKcP12OrtAllocatorPP14OrtLoraAdapter(
    const char* path, void* allocator, void** out) {
  (void)path; (void)allocator; (void)out;
  /* Return non-null = error. ORT interprets any non-null OrtStatus* as failure.
   * We return a sentinel; caller checks via OrtGetErrorCode. */
  static char err[] = "LoRA not available in WASM";
  return err;
}

/* OrtApis::CreateLoraAdapterFromArray(const void*, size_t, OrtAllocator*, OrtLoraAdapter**) -> OrtStatusPtr */
void* _ZN7OrtApis26CreateLoraAdapterFromArrayEPKvmP12OrtAllocatorPP14OrtLoraAdapter(
    const void* bytes, size_t len, void* allocator, void** out) {
  (void)bytes; (void)len; (void)allocator; (void)out;
  static char err[] = "LoRA not available in WASM";
  return err;
}

/* OrtApis::ReleaseLoraAdapter(OrtLoraAdapter*) -> void */
void _ZN7OrtApis18ReleaseLoraAdapterEP14OrtLoraAdapter(void* adapter) {
  (void)adapter;
}
