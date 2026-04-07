// Emscripten replacement for Zig WASM build.
// EM_ASM macros replaced with extern function calls provided by JS at instantiation.
#pragma once

#define EMSCRIPTEN_KEEPALIVE __attribute__((used))

// JS-provided functions (imported at WASM instantiation)
#ifdef __cplusplus
extern "C" {
#endif

// JSEP allocator
void* jsep_alloc(size_t size);
size_t jsep_free(void* ptr);

// JSEP data transfer
void jsep_download(const void* src, void* dst, size_t bytes);
void jsep_copy(const void* src, void* dst, size_t bytes, int gpu_to_cpu);

// JSEP kernel lifecycle
void jsep_create_kernel(const char* optype, void* kernel_ptr, const char* attr);
void jsep_release_kernel(void* kernel_ptr);
int jsep_run(void* kernel_ptr, int num_inputs, const void** inputs,
             int num_outputs, void** outputs, const void* attrs);

// JSEP graph capture
void jsep_capture_begin(void);
void jsep_capture_end(void);
void jsep_replay(void);

// Timing
double emscripten_get_now(void);

#ifdef __cplusplus
}
#endif

// EM_ASM macros — route to the extern functions above.
// Each macro usage in ORT is replaced with the corresponding function call.
// The actual JS code in the macro is ignored — our extern fn does the real work.

#define EM_ASM(code, ...) ((void)0)
#define EM_ASM_PTR(code, ...) ((void*)0)
#define EM_ASM_INT(code, ...) (0)

// EM_ASYNC_JS declares an extern function (our shim already provides it above)
#define EM_ASYNC_JS(ret, name, params, code) /* provided by extern decl above */
