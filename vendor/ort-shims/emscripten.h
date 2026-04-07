// Emscripten shim for Zig WASM build
// Replace EM_ASM macros with extern function declarations
#pragma once

#define EMSCRIPTEN_KEEPALIVE __attribute__((used))

// EM_ASM: inline JS execution — replaced with extern fn calls
#define EM_ASM(code, ...) ((void)0)
#define EM_ASM_PTR(code, ...) ((void*)0)
#define EM_ASM_INT(code, ...) (0)

// EM_ASYNC_JS: async JS — needs JSPI replacement
#define EM_ASYNC_JS(ret, name, params, code) \
  extern ret name params;

// Timing
extern double emscripten_get_now(void);
