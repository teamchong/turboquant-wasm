// C++ exception ABI for WASM — exceptions abort since there is no unwinder.
// ORT on WASM (Emscripten) normally uses Asyncify for this.
// In our build, exceptions terminate the program.
#include <cstdlib>
#include <cstdio>

extern "C" {

void* __cxa_allocate_exception(unsigned long) {
  static char buf[256];
  return buf;
}

void __cxa_throw(void*, void*, void (*)(void*)) {
  fprintf(stderr, "C++ exception thrown in WASM — aborting\n");
  abort();
}

void* __cxa_begin_catch(void* p) { return p; }
void __cxa_end_catch() {}
void __cxa_rethrow() { abort(); }
int __cxa_guard_acquire(long long* guard) { return !*(char*)guard; }
void __cxa_guard_release(long long* guard) { *(char*)guard = 1; }
void __cxa_guard_abort(long long*) {}

}
