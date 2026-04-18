/**
 * WASM platform layer — provides the main() entry point required by WASI libc's
 * startup code. This module is a library (entry = disabled), but WASI's _start
 * references main(). We provide an empty one to satisfy the linker.
 */

int main(void) {
    return 0;
}
