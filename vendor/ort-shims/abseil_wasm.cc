/**
 * Abseil implementations for single-threaded WASM.
 *
 * Ruy (matrix multiplication library) uses abseil for:
 * - Mutex (thread safety — single thread, so lock/unlock are no-ops)
 * - Notification (thread signaling — always immediately notified)
 * - Time (Now, Duration — for performance measurement)
 * - RawLog (error logging)
 * - GetStackTrace (debugging)
 * - AbslInternalSleepFor (thread sleep — no-op)
 * - hash_internal::MixingHashState::kSeed (hash seed constant)
 *
 * These are correct implementations for wasm32-wasi with a single thread.
 */

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace absl {
inline namespace lts_20250814 {

// ============================================================================
// Duration — simple wrapper around nanoseconds
// ============================================================================
// Match the real abseil Duration layout: rep_hi_ (int64_t) + rep_lo_ (uint32_t)
class Duration {
 public:
  constexpr Duration() : rep_hi_(0), rep_lo_(0) {}
  Duration& operator-=(Duration d);
 private:
  int64_t rep_hi_;
  uint32_t rep_lo_;
};

Duration& Duration::operator-=(Duration d) {
  rep_hi_ -= d.rep_hi_;
  if (rep_lo_ < d.rep_lo_) { rep_hi_--; }
  rep_lo_ -= d.rep_lo_;
  return *this;
}

double ToDoubleMilliseconds(Duration) {
  return 0.0;  // timing not used in WASM
}

// ============================================================================
// Time / Now — returns zero-based monotonic time
// In WASM there's no high-res clock; this is used only for ruy's
// optional performance tuning which we don't need.
// ============================================================================
Duration Now() { return Duration(); }

// ============================================================================
// Mutex — single-threaded WASM, all operations are no-ops
// ============================================================================
class Mutex {
 public:
  void lock();
  void unlock();
  void ForgetDeadlockInfo();
};

void Mutex::lock() {}
void Mutex::unlock() {}
void Mutex::ForgetDeadlockInfo() {}

// ============================================================================
// Notification — single-threaded, Notify() sets flag, always ready
// ============================================================================
class Notification {
 public:
  ~Notification();
  void Notify();
  bool HasBeenNotified() const { return notified_; }
 private:
  bool notified_ = false;
};

Notification::~Notification() {}
void Notification::Notify() { notified_ = true; }

// ============================================================================
// raw_log — print to stderr (routes to fd_write in WASM)
// ============================================================================
// LogSeverity is an enum class : int in abseil
enum class LogSeverity : int {
  kInfo = 0,
  kWarning = 1,
  kError = 2,
  kFatal = 3,
};

namespace raw_log_internal {
void RawLog(LogSeverity severity, const char* file, int line, const char* format, ...) {
  (void)severity; (void)file; (void)line; (void)format;
  // In production WASM, we silence these. Could vfprintf(stderr,...) if debugging.
}
}  // namespace raw_log_internal

// ============================================================================
// GetStackTrace — no stack unwinding in WASM
// ============================================================================
int GetStackTrace(void** result, int max_depth, int skip_count) {
  (void)result; (void)max_depth; (void)skip_count;
  return 0;
}

// ============================================================================
// hash_internal — kSeed is a compile-time constant used by abseil hash tables
// ============================================================================
namespace hash_internal {
class MixingHashState {
 public:
  static const uint64_t kSeed;
};
const uint64_t MixingHashState::kSeed = 0xec53c1a1c8e845e1ULL;
}  // namespace hash_internal

}  // inline namespace lts_20250814
}  // namespace absl

// ============================================================================
// AbslInternalSleepFor — no-op on single-threaded WASM
// ============================================================================
extern "C" void AbslInternalSleepFor_lts_20250814() {}

// ============================================================================
// pthread_once — execute initialization exactly once (trivial in single thread)
// ============================================================================
extern "C" int pthread_once(int* once_control, void (*init_routine)(void)) {
  if (once_control && *once_control == 0) {
    *once_control = 1;
    if (init_routine) init_routine();
  }
  return 0;
}
