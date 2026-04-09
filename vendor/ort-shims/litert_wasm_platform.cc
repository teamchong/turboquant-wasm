/**
 * WASM platform layer for LiteRT runtime.
 *
 * Provides implementations for symbols that execute at runtime on WASM:
 * - SentencePiece UTF-8 utilities (tokenization)
 * - Protobuf byte order (model deserialization)
 * - Abseil platform functions (sync/threading infrastructure)
 *
 * Symbols for hardware-unavailable features (GPU, compiler, vendor NPUs)
 * are handled by wasm-ld --allow-undefined — they trap if ever reached,
 * which is correct since those code paths are unreachable on WASM.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// ============================================================================
// SentencePiece: UTF-8 and utility functions (used during tokenization)
// ============================================================================
namespace sentencepiece {

std::string GetDataDir() { return "/"; }

namespace log_domain {
  double LogSum(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double max_val = v[0];
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > max_val) max_val = v[i];
    double sum = 0.0;
    for (double x : v) sum += __builtin_exp(x - max_val);
    return max_val + __builtin_log(sum);
  }
}

namespace random {
  void* GetRandomGenerator() {
    static int rng;
    return &rng;
  }
}

namespace filesystem {
  class ReadableFile { public: virtual ~ReadableFile() = default; };
  class WritableFile { public: virtual ~WritableFile() = default; };
  std::unique_ptr<ReadableFile> NewReadableFile(std::string_view, bool) { return nullptr; }
  std::unique_ptr<WritableFile> NewWritableFile(std::string_view, bool) { return nullptr; }
}

namespace string_util {
  bool IsStructurallyValid(std::string_view s) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    const uint8_t* end = p + s.size();
    while (p < end) {
      if (*p < 0x80) { ++p; }
      else if ((*p & 0xE0) == 0xC0 && p + 1 < end) { p += 2; }
      else if ((*p & 0xF0) == 0xE0 && p + 2 < end) { p += 3; }
      else if ((*p & 0xF8) == 0xF0 && p + 3 < end) { p += 4; }
      else return false;
    }
    return p == end;
  }

  std::vector<uint32_t> UTF8ToUnicodeText(std::string_view s) {
    std::vector<uint32_t> result;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    const uint8_t* end = p + s.size();
    while (p < end) {
      uint32_t cp;
      if (*p < 0x80) { cp = *p++; }
      else if ((*p & 0xE0) == 0xC0) { cp = (*p++ & 0x1F); cp = (cp << 6) | (*p++ & 0x3F); }
      else if ((*p & 0xF0) == 0xE0) { cp = (*p++ & 0x0F); cp = (cp << 6) | (*p++ & 0x3F); cp = (cp << 6) | (*p++ & 0x3F); }
      else { cp = (*p++ & 0x07); cp = (cp << 6) | (*p++ & 0x3F); cp = (cp << 6) | (*p++ & 0x3F); cp = (cp << 6) | (*p++ & 0x3F); }
      result.push_back(cp);
    }
    return result;
  }

  std::string UnicodeTextToUTF8(const std::vector<uint32_t>& text) {
    std::string result;
    for (uint32_t cp : text) {
      if (cp < 0x80) { result += static_cast<char>(cp); }
      else if (cp < 0x800) { result += static_cast<char>(0xC0 | (cp >> 6)); result += static_cast<char>(0x80 | (cp & 0x3F)); }
      else if (cp < 0x10000) { result += static_cast<char>(0xE0 | (cp >> 12)); result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); result += static_cast<char>(0x80 | (cp & 0x3F)); }
      else { result += static_cast<char>(0xF0 | (cp >> 18)); result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); result += static_cast<char>(0x80 | (cp & 0x3F)); }
    }
    return result;
  }

  size_t DecodeUTF8(const char* begin, const char* end, size_t* mblen) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(begin);
    if (p >= reinterpret_cast<const uint8_t*>(end)) { *mblen = 0; return 0; }
    if (*p < 0x80) { *mblen = 1; return *p; }
    else if ((*p & 0xE0) == 0xC0) { *mblen = 2; return ((*p & 0x1F) << 6) | (p[1] & 0x3F); }
    else if ((*p & 0xF0) == 0xE0) { *mblen = 3; return ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); }
    else { *mblen = 4; return ((*p & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); }
  }
}
}

// ============================================================================
// Protobuf: network byte order conversion (model deserialization)
// ============================================================================
namespace google::protobuf {
  uint32_t ghtonl(uint32_t x) {
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) |
           ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24);
  }
}

// ============================================================================
// Abseil platform layer (sync/threading/debugging infrastructure)
// ============================================================================
namespace absl::lts_20250814 {
  namespace debugging_internal {
    void DumpStackTrace(int, int, bool, void (*)(const char*, void*), void*) {}
  }
  namespace synchronization_internal {
    void* CreateThreadIdentity() { return nullptr; }
  }
}

// ============================================================================
// tflite: model loading from memory (no filesystem in WASM)
// ============================================================================
namespace tflite {
  class ErrorReporter;
  class Allocation;
  Allocation* GetAllocationFromFile(const char*, ErrorReporter*, bool) { return nullptr; }
}
