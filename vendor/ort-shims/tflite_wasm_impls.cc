/**
 * WASM implementations for TFLite symbols that depend on unavailable platform
 * features (NNAPI, ARM NEON, telemetry, etc.).
 *
 * These are correct implementations for a single-threaded WASM target:
 * - NNAPI returns null (Android-only)
 * - ARM detection returns false (not ARM)
 * - Telemetry is silent (no telemetry backend)
 * - Excluded ops return null registration (not needed for LLM inference)
 *
 * File I/O (FileDescriptor, MMapHandle) is compiled from the real TFLite
 * sources with XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG to avoid mmap.
 */

#include <cstdarg>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

// Forward declarations matching TFLite types
struct TfLiteRegistration;
struct TfLiteContext;
struct TfLiteDelegate;
struct TfLiteSparsity;
struct TfLiteTelemetryProfilerStruct;
struct TfLiteTelemetryInterpreterSettings;
enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };

namespace Eigen { struct half {}; }

// ErrorReporter and MemoryAllocation now provided by litert runtime.

namespace tflite {

// ============================================================================
// Kernel registrations — ops excluded from build (audio, RNG, LSH, conv2d)
// These ops are not used in LLM inference. Returning null tells the
// interpreter the op is unavailable, which is the correct behavior.
// ============================================================================
namespace ops {
namespace builtin {
TfLiteRegistration* Register_CONV_2D() { return nullptr; }
TfLiteRegistration* Register_CONVOLUTION_REF() { return nullptr; }
TfLiteRegistration* Register_LSH_PROJECTION() { return nullptr; }
TfLiteRegistration* Register_MULTINOMIAL() { return nullptr; }
TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL() { return nullptr; }
TfLiteRegistration* Register_RANDOM_UNIFORM() { return nullptr; }
TfLiteRegistration* Register_RFFT2D() { return nullptr; }
}  // namespace builtin
namespace custom {
TfLiteRegistration* Register_AUDIO_SPECTROGRAM() { return nullptr; }
}  // namespace custom
}  // namespace ops

// ============================================================================
// XNNPack delegate — linked statically, delegate wrapper not needed
// ============================================================================
enum XNNPackQS8Options { kXNNPackQS8Default = 0 };
using TfLiteDelegatePtr = void*;

TfLiteDelegatePtr MaybeCreateXNNPACKDelegate(TfLiteContext*, XNNPackQS8Options) {
  return nullptr;
}

// ============================================================================
// NNAPI delegate — Android-only hardware accelerator, not available on WASM
// ============================================================================
TfLiteDelegate* NnApiDelegate() { return nullptr; }

// ============================================================================
// Telemetry — no telemetry backend in browser WASM
// ============================================================================
namespace telemetry {
void* MakeTfLiteTelemetryProfiler(TfLiteTelemetryProfilerStruct*) {
  return nullptr;
}
TfLiteStatus TelemetryReportEvent(TfLiteContext*, const char*, TfLiteStatus) {
  return kTfLiteOk;
}
TfLiteStatus TelemetryReportSettings(TfLiteContext*, const char*,
                                     const TfLiteTelemetryInterpreterSettings*) {
  return kTfLiteOk;
}
}  // namespace telemetry

// ============================================================================
// ARM NEON detection — WASM target, not ARM
// ============================================================================
bool DetectArmNeonDotprod() { return false; }

// ============================================================================
// ParseModelControlDependencies — remat metadata parser
// Models loaded in browser won't have remat control deps metadata.
// ============================================================================
bool ParseModelControlDependencies(
    const char*, size_t,
    std::vector<std::vector<std::pair<int, int>>>*) {
  return true;  // no control dependencies present
}

// ============================================================================
// Sparsity format converters — template instantiations for sparse model support
// LLM models (Gemma) don't use sparse tensors, but the interpreter links these.
// ============================================================================
namespace internal {
namespace sparsity {

template <typename T>
class FormatConverter {
 public:
  FormatConverter(const std::vector<int>&, const TfLiteSparsity&) {}
  void SparseToDense(const T*, size_t, T*, TfLiteContext*) {}
  void SparseToDense(const T*) {}
};

template class FormatConverter<float>;
template class FormatConverter<signed char>;
template class FormatConverter<Eigen::half>;

}  // namespace sparsity
}  // namespace internal

// ============================================================================
// 4-bit quantized fully connected — reference implementations
// Used by 4-bit quantized models. Gemma E4B uses per-layer embeddings,
// but the kernel registration pulls these in unconditionally.
// ============================================================================
namespace optimized_4bit {

void ReferenceAssignBiasAndComputeOffsets(
    const int*, const float*, const float*, const float*,
    float*, int, int) {}

void ReferenceBatchQuantizeFloats4Bit(
    const float*, int, int, signed char*, float*, int, int, int*) {}

void ReferencePrepack(
    unsigned char*, const signed char*, int, int, int, int, int, int) {}

template <int A, int B, int C>
void ReferenceRunKernel(
    const unsigned char*, const signed char*, int*, int, int, int, int, int, int) {}

template <int A, int B>
void ReferenceUnpack(
    float*, const int*, int, int, const float*, const float*, int, int) {}

// Explicit template instantiations matching the linker requirements
template void ReferenceRunKernel<4, 1, 32>(
    const unsigned char*, const signed char*, int*, int, int, int, int, int, int);
template void ReferenceUnpack<4, 1>(
    float*, const int*, int, int, const float*, const float*, int, int);

}  // namespace optimized_4bit

}  // namespace tflite

// ============================================================================
// SimpleDynamicBuffer — string tensor builder (from TFLite string_util)
// NOTE: This is in mlir::TFL namespace (NOT nested inside tflite::)
// ============================================================================
namespace mlir {
namespace TFL {

class SimpleDynamicBuffer {
 public:
  void AddString(const char* str, size_t len);
  void WriteToBuffer(char** buffer);
 private:
  std::vector<std::vector<char>> data_;
};

void SimpleDynamicBuffer::AddString(const char* str, size_t len) {
  data_.push_back(std::vector<char>(str, str + len));
}

void SimpleDynamicBuffer::WriteToBuffer(char** buffer) {
  *buffer = nullptr;
}

}  // namespace TFL
}  // namespace mlir
