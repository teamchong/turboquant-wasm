/**
 * Empty kernel registrations for ops not needed in browser LLM inference.
 * Tokenizer and RegexFullMatch require RE2 regex library which we don't compile.
 * These ops are registered in ORT's kernel tables but never instantiated
 * when running Gemma/LLM models.
 */

#include "core/framework/op_kernel.h"

namespace onnxruntime {

// RegexFullMatch — registered but not used by LLM models
struct kCpuExecutionProvider_RegexFullMatch_kOnnxDomain_ver20 {};
template <>
KernelCreateInfo BuildKernelCreateInfo<kCpuExecutionProvider_RegexFullMatch_kOnnxDomain_ver20>() {
  return {};
}

namespace contrib {

// Tokenizer — requires RE2, not used by Gemma
struct kCpuExecutionProvider_Tokenizer_kMSDomain_ver1_string {};
template <>
KernelCreateInfo BuildKernelCreateInfo<kCpuExecutionProvider_Tokenizer_kMSDomain_ver1_string>() {
  return {};
}

}  // namespace contrib
}  // namespace onnxruntime
