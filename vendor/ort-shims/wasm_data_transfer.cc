/**
 * WASM DataTransfer for JS execution provider.
 * In our single-WASM binary, all data is in linear memory — copy is memcpy.
 * The jsep_download/jsep_copy JS imports handle any GPU↔CPU transfer.
 */

#include "core/providers/js/data_transfer.h"
#include <cstring>

extern "C" {
  void jsep_download(const void* src, void* dst, size_t bytes);
  void jsep_copy(const void* src, void* dst, size_t bytes, int gpu_to_cpu);
}

namespace onnxruntime {
namespace js {

bool DataTransfer::CanCopy(const OrtDevice& src, const OrtDevice& dst) const {
  return src.Type() == OrtDevice::CPU || dst.Type() == OrtDevice::CPU;
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  const auto bytes = src.SizeInBytes();
  if (bytes == 0) return common::Status::OK();
  memcpy(dst.MutableDataRaw(), src.DataRaw(), bytes);
  return common::Status::OK();
}

}  // namespace js
}  // namespace onnxruntime
