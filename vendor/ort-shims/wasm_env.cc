/**
 * WASM implementation of onnxruntime::Env.
 * Single-threaded, no filesystem, no dynamic libraries.
 * Provides the minimum needed for OrtInit + OrtCreateSession + OrtRun.
 */

#include "core/platform/env.h"
#include "core/platform/device_discovery.h"
#include "core/common/logging/logging.h"
#include <cstring>

namespace onnxruntime {

namespace {

class WasmEnv : public Env {
 public:
  EnvThread* CreateThread(const ORTCHAR_T*, int,
                          unsigned (*)(int, Eigen::ThreadPoolInterface*),
                          Eigen::ThreadPoolInterface*,
                          const ThreadOptions&) override { return nullptr; }

  int GetNumPhysicalCpuCores() const override { return 1; }
  std::vector<LogicalProcessors> GetDefaultThreadAffinities() const override { return {}; }
  int GetL2CacheSize() const override { return 256 * 1024; }
  void SleepForMicroseconds(int64_t) const override {}

  common::Status GetFileLength(const ORTCHAR_T*, size_t&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status GetFileLength(int, size_t&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status ReadFileIntoBuffer(const ORTCHAR_T*, FileOffsetType, size_t,
                                     gsl::span<char>) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status MapFileIntoMemory(const ORTCHAR_T*, FileOffsetType, size_t,
                                    MappedMemoryPtr&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }

  bool FolderExists(const std::string&) const override { return false; }
  bool FileExists(const std::string&) const override { return false; }
  common::Status CreateFolder(const std::string&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status DeleteFolder(const PathString&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status FileOpenRd(const std::string&, int&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status FileOpenWr(const std::string&, int&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }
  common::Status FileClose(int) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No filesystem");
  }

  common::Status GetCanonicalPath(const PathString& path, PathString& out) const override {
    out = path;
    return common::Status::OK();
  }

  PIDType GetSelfPid() const override { return 1; }

  common::Status LoadDynamicLibrary(const PathString&, bool, void**) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No dlopen");
  }
  common::Status UnloadDynamicLibrary(void*) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No dlopen");
  }
  PathString GetRuntimePath() const override { return {}; }
  common::Status GetSymbolFromLibrary(void*, const std::string&, void**) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "No dlsym");
  }
  std::string FormatLibraryFileName(const std::string& name, const std::string&) const override {
    return name;
  }

  const Telemetry& GetTelemetryProvider() const override { return telemetry_; }
  std::string GetEnvironmentVar(const std::string&) const override { return {}; }

 private:
  Telemetry telemetry_;
};

}  // namespace

Env& Env::Default() {
  static WasmEnv instance;
  return instance;
}


}  // namespace onnxruntime
