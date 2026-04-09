#pragma once

// POSIX functions not available in WASI but referenced by TFLite/XNNPack
#ifdef __cplusplus
extern "C" {
#endif
int dup(int fd);
int getpagesize(void);
unsigned int getuid(void);
unsigned int getgid(void);
#ifdef __cplusplus
}
#endif

#include <filesystem>
#include <system_error>

// Provide filesystem functions not implemented on wasm32-wasi
namespace std { namespace filesystem {
  inline bool exists(const path&) { return false; }
  inline bool exists(const path&, error_code& ec) { ec.clear(); return false; }
  inline uintmax_t file_size(const path&) { return 0; }
  inline uintmax_t file_size(const path&, error_code& ec) { ec.clear(); return 0; }
  inline bool is_regular_file(const path&) { return false; }
  inline bool is_directory(const path&) { return false; }
  inline bool is_symlink(const path&) { return false; }
  inline bool is_symlink(const path&, error_code&) { return false; }
  inline path weakly_canonical(const path& p) { return p; }
  inline path weakly_canonical(const path& p, error_code&) { return p; }
  inline path current_path() { return path("/"); }
  inline path current_path(error_code&) { return path("/"); }
  inline path absolute(const path& p) { return p; }
  inline path absolute(const path& p, error_code&) { return p; }
  inline path relative(const path& p, const path&) { return p; }
  inline path relative(const path& p, const path&, error_code&) { return p; }
  inline bool create_directories(const path&) { return false; }
  inline bool create_directories(const path&, error_code&) { return false; }
  inline bool remove(const path&) { return false; }
  inline bool remove(const path&, error_code&) { return false; }
  inline void rename(const path&, const path&) {}
  inline void rename(const path&, const path&, error_code&) {}
  inline path read_symlink(const path& p) { return p; }
  inline path read_symlink(const path& p, error_code&) { return p; }
}}
