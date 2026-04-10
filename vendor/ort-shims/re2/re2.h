/**
 * Minimal RE2 interface for LiteRT-LM prompt template editing.
 * The template editor uses RE2::GlobalReplace to rewrite Python string
 * methods (startswith, endswith, etc.) to Jinja filter syntax.
 *
 * Since we replaced the Jinja engine with a C++ implementation that
 * handles these patterns natively, the regex rewrites are no-ops.
 * This header provides the API surface so prompt_template.cc compiles.
 */
#pragma once

#include <string>

namespace re2 {

class RE2 {
 public:
  explicit RE2(const char* pattern) : pattern_(pattern) {}
  explicit RE2(const std::string& pattern) : pattern_(pattern) {}

  bool ok() const { return true; }

  static bool GlobalReplace(std::string* str, const RE2& pattern,
                            const char* rewrite) {
    (void)str; (void)pattern; (void)rewrite;
    return false;
  }

  static bool GlobalReplace(std::string* str, const RE2& pattern,
                            const std::string& rewrite) {
    (void)str; (void)pattern; (void)rewrite;
    return false;
  }

  static bool GlobalReplace(std::string* str, const char* pattern,
                            const char* rewrite) {
    (void)str; (void)pattern; (void)rewrite;
    return false;
  }

  static bool GlobalReplace(std::string* str, const char* pattern,
                            const std::string& rewrite) {
    (void)str; (void)pattern; (void)rewrite;
    return false;
  }

  // Escapes special regex characters in the input string.
  static std::string QuoteMeta(const std::string& str) {
    std::string result;
    for (char c : str) {
      if (c == '\\' || c == '.' || c == '*' || c == '+' || c == '?' ||
          c == '(' || c == ')' || c == '[' || c == ']' || c == '{' ||
          c == '}' || c == '|' || c == '^' || c == '$') {
        result += '\\';
      }
      result += c;
    }
    return result;
  }

 private:
  std::string pattern_;
};

}  // namespace re2

// Bring into global scope for includes like "re2/re2.h"
using re2::RE2;
