#pragma once
#include <thread>
#include <ostream>
// Provide the implementation that libc++ declares but doesn't link
namespace std {
  template<class CharT, class Traits>
  basic_ostream<CharT, Traits>& operator<<(basic_ostream<CharT, Traits>& os, thread::id) {
    return os << CharT('0');
  }
}
