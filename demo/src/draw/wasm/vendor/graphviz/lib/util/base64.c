#include "config.h"

#include <stddef.h>
#include <util/alloc.h>
#include <util/base64.h>

static size_t div_up(size_t dividend, size_t divisor) {
  return dividend / divisor + (dividend % divisor != 0);
}

static const char base64_alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

size_t gv_base64_size(size_t source_size) { return div_up(source_size, 3) * 4; }

char *gv_base64(const unsigned char *source, size_t size) {
  size_t buf_i = 0;
  char *buf = gv_alloc(gv_base64_size(size) + 1);

  for (size_t data_i = 0; data_i < size; data_i += 3) {
    unsigned char d0 = source[data_i];
    int v = (d0 & 0xFC) >> 2; // 1111_1100
    buf[buf_i++] = base64_alphabet[v];

    unsigned char d1 = data_i + 1 < size ? source[data_i + 1] : 0;
    v = (d0 & 0x03) << 4    // 0000_0011
        | (d1 & 0xF0) >> 4; // 1111_0000
    buf[buf_i++] = base64_alphabet[v];
    if (size <= data_i + 1) {
      goto end;
    }

    unsigned char d2 = data_i + 2 < size ? source[data_i + 2] : 0;
    v = (d1 & 0x0F) << 2    // 0000_1111
        | (d2 & 0xC0) >> 6; // 1100_0000
    buf[buf_i++] = base64_alphabet[v];
    if (size <= data_i + 2) {
      goto end;
    }

    v = d2 & 0x3F; // 0011_1111
    buf[buf_i++] = base64_alphabet[v];
  }

end:
  while (buf_i % 4 != 0) {
    buf[buf_i++] = base64_alphabet[64];
  }

  // Write a string terminator. This is not strictly necessary, but is a
  // backstop against callers who do not use `gv_base64_size` and assume they
  // can e.g. `fputs` the returned data.
  buf[buf_i++] = '\0';

  return buf;
}
