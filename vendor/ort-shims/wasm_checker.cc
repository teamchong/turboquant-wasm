/**
 * WASM implementation of onnx::checker::check_node.
 *
 * The full checker.cc requires std::filesystem::hard_link_count,
 * is_symlink, and is_regular_file — none available in WASM.
 *
 * For browser inference, model validation is unnecessary: models are
 * loaded from trusted ArrayBuffers, not untrusted filesystem paths.
 * ORT's own graph-level validation still runs independently.
 */

#include "onnx/checker.h"

namespace onnx {
namespace checker {

void check_node(const NodeProto& /*node*/, const CheckerContext& /*ctx*/,
                const LexicalScopeContext& /*lex_ctx*/) {
  // WASM: no filesystem, no external data — validation not applicable
}

}  // namespace checker
}  // namespace onnx
