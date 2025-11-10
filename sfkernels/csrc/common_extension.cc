
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#if defined(_WIN32) || defined(_WIN64)
#include <torch/extension.h>
#endif

#include "ops.h"

TORCH_LIBRARY_FRAGMENT(sfkernels, m) {

      /*
   * From csrc/speculative
   */
  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict, int cuda_stream) -> ()");
  m.impl("verify_tree_greedy", torch::kCUDA, &verify_tree_greedy);

  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> "
      "()");
  m.impl("build_tree_kernel_efficient", torch::kCUDA, &build_tree_kernel_efficient);

}

#if defined(_WIN32) || defined(_WIN64)
// Empty Python module initialization for TORCH_LIBRARY_FRAGMENT
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Empty module - the actual operators are registered via TORCH_LIBRARY_FRAGMENT
}
#endif