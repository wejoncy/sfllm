
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
     /*
   * From csrc/elementwise
   */
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, Tensor? input_2=None) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  // m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  // m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def(
      "apply_rope_pos_ids_cos_sin_cache(Tensor q, Tensor k, Tensor! q_rope, Tensor! k_rope, Tensor cos_sin_cache, "
      "Tensor pos_ids, bool interleave, bool enable_pdl, "
      "Tensor? v, Tensor!? k_buffer, Tensor!? v_buffer, Tensor? kv_cache_loc) -> ()");
  m.impl("apply_rope_pos_ids_cos_sin_cache", torch::kCUDA, &apply_rope_pos_ids_cos_sin_cache);

}

#if defined(_WIN32) || defined(_WIN64)
// Empty Python module initialization for TORCH_LIBRARY_FRAGMENT
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Empty module - the actual operators are registered via TORCH_LIBRARY_FRAGMENT
}
#endif