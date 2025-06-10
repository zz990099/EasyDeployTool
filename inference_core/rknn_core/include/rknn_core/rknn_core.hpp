#pragma once

#include "deploy_core/base_infer_core.hpp"

namespace easy_deploy {

enum RknnInputTensorType {
  RK_UINT8,
  RK_INT8,
  RK_FLOAT16,
  RK_FLOAT32,
  RK_UINT32,
  RK_INT32,
  RK_INT64
};

std::shared_ptr<BaseInferCore> CreateRknnInferCore(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type    = {},
    const int                                                   mem_buf_size     = 5,
    const int                                                   parallel_ctx_num = 1);

std::shared_ptr<BaseInferCoreFactory> CreateRknnInferCoreFactory(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type    = {},
    const int                                                   mem_buf_size     = 5,
    const int                                                   parallel_ctx_num = 1);

} // namespace easy_deploy
