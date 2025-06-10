#pragma once
#include "deploy_core/base_infer_core.hpp"

namespace easy_deploy {

std::shared_ptr<BaseInferCore> CreateOrtInferCore(
    const std::string                                             onnx_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape  = {},
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape = {},
    const int                                                     num_threads        = 0);

std::shared_ptr<BaseInferCoreFactory> CreateOrtInferCoreFactory(
    const std::string                                             onnx_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape  = {},
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape = {},
    const int                                                     num_threads        = 0);

} // namespace easy_deploy
