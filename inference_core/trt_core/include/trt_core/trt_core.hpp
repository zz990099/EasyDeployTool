#pragma once

#include <unordered_map>
#include "deploy_core/base_infer_core.hpp"

namespace easy_deploy {

/**
 * @brief Construct `TrtInferCore` by providing tensorrt engine file path, max(default) blobs
 * shape and blob buffer pool size (defualt=5). If your model is parsed from a onnx model with
 * dynamic blob shape (e.g. blob_dim=-1), a mapping of blob_name and blob_shape should be provided
 * to help `TrtInferCore` alloc a apposite size blob buffer.
 *
 * @param engine_path Tensorrt engine file path.
 * @param blobs_shape Mapping of blob_name and blob_shape.
 * @param mem_buf_size Size of buffer pool.
 */
std::shared_ptr<BaseInferCore> CreateTrtInferCore(
    std::string                                                   model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape  = {},
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape = {},
    const int                                                     mem_buf_size       = 5);

std::shared_ptr<BaseInferCoreFactory> CreateTrtInferCoreFactory(
    std::string                                                   model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape  = {},
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape = {},
    const int                                                     mem_buf_size       = 5);

} // namespace easy_deploy
