/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:05
 * @LastEditTime: 2024-12-02 19:43:10
 * @FilePath: /easy_deploy/inference_core/rknn_core/include/rknn_core/rknn_core.h
 */
#ifndef __EASY_DEPLOY_INFERENCE_CORE_ORT_CORE_H
#define __EASY_DEPLOY_INFERENCE_CORE_ORT_CORE_H

#include "deploy_core/base_infer_core.h"

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

#endif
