#include "ort_core/ort_core.hpp"

namespace easy_deploy {

struct OrtInferCoreParams {
  std::string                                            onnx_path;
  std::unordered_map<std::string, std::vector<uint64_t>> input_blobs_shape;
  std::unordered_map<std::string, std::vector<uint64_t>> output_blobs_shape;
  int                                                    num_threads;
};

class OrtInferCoreFactory : public BaseInferCoreFactory {
public:
  OrtInferCoreFactory(const OrtInferCoreParams &params) : params_(params)
  {}

  std::shared_ptr<BaseInferCore> Create() override
  {
    return CreateOrtInferCore(params_.onnx_path, params_.input_blobs_shape,
                              params_.output_blobs_shape, params_.num_threads);
  }

private:
  const OrtInferCoreParams params_;
};

std::shared_ptr<BaseInferCoreFactory> CreateOrtInferCoreFactory(
    const std::string                                             onnx_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                     num_threads)
{
  OrtInferCoreParams params;
  params.onnx_path          = onnx_path;
  params.input_blobs_shape  = input_blobs_shape;
  params.output_blobs_shape = output_blobs_shape;
  params.num_threads        = num_threads;

  return std::make_shared<OrtInferCoreFactory>(params);
}

} // namespace easy_deploy
