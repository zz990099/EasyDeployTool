#pragma once

#include "deploy_core/base_infer_core.hpp"
#include "deploy_core/common.hpp"

#include <opencv2/opencv.hpp>

namespace easy_deploy {

struct StereoPipelinePackage : public IPipelinePackage {
  // the wrapped pipeline image data
  std::shared_ptr<IPipelineImageData> left_image_data;
  std::shared_ptr<IPipelineImageData> right_image_data;
  // confidence used in postprocess
  float conf_thresh;
  // record the transform factor during image preprocess
  float transform_scale;

  //
  cv::Mat disp;

  // maintain the blobs buffer instance
  std::shared_ptr<BlobsTensor> infer_buffer;

  // override from `IPipelinePakcage`, to provide the blobs buffer to inference_core
  BlobsTensor *GetInferBuffer() override
  {
    return infer_buffer.get();
  }
};

/**
 * @brief A functor to generate sam results from `SamPipelinePackage`. Used in async pipeline.
 *
 */
class StereoGenResultType {
public:
  cv::Mat operator()(const std::shared_ptr<IPipelinePackage> &package)
  {
    auto stereo_package = std::dynamic_pointer_cast<StereoPipelinePackage>(package);
    if (stereo_package == nullptr)
    {
      LOG_ERROR("[StereoGenResultType] Got INVALID package ptr!!!");
      return {};
    }
    return std::move(stereo_package->disp);
  }
};

class BaseStereoMatchingModel : public BaseAsyncPipeline<cv::Mat, StereoGenResultType> {
protected:
  using ParsingType = std::shared_ptr<IPipelinePackage>;

  BaseStereoMatchingModel(const std::shared_ptr<BaseInferCore> &inference_core);

public:
  bool ComputeDisp(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disp_output);

  [[nodiscard]] std::future<cv::Mat> ComputeDispAsync(const cv::Mat &left_image,
                                                      const cv::Mat &right_image);

protected:
  virtual bool PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) = 0;

  virtual bool PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) = 0;

private:
  using BaseAsyncPipeline::PushPipeline;

protected:
  std::shared_ptr<BaseInferCore> inference_core_;

  static const std::string stereo_pipeline_name_;
};

struct MonoStereoPipelinePackage : public IPipelinePackage {
  // the wrapped pipeline image data
  std::shared_ptr<IPipelineImageData> input_image_data;
  // record the transform factor during image preprocess
  float transform_scale;

  //
  cv::Mat depth;

  // maintain the blobs buffer instance
  std::shared_ptr<BlobsTensor> infer_buffer;

  // override from `IPipelinePakcage`, to provide the blobs buffer to inference_core
  BlobsTensor *GetInferBuffer() override
  {
    return infer_buffer.get();
  }
};

/**
 * @brief A functor to generate sam results from `SamPipelinePackage`. Used in async pipeline.
 *
 */
class MonoStereoGenResultType {
public:
  cv::Mat operator()(const std::shared_ptr<IPipelinePackage> &package)
  {
    auto stereo_package = std::dynamic_pointer_cast<MonoStereoPipelinePackage>(package);
    if (stereo_package == nullptr)
    {
      LOG_ERROR("[MonoStereoGenResultType] Got INVALID package ptr!!!");
      return {};
    }
    return std::move(stereo_package->depth);
  }
};

class BaseMonoStereoModel : public BaseAsyncPipeline<cv::Mat, MonoStereoGenResultType> {
protected:
  using ParsingType = std::shared_ptr<IPipelinePackage>;

  BaseMonoStereoModel(const std::shared_ptr<BaseInferCore> &inference_core);

public:
  bool ComputeDepth(const cv::Mat &input_image, cv::Mat &depth_output);

  [[nodiscard]] std::future<cv::Mat> ComputeDepthAsync(const cv::Mat &input_image);

protected:
  virtual bool PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) = 0;

  virtual bool PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) = 0;

private:
  using BaseAsyncPipeline::PushPipeline;

protected:
  std::shared_ptr<BaseInferCore> inference_core_;

  static const std::string mono_stereo_pipeline_name_;
};

} // namespace easy_deploy
