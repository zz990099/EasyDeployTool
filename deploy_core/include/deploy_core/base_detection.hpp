#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deploy_core/async_pipeline.hpp"
#include "deploy_core/base_infer_core.hpp"
#include "common_utils/pipeline_image.hpp"

namespace easy_deploy {

/**
 * @brief The common detection_2d pipeline package wrapper.
 *
 */
struct DetectionPipelinePackage : public IPipelinePackage {
  // the wrapped pipeline image data
  std::shared_ptr<IPipelineImageData> input_image_data;
  // confidence used in postprocess
  float conf_thresh;
  // record the transform factor during image preprocess
  float transform_scale;
  // the detection result
  std::vector<BBox2D> results;

  // maintain the blobs buffer instance
  std::shared_ptr<BlobsTensor> infer_buffer;

  // override from `IPipelinePakcage`, to provide the blobs buffer to inference_core
  BlobsTensor *GetInferBuffer() override
  {
    return infer_buffer.get();
  }
};

/**
 * @brief A abstract class defines two pure virtual methods -- `PreProcess` and `PostProcess`.
 * The derived class could only override these methods to make it work.
 *
 */
class IDetectionModel {
public:
  IDetectionModel() = default;

protected:
  virtual ~IDetectionModel() = default;

  /**
   * @brief PreProcess-Stage. Inside the method, you should cast the `pipeline_unit` pointer to
   * `DetectionPipelinePackage` type pointer, and check if the convertion works. If the package
   * pointer is not valid or anything goes wrong, it should return `false` to mention the inference
   * processing to drop the package.
   *
   * @param pipeline_unit
   * @return true
   * @return false
   */
  virtual bool PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) = 0;

  /**
   * @brief PostProcess-Stage. Inside the method, you should cast the `pipeline_unit` pointer to
   * `DetectionPipelinePackage` type pointer, and check if the convertion works. If the package
   * pointer is not valid or anything goes wrong, it should return `false` to mention the inference
   * processing to drop the package.
   *
   * @param pipeline_unit
   * @return true
   * @return false
   */
  virtual bool PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) = 0;
};

/**
 * @brief A functor to generate detection results from `DetectionPipelinePackage`. Used in async
 * pipeline.
 *
 */
class DetectionGenResultType {
public:
  std::vector<BBox2D> operator()(const std::shared_ptr<IPipelinePackage> &package)
  {
    auto detection_package = std::dynamic_pointer_cast<DetectionPipelinePackage>(package);
    if (detection_package == nullptr)
    {
      LOG_ERROR("[DetectionGenResult] Got INVALID package ptr!!!");
      return {};
    }
    return std::move(detection_package->results);
  }
};

/**
 * @brief The base class of detection_2d algorithms. It implements `Detect` which is the synchronous
 * version of detection and `DetectAsync` which is the asynchronous version of detection.
 *
 * @note Call `InitPipeline()` before you intend to use `DetectAsync`. And Check if `DetectAsync`
 * returns a valid `std::future<>` instance before involke `get()` method.
 *
 */
class BaseDetectionModel : public IDetectionModel,
                           public BaseAsyncPipeline<std::vector<BBox2D>, DetectionGenResultType> {
  typedef std::shared_ptr<IPipelinePackage> ParsingType;

public:
  BaseDetectionModel(std::shared_ptr<BaseInferCore> infer_core);

  /**
   * @brief Run the detection processing in synchronous mode.
   *
   * @param input_image input image in cv::Mat format.
   * @param det_results the output results
   * @param conf_thresh confidence threshold
   * @param isRGB if the input is rgb format. Will flip channels if `isRGB` == false.
   * @return true
   * @return false
   */
  bool Detect(const cv::Mat       &input_image,
              std::vector<BBox2D> &det_results,
              float                conf_thresh,
              bool                 isRGB = false) noexcept;

  /**
   * @brief Run the detection processing in asynchronous mode.
   *
   * @param input_image input image in cv::Mat format.
   * @param conf_thresh confidence threshold
   * @param isRGB if the input is rgb format. Will flip channels if `isRGB` == false. default=false.
   * @param cover_oldest whether cover the oldest package if the pipeline queue is full.
   * default=false.
   * @return std::future<std::vector<BBox2D>>
   */
  [[nodiscard]] std::future<std::vector<BBox2D>> DetectAsync(const cv::Mat &input_image,
                                                             float          conf_thresh,
                                                             bool           isRGB = false,
                                                             bool cover_oldest    = false) noexcept;

protected:
  // forbidden the access from outside to `BaseAsyncPipeline::PushPipeline`
  using BaseAsyncPipeline::PushPipeline;

  virtual ~BaseDetectionModel();

  std::shared_ptr<BaseInferCore> infer_core_{nullptr};

  static std::string detection_pipeline_name_;
};

/**
 * @brief Abstract factory class of detection_2d model.
 *
 */
class BaseDetection2DFactory {
public:
  virtual std::shared_ptr<BaseDetectionModel> Create() = 0;
};

} // namespace easy_deploy
