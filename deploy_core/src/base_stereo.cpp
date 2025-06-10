#include "deploy_core/base_stereo.hpp"
#include "deploy_core/wrapper.hpp"

namespace easy_deploy {

const std::string BaseStereoMatchingModel::stereo_pipeline_name_ = "stereo_pipeline";

BaseStereoMatchingModel::BaseStereoMatchingModel(
    const std::shared_ptr<BaseInferCore> &inference_core)
    : inference_core_(inference_core)
{
  auto preprocess_block = BaseAsyncPipeline::BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return PreProcess(unit); }, "[StereoPreProcess]");

  auto postprocess_block = BaseAsyncPipeline::BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return PostProcess(unit); }, "[StereoPostProcess]");

  const auto &inference_core_context = inference_core->GetPipelineContext();

  BaseAsyncPipeline::ConfigPipeline(stereo_pipeline_name_,
                                    {preprocess_block, inference_core_context, postprocess_block});
}

bool BaseStereoMatchingModel::ComputeDisp(const cv::Mat &left_image,
                                          const cv::Mat &right_image,
                                          cv::Mat       &disp_output)
{
  CHECK_STATE(!left_image.empty() && !right_image.empty(),
              "[BaseStereoMatchingModel] `ComputeDisp` Got invalid input images !!!");

  auto package              = std::make_shared<StereoPipelinePackage>();
  package->left_image_data  = std::make_shared<PipelineCvImageWrapper>(left_image);
  package->right_image_data = std::make_shared<PipelineCvImageWrapper>(right_image);
  package->infer_buffer     = inference_core_->GetBuffer(true);
  CHECK_STATE(package->infer_buffer != nullptr,
              "[BaseStereoMatchingModel] `ComputeDisp` Got invalid inference core buffer ptr !!!");

  MESSURE_DURATION_AND_CHECK_STATE(
      PreProcess(package), "[BaseStereoMatchingModel] `ComputeDisp` Failed execute PreProcess !!!");
  MESSURE_DURATION_AND_CHECK_STATE(
      inference_core_->SyncInfer(package->infer_buffer.get()),
      "[BaseStereoMatchingModel] `ComputeDisp` Failed execute inference sync infer !!!");
  MESSURE_DURATION_AND_CHECK_STATE(
      PostProcess(package),
      "[BaseStereoMatchingModel] `ComputeDisp` Failed execute PostProcess !!!");

  disp_output = std::move(package->disp);

  return true;
}

std::future<cv::Mat> BaseStereoMatchingModel::ComputeDispAsync(const cv::Mat &left_image,
                                                               const cv::Mat &right_image)
{
  if (left_image.empty() || right_image.empty())
  {
    LOG_ERROR("[BaseStereoMatchingModel] `ComputeDispAsync` Got invalid input images !!!");
    return std::future<cv::Mat>();
  }

  auto package              = std::make_shared<StereoPipelinePackage>();
  package->left_image_data  = std::make_shared<PipelineCvImageWrapper>(left_image);
  package->right_image_data = std::make_shared<PipelineCvImageWrapper>(right_image);
  package->infer_buffer     = inference_core_->GetBuffer(true);
  if (package->infer_buffer == nullptr)
  {
    LOG_ERROR(
        "[BaseStereoMatchingModel] `ComputeDispAsync` Got invalid inference core buffer ptr !!!");
    return std::future<cv::Mat>();
  }

  return BaseAsyncPipeline::PushPipeline(stereo_pipeline_name_, package);
}

} // namespace easy_deploy
