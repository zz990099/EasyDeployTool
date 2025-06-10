#include "deploy_core/base_stereo.hpp"
#include "deploy_core/wrapper.hpp"

namespace easy_deploy {

const std::string BaseMonoStereoModel::mono_stereo_pipeline_name_ = "stereo_pipeline";

BaseMonoStereoModel::BaseMonoStereoModel(const std::shared_ptr<BaseInferCore> &inference_core)
    : inference_core_(inference_core)
{
  auto preprocess_block = BaseAsyncPipeline::BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return PreProcess(unit); }, "[StereoPreProcess]");

  auto postprocess_block = BaseAsyncPipeline::BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return PostProcess(unit); }, "[StereoPostProcess]");

  const auto &inference_core_context = inference_core->GetPipelineContext();

  BaseAsyncPipeline::ConfigPipeline(mono_stereo_pipeline_name_,
                                    {preprocess_block, inference_core_context, postprocess_block});
}

bool BaseMonoStereoModel::ComputeDepth(const cv::Mat &input_image, cv::Mat &disp_output)
{
  CHECK_STATE(!input_image.empty(),
              "[BaseMonoStereoModel] `ComputeDepth` Got invalid input images !!!");

  auto package              = std::make_shared<MonoStereoPipelinePackage>();
  package->input_image_data = std::make_shared<PipelineCvImageWrapper>(input_image);
  package->infer_buffer     = inference_core_->GetBuffer(true);
  CHECK_STATE(package->infer_buffer != nullptr,
              "[BaseMonoStereoModel] `ComputeDisp` Got invalid inference core buffer ptr !!!");

  MESSURE_DURATION_AND_CHECK_STATE(
      PreProcess(package), "[BaseMonoStereoModel] `ComputeDisp` Failed execute PreProcess !!!");
  MESSURE_DURATION_AND_CHECK_STATE(
      inference_core_->SyncInfer(package->infer_buffer.get()),
      "[BaseMonoStereoModel] `ComputeDisp` Failed execute inference sync infer !!!");
  MESSURE_DURATION_AND_CHECK_STATE(
      PostProcess(package), "[BaseMonoStereoModel] `ComputeDisp` Failed execute PostProcess !!!");

  disp_output = std::move(package->depth);

  return true;
}

std::future<cv::Mat> BaseMonoStereoModel::ComputeDepthAsync(const cv::Mat &input_image)
{
  if (input_image.empty())
  {
    LOG_ERROR("[BaseMonoStereoModel] `ComputeDepthAsync` Got invalid input images !!!");
    return std::future<cv::Mat>();
  }

  auto package              = std::make_shared<MonoStereoPipelinePackage>();
  package->input_image_data = std::make_shared<PipelineCvImageWrapper>(input_image);
  package->infer_buffer     = inference_core_->GetBuffer(true);
  if (package->infer_buffer == nullptr)
  {
    LOG_ERROR(
        "[BaseMonoStereoModel] `ComputeDepthAsync` Got invalid inference core buffer ptr !!!");
    return std::future<cv::Mat>();
  }

  return BaseAsyncPipeline::PushPipeline(mono_stereo_pipeline_name_, package);
}

} // namespace easy_deploy
