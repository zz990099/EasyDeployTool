#include "test_utils/stereo_matching_test_utils.hpp"

#include "common_utils/image_drawer.hpp"

#include <gtest/gtest.h>

namespace easy_deploy {

void test_stereo_matching_algorithm_correctness(
    const std::shared_ptr<BaseStereoMatchingModel> &model,
    const std::string                              &left_image_path,
    const std::string                              &right_image_path,
    const std::string                              &test_visual_result_save_path)
{
  cv::Mat left_image  = cv::imread(left_image_path);
  cv::Mat right_image = cv::imread(right_image_path);
  ASSERT_TRUE(!left_image.empty()) << "Got invalid left_image_path : " << left_image_path;
  ASSERT_TRUE(!right_image.empty()) << "Got invalid right_image_path : " << right_image_path;

  cv::Mat disp;
  ASSERT_TRUE(model->ComputeDisp(left_image, right_image, disp));

  if (!test_visual_result_save_path.empty())
  {
    double minVal, maxVal;
    cv::minMaxLoc(disp, &minVal, &maxVal);
    cv::Mat normalized_disp_pred;
    disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                   -minVal * 255.0 / (maxVal - minVal));

    cv::Mat color_normalized_disp_pred;
    cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
    cv::imwrite(test_visual_result_save_path, color_normalized_disp_pred);
  }
}

void test_stereo_matching_algorithm_async_correctness(
    const std::shared_ptr<BaseStereoMatchingModel> &model,
    const std::string                              &left_image_path,
    const std::string                              &right_image_path,
    const std::string                              &test_visual_result_save_path)
{
  cv::Mat left_image  = cv::imread(left_image_path);
  cv::Mat right_image = cv::imread(right_image_path);
  ASSERT_TRUE(!left_image.empty()) << "Got invalid left_image_path : " << left_image_path;
  ASSERT_TRUE(!right_image.empty()) << "Got invalid right_image_path : " << right_image_path;

  model->InitPipeline();

  std::shared_ptr<std::future<cv::Mat>> shared_async_future;

  auto async_call_func = [&]() {
    shared_async_future = std::make_shared<std::future<cv::Mat>>(
        model->ComputeDispAsync(left_image, right_image));
  };

  std::thread async_call_thread(async_call_func);
  async_call_thread.join();
  ASSERT_TRUE(shared_async_future != nullptr && shared_async_future->valid())
      << " Got invalid future while calling async stereo matching API";

  cv::Mat disp;
  EXPECT_NO_THROW(disp = shared_async_future->get());

  if (!test_visual_result_save_path.empty())
  {
    double minVal, maxVal;
    cv::minMaxLoc(disp, &minVal, &maxVal);
    cv::Mat normalized_disp_pred;
    disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                   -minVal * 255.0 / (maxVal - minVal));

    cv::Mat color_normalized_disp_pred;
    cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
    cv::imwrite(test_visual_result_save_path, color_normalized_disp_pred);
  }
}

} // namespace easy_deploy
