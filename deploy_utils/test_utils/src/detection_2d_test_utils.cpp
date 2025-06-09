#include "test_utils/detection_2d_test_utils.hpp"

#include "common_utils/image_drawer.hpp"

#include <gtest/gtest.h>

namespace test_utils {

using namespace detection_2d;
using namespace common_utils;

void test_detection_2d_algorithm_correctness(const std::shared_ptr<BaseDetectionModel> &model,
                                             const std::string &test_image_path,
                                             float              conf_threshold,
                                             size_t             expected_obj_num,
                                             const std::string &test_visual_result_save_path)
{
  cv::Mat test_image = cv::imread(test_image_path);
  ASSERT_TRUE(!test_image.empty()) << "Got invalid test_image_path : " << test_image_path;

  std::vector<BBox2D> det_results;
  ASSERT_TRUE(model->Detect(test_image, det_results, conf_threshold, false));

  if (!test_visual_result_save_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(test_image.clone()));
    for (const auto &box : det_results)
    {
      helper.drawRect2D(box);
    }
    cv::imwrite(test_visual_result_save_path, *helper.getImage());
  }

  ASSERT_TRUE(det_results.size() == expected_obj_num)
      << "Got unexpected obj num, expected : " << expected_obj_num
      << ", but got : " << det_results.size();
}

void test_detection_2d_algorithm_async_correctness(const std::shared_ptr<BaseDetectionModel> &model,
                                                   const std::string &test_image_path,
                                                   float              conf_threshold,
                                                   size_t             expected_obj_num,
                                                   const std::string &test_visual_result_save_path)
{
  cv::Mat test_image = cv::imread(test_image_path);
  ASSERT_TRUE(!test_image.empty()) << "Got invalid test_image_path : " << test_image_path;

  model->InitPipeline();

  std::shared_ptr<std::future<std::vector<BBox2D>>> shared_async_future;

  auto async_call_func = [&]() {
    shared_async_future = std::make_shared<std::future<std::vector<BBox2D>>>(
        model->DetectAsync(test_image, conf_threshold, false));
  };

  std::thread async_call_thread(async_call_func);
  async_call_thread.join();
  ASSERT_TRUE(shared_async_future != nullptr && shared_async_future->valid())
      << " Got invalid future while calling async detection API";

  std::vector<BBox2D> det_results;
  EXPECT_NO_THROW(det_results = shared_async_future->get());

  if (!test_visual_result_save_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(test_image.clone()));
    for (const auto &box : det_results)
    {
      helper.drawRect2D(box);
    }
    cv::imwrite(test_visual_result_save_path, *helper.getImage());
  }

  ASSERT_TRUE(det_results.size() == expected_obj_num)
      << "Got unexpected obj num, expected : " << expected_obj_num
      << ", but got : " << det_results.size();
}

} // namespace test_utils
