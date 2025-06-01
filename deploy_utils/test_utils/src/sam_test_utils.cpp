#include "test_utils/sam_test_utils.hpp"

#include "image_drawer.hpp"

#include <gtest/gtest.h>

namespace test_utils {

using namespace sam;

void test_sam_algorithm_correctness_with_points(const std::shared_ptr<BaseSamModel>    &model,
                                                const std::vector<std::pair<int, int>> &points,
                                                const std::vector<int>                 &labels,
                                                const std::string &test_image_path,
                                                const std::string &test_visual_result_save_path)
{
  cv::Mat test_image = cv::imread(test_image_path);
  ASSERT_TRUE(!test_image.empty()) << "Got invalid test_image_path : " << test_image_path;

  cv::Mat masks;
  ASSERT_TRUE(model->GenerateMask(test_image, points, labels, masks, false));
  ASSERT_TRUE(!masks.empty());

  if (!test_visual_result_save_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(test_image.clone()));
    helper.addRedMaskToForeground(masks);
    cv::imwrite(test_visual_result_save_path, *helper.getImage());
  }
}

void test_sam_algorithm_correctness_with_boxes(const std::shared_ptr<BaseSamModel> &model,
                                               const std::vector<BBox2D>           &boxes,
                                               const std::string                   &test_image_path,
                                               const std::string &test_visual_result_save_path)
{
  cv::Mat test_image = cv::imread(test_image_path);
  ASSERT_TRUE(!test_image.empty()) << "Got invalid test_image_path : " << test_image_path;

  cv::Mat masks;
  ASSERT_TRUE(model->GenerateMask(test_image, boxes, masks, false));
  ASSERT_TRUE(!masks.empty());

  if (!test_visual_result_save_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(test_image.clone()));
    helper.addRedMaskToForeground(masks);
    cv::imwrite(test_visual_result_save_path, *helper.getImage());
  }
}

void test_sam_algorithm_async_correctness_with_points(
    const std::shared_ptr<BaseSamModel>    &model,
    const std::vector<std::pair<int, int>> &points,
    const std::vector<int>                 &labels,
    const std::string                      &test_image_path,
    const std::string                      &test_visual_result_save_path)
{
  cv::Mat test_image = cv::imread(test_image_path);
  ASSERT_TRUE(!test_image.empty()) << "Got invalid test_image_path : " << test_image_path;

  model->InitPipeline();

  std::shared_ptr<std::future<cv::Mat>> shared_async_future;

  auto async_call_func = [&]() {
    shared_async_future = std::make_shared<std::future<cv::Mat>>(
        model->GenerateMaskAsync(test_image, points, labels, false, false));
  };

  std::thread async_call_thread(async_call_func);
  async_call_thread.join();
  ASSERT_TRUE(shared_async_future != nullptr && shared_async_future->valid())
      << " Got invalid future while calling async sam API";

  cv::Mat masks;
  EXPECT_NO_THROW(masks = shared_async_future->get());

  if (!test_visual_result_save_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(test_image.clone()));
    helper.addRedMaskToForeground(masks);
    cv::imwrite(test_visual_result_save_path, *helper.getImage());
  }
}

void test_sam_algorithm_async_correctness_with_boxes(
    const std::shared_ptr<BaseSamModel> &model,
    const std::vector<BBox2D>           &boxes,
    const std::string                   &test_image_path,
    const std::string                   &test_visual_result_save_path)
{
  cv::Mat test_image = cv::imread(test_image_path);
  ASSERT_TRUE(!test_image.empty()) << "Got invalid test_image_path : " << test_image_path;

  model->InitPipeline();

  std::shared_ptr<std::future<cv::Mat>> shared_async_future;

  auto async_call_func = [&]() {
    shared_async_future = std::make_shared<std::future<cv::Mat>>(
        model->GenerateMaskAsync(test_image, boxes, false, false));
  };

  std::thread async_call_thread(async_call_func);
  async_call_thread.join();
  ASSERT_TRUE(shared_async_future != nullptr && shared_async_future->valid())
      << " Got invalid future while calling async sam API";

  cv::Mat masks;
  EXPECT_NO_THROW(masks = shared_async_future->get());

  if (!test_visual_result_save_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(test_image.clone()));
    helper.addRedMaskToForeground(masks);
    cv::imwrite(test_visual_result_save_path, *helper.getImage());
  }
}

} // namespace test_utils
