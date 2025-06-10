#pragma once

#include "deploy_core/base_sam.hpp"

namespace easy_deploy {

void test_sam_algorithm_correctness_with_points(
    const std::shared_ptr<BaseSamModel>    &model,
    const std::vector<std::pair<int, int>> &points,
    const std::vector<int>                 &labels,
    const std::string                      &test_image_path,
    const std::string                      &test_visual_result_save_path = "");

void test_sam_algorithm_correctness_with_boxes(
    const std::shared_ptr<BaseSamModel> &model,
    const std::vector<BBox2D>           &boxes,
    const std::string                   &test_image_path,
    const std::string                   &test_visual_result_save_path = "");

void test_sam_algorithm_async_correctness_with_points(
    const std::shared_ptr<BaseSamModel>    &model,
    const std::vector<std::pair<int, int>> &points,
    const std::vector<int>                 &labels,
    const std::string                      &test_image_path,
    const std::string                      &test_visual_result_save_path = "");

void test_sam_algorithm_async_correctness_with_boxes(
    const std::shared_ptr<BaseSamModel> &model,
    const std::vector<BBox2D>           &boxes,
    const std::string                   &test_image_path,
    const std::string                   &test_visual_result_save_path = "");

} // namespace easy_deploy
