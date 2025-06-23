#pragma once

#include "deploy_core/base_stereo.hpp"

namespace easy_deploy {

void test_stereo_matching_algorithm_correctness(
    const std::shared_ptr<BaseStereoMatchingModel> &model,
    const std::string                              &left_image_path,
    const std::string                              &right_image_path,
    const std::string                              &test_visual_result_save_path = "");

void test_stereo_matching_algorithm_async_correctness(
    const std::shared_ptr<BaseStereoMatchingModel> &model,
    const std::string                              &left_image_path,
    const std::string                              &right_image_path,
    const std::string                              &test_visual_result_save_path = "");

} // namespace easy_deploy
