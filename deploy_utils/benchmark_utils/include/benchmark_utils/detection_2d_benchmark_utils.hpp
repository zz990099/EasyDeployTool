#pragma once

#include "deploy_core/base_detection.hpp"

#include <benchmark/benchmark.h>

namespace easy_deploy {

void benchmark_detection_2d_sync(benchmark::State                          &state,
                                 const std::shared_ptr<BaseDetectionModel> &model);

void benchmark_detection_2d_async(benchmark::State                          &state,
                                  const std::shared_ptr<BaseDetectionModel> &model);

} // namespace easy_deploy
