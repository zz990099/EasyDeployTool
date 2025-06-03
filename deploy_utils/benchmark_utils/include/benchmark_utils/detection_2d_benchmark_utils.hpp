#pragma once

#include "deploy_core/base_detection.h"

#include <benchmark/benchmark.h>

namespace benchmark_utils {

void benchmark_detection_2d_sync(benchmark::State                                        &state,
                                 const std::shared_ptr<detection_2d::BaseDetectionModel> &model);

void benchmark_detection_2d_async(benchmark::State                                        &state,
                                  const std::shared_ptr<detection_2d::BaseDetectionModel> &model);

} // namespace benchmark_utils
