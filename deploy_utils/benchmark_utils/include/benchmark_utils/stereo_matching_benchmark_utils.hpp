#pragma once

#include "deploy_core/base_stereo.hpp"

#include <benchmark/benchmark.h>

namespace easy_deploy {

void benchmark_stereo_matching_sync(benchmark::State                               &state,
                                    const std::shared_ptr<BaseStereoMatchingModel> &model);

void benchmark_stereo_matching_async(benchmark::State                               &state,
                                     const std::shared_ptr<BaseStereoMatchingModel> &model);

} // namespace easy_deploy
