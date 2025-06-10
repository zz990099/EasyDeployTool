#pragma once

#include "deploy_core/base_sam.hpp"

#include <benchmark/benchmark.h>

namespace easy_deploy {

void benchmark_sam_sync(benchmark::State &state, const std::shared_ptr<BaseSamModel> &model);

void benchmark_sam_async(benchmark::State &state, const std::shared_ptr<BaseSamModel> &model);

} // namespace easy_deploy
