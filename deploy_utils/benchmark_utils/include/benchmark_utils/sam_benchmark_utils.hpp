#pragma once

#include "deploy_core/base_sam.h"

#include <benchmark/benchmark.h>

namespace benchmark_utils {

void benchmark_sam_sync(benchmark::State &state, const std::shared_ptr<sam::BaseSamModel> &model);

void benchmark_sam_async(benchmark::State &state, const std::shared_ptr<sam::BaseSamModel> &model);

} // namespace benchmark_utils
