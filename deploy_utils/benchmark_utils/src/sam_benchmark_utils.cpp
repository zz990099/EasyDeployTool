#include "benchmark_utils/sam_benchmark_utils.hpp"

#include <benchmark/benchmark.h>

namespace benchmark_utils {

using namespace sam;

void benchmark_sam_sync(benchmark::State &state, const std::shared_ptr<BaseSamModel> &model)
{
  cv::Mat dummy_input(480, 720, CV_8UC3);

  // 基准测试主循环
  for (auto _ : state)
  {
    for (size_t i = 0; i < state.range(0); ++i)
    {
      cv::Mat masks;
      CHECK(model->GenerateMask(dummy_input.clone(), {{200, 200}}, {1}, masks, false));
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

void benchmark_sam_async(benchmark::State &state, const std::shared_ptr<BaseSamModel> &model)
{
  cv::Mat dummy_input(480, 720, CV_8UC3);
  model->InitPipeline();

  // 基准测试主循环
  for (auto _ : state)
  {
    std::vector<std::future<cv::Mat>> futs;
    for (size_t i = 0; i < state.range(0); ++i)
    {
      auto fut = model->GenerateMaskAsync(dummy_input.clone(), {{200, 200}}, {1}, false, false);
      CHECK(fut.valid());
      futs.push_back(std::move(fut));
    }
    for (auto &f : futs) f.get();
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

} // namespace benchmark_utils
