#include "benchmark_utils/stereo_matching_benchmark_utils.hpp"

#include <benchmark/benchmark.h>
#include <glog/logging.h>

namespace easy_deploy {

void benchmark_stereo_matching_sync(benchmark::State                               &state,
                                    const std::shared_ptr<BaseStereoMatchingModel> &model)
{
  cv::Mat dummy_input(480, 720, CV_8UC3);

  // 基准测试主循环
  for (auto _ : state)
  {
    for (size_t i = 0; i < state.range(0); ++i)
    {
      cv::Mat disp;
      CHECK(model->ComputeDisp(dummy_input, dummy_input, disp));
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

void benchmark_stereo_matching_async(benchmark::State                               &state,
                                     const std::shared_ptr<BaseStereoMatchingModel> &model)
{
  cv::Mat dummy_input(480, 720, CV_8UC3);
  model->InitPipeline();

  // 基准测试主循环
  for (auto _ : state)
  {
    std::vector<std::future<cv::Mat>> futs;
    for (size_t i = 0; i < state.range(0); ++i)
    {
      auto fut = model->ComputeDispAsync(dummy_input.clone(), dummy_input.clone());
      CHECK(fut.valid());
      futs.push_back(std::move(fut));
    }
    for (auto &f : futs) f.get();
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

} // namespace easy_deploy
