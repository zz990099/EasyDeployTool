#include "benchmark_utils/detection_2d_benchmark_utils.hpp"

#include <benchmark/benchmark.h>

namespace benchmark_utils {

using namespace detection_2d;

void benchmark_detection_2d_sync(benchmark::State                          &state,
                                 const std::shared_ptr<BaseDetectionModel> &model)
{
  cv::Mat dummy_input(480, 720, CV_8UC3);

  // 基准测试主循环
  for (auto _ : state)
  {
    for (size_t i = 0; i < state.range(0); ++i)
    {
      std::vector<BBox2D> det_results;
      model->Detect(dummy_input.clone(), det_results, 0.4, false);
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

void benchmark_detection_2d_async(benchmark::State                          &state,
                                  const std::shared_ptr<BaseDetectionModel> &model)
{
  cv::Mat dummy_input(480, 720, CV_8UC3);
  model->InitPipeline();

  // 基准测试主循环
  for (auto _ : state)
  {
    std::vector<std::future<std::vector<BBox2D>>> futs;
    for (size_t i = 0; i < state.range(0); ++i)
    {
      auto fut = model->DetectAsync(dummy_input.clone(), 0.4, false, false);
      CHECK(fut.valid());
      futs.push_back(std::move(fut));
    }
    for (auto &f : futs) f.get();
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

} // namespace benchmark_utils
