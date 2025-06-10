cmake_minimum_required(VERSION 3.8)
project(deploy_utils)

add_subdirectory(common_utils)
add_subdirectory(image_processing_utils)

if (BUILD_TESTING)
  add_subdirectory(test_utils)
endif()

if (BUILD_BENCHMARK)
  add_subdirectory(benchmark_utils)
endif()

if (BUILD_EVAL)
  add_subdirectory(eval_utils)
endif()
