cmake_minimum_required(VERSION 3.8)
project(benchmark_utils)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(benchmark REQUIRED)

set(source_file
    src/detection_2d_benchmark_utils.cpp
    src/sam_benchmark_utils.cpp
    src/stereo_matching_benchmark_utils.cpp
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
  benchmark::benchmark
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
