cmake_minimum_required(VERSION 3.8)
project(eval_utils)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(source_file
    src/detection_2d_eval_utils.cpp
    src/stereo_matching_eval_utils.cpp
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  deploy_core
  common_utils
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ DESTINATION ${CMAKE_BINARY_DIR}/scripts/eval_utils/)
