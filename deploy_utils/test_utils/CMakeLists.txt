cmake_minimum_required(VERSION 3.8)
project(test_utils)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(source_file
    src/detection_2d_test_utils.cpp
    src/sam_test_utils.cpp
    src/stereo_matching_test_utils.cpp
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
