cmake_minimum_required(VERSION 3.8)
project(common_utils)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

set(source_file
  src/log.cpp
)

include_directories(
  include
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

if (ENABLE_DEBUG_OUTPUT)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_DEBUG_OUTPUT)
endif()
