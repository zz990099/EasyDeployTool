#pragma once

#include "common_utils/log.hpp"

namespace easy_deploy {

/**
 * @brief Defination of common 2D bounding box
 *
 * @param x center of bbox `x`
 * @param y center of bbox `y`
 * @param w width of bbox
 * @param h height of bbox
 * @param conf confidence of bbox
 * @param cls classification of bbox
 */
struct BBox2D {
  float x;
  float y;
  float w;
  float h;
  float conf;
  float cls;
};

/**
 * @brief Enum of data loacation
 *
 * @param HOST data is host accessable
 * @param DEVICE data is device accessable, means host cant read/write the data buffer directly
 * @param UNKOWN some other condition
 *
 */
enum DataLocation { HOST = 0, DEVICE = 1, UNKOWN = 2 };

/**
 * @brief Defination of common image format.
 *
 */
enum ImageDataFormat { YUV = 0, RGB = 1, BGR = 2, GRAY = 3 };

// some macro
#define CHECK_STATE(state, fmt, ...) \
  {                                  \
    if (!(state))                    \
    {                                \
      LOG_ERROR(fmt, ##__VA_ARGS__); \
      return false;                  \
    }                                \
  }

#define CHECK_STATE_THROW(state, fmt, ...)               \
  {                                                      \
    if (!(state))                                        \
    {                                                    \
      LOG_ERROR(fmt, ##__VA_ARGS__);                     \
      char _msg[1024];                                   \
      FormatMsg(_msg, sizeof(_msg), fmt, ##__VA_ARGS__); \
      throw std::runtime_error(_msg);                    \
    }                                                    \
  }

#define MESSURE_DURATION(run)                                                              \
  {                                                                                        \
    auto start = std::chrono::high_resolution_clock::now();                                \
    (run);                                                                                 \
    auto end = std::chrono::high_resolution_clock::now();                                  \
    LOG_DEBUG("%s cost(us): %ld", #run,                                                    \
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()); \
  }

#define MESSURE_DURATION_AND_CHECK_STATE(run, fmt, ...)                                    \
  {                                                                                        \
    auto start = std::chrono::high_resolution_clock::now();                                \
    CHECK_STATE((run), fmt, ##__VA_ARGS__);                                                \
    auto end = std::chrono::high_resolution_clock::now();                                  \
    LOG_DEBUG("%s cost(us): %ld", #run,                                                    \
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()); \
  }

} // namespace easy_deploy
