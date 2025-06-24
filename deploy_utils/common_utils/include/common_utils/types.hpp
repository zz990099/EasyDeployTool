#pragma once

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


} // namespace easy_deploy
