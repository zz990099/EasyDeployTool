#pragma once

#include "common_utils/types.hpp"

#include <stdint.h>

namespace easy_deploy {

/**
 * @brief A abstract class of image data. Needed by pipeline processing. Useful when data is
 * on device or other location which host cant read/write data directly. Could enable the
 * zero-copy feature if needed.
 *
 */
class IPipelineImageData {
public:
  struct ImageDataInfo {
    uint8_t        *data_pointer;
    int             image_height;
    int             image_width;
    int             image_channels;
    DataLocation    location;
    ImageDataFormat format;
  };
  virtual const ImageDataInfo &GetImageDataInfo() const = 0;

protected:
  virtual ~IPipelineImageData() = default;
};

} // namespace easy_deploy
