#pragma once

#include "deploy_core/async_pipeline.hpp"

#include <opencv2/opencv.hpp>

#include <unordered_map>

namespace easy_deploy {

/**
 * @brief A simple wrapper of cv::Mat. Used in pipeline.
 *
 */
class PipelineCvImageWrapper : public IPipelineImageData {
public:
  PipelineCvImageWrapper(const cv::Mat &cv_image, bool isRGB = false) : inner_cv_image(cv_image)
  {
    image_data_info.data_pointer   = cv_image.data;
    image_data_info.format         = isRGB ? ImageDataFormat::RGB : ImageDataFormat::BGR;
    image_data_info.image_height   = cv_image.rows;
    image_data_info.image_width    = cv_image.cols;
    image_data_info.image_channels = cv_image.channels();
    image_data_info.location       = DataLocation::HOST;
  }

  const ImageDataInfo &GetImageDataInfo() const
  {
    return image_data_info;
  }

private:
  IPipelineImageData::ImageDataInfo image_data_info;
  const cv::Mat                     inner_cv_image;
};

} // namespace easy_deploy
