#include "image_processing_utils/image_processing_utils.hpp"

namespace easy_deploy {

class ImageProcessingCpuResizePad : public IImageProcessing {
public:
  ImageProcessingCpuResizePad(ImageProcessingPadMode    pad_mode,
                              ImageProcessingPadValue   pad_value,
                              bool                      do_transpose = true,
                              bool                      do_norm      = true,
                              const std::vector<float> &mean         = {0, 0, 0},
                              const std::vector<float> &val          = {255, 255, 255},
                              const std::vector<float> &pad_color    = {0, 0, 0});

  float Process(std::shared_ptr<IPipelineImageData> input_image_data,
                ITensor                            *tensor,
                int                                 dst_height,
                int                                 dst_width) override;

private:
  void FlipChannelsWithNorm(const cv::Mat &image, float *dst_ptr, bool flip);
  void FlipChannelsWithoutNorm(const cv::Mat &image, u_char *dst_ptr, bool flip);
  void TransposeAndFilpWithNorm(const cv::Mat &image, float *dst_ptr, bool flip);
  void TransposeAndFilpWithoutNorm(const cv::Mat &image, u_char *dst_ptr, bool flip);

private:
  ImageProcessingPadMode   pad_mode_;
  ImageProcessingPadValue  pad_value_;
  const std::vector<float> mean_, val_;
  const bool               do_transpose_, do_norm_;
  const std::vector<float> pad_color_;
};

ImageProcessingCpuResizePad::ImageProcessingCpuResizePad(ImageProcessingPadMode    pad_mode,
                                                         ImageProcessingPadValue   pad_value,
                                                         bool                      do_transpose,
                                                         bool                      do_norm,
                                                         const std::vector<float> &mean,
                                                         const std::vector<float> &val,
                                                         const std::vector<float> &pad_color)
    : pad_mode_(pad_mode),
      pad_value_(pad_value),
      mean_(mean),
      val_(val),
      do_transpose_(do_transpose),
      do_norm_(do_norm),
      pad_color_(pad_color)
{}

float ImageProcessingCpuResizePad::Process(std::shared_ptr<IPipelineImageData> input_image_data,
                                           ITensor                            *tensor,
                                           int                                 dst_height,
                                           int                                 dst_width)
{
  // 0. Make sure read/write on the host-side memory buffer
  tensor->SetBufferLocation(DataLocation::HOST);
  float      *dst_ptr         = tensor->Cast<float>();
  const auto &image_data_info = input_image_data->GetImageDataInfo();
  const int   image_height    = image_data_info.image_height;
  const int   image_width     = image_data_info.image_width;
  int         fix_height, fix_width;
  float       scale;

  const float s_w = static_cast<float>(dst_width) / image_width;
  const float s_h = static_cast<float>(dst_height) / image_height;

  if (s_h < s_w)
  {
    fix_height = dst_height;
    scale      = s_h;
    fix_width  = static_cast<int>(image_width * scale);
  } else
  {
    fix_width  = dst_width;
    scale      = s_w;
    fix_height = static_cast<int>(image_height * scale);
  }

  // 2. rebuild the cv::Mat format image
  cv::Mat input_image(image_height, image_width, CV_8UC3, image_data_info.data_pointer);

  // 3. resize and padding to the left-top
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, {fix_width, fix_height});

  int top = 0, bottom = 0, left = 0, right = 0;
  switch (pad_mode_)
  {
    case LETTER_BOX:
      top    = (dst_height - fix_height) / 2;
      bottom = dst_height - fix_height - top;
      left   = (dst_width - fix_width) / 2;
      right  = dst_width - fix_width - left;
      break;
    case BOTTOM_RIGHT:
      top    = 0;
      bottom = dst_height - fix_height;
      left   = 0;
      right  = dst_width - fix_width;
      break;
    case TOP_RIGHT:
      top    = dst_height - fix_height;
      bottom = 0;
      left   = 0;
      right  = dst_width - fix_width;
      break;
    default:
      throw std::runtime_error("[ImageProcessingCpu] Unkown pad mode!");
      break;
  }

  cv::Mat dst_image;
  switch (pad_value_)
  {
    case EDGE:
      cv::copyMakeBorder(resized_image, dst_image, top, bottom, left, right, cv::BORDER_REPLICATE);
      break;
    case CONSTANT:
      cv::copyMakeBorder(resized_image, dst_image, top, bottom, left, right, cv::BORDER_CONSTANT,
                         (image_data_info.image_channels == 1
                              ? cv::Scalar{pad_color_[0]}
                              : cv::Scalar{pad_color_[0], pad_color_[1], pad_color_[2]}));
      break;
    default:
      throw std::runtime_error("[ImageProcessingCpu] Unkown pad value!");
      break;
  }

  if (!do_transpose_)
  {
    // 4. flip and norm
    if (do_norm_)
    {
      FlipChannelsWithNorm(dst_image, dst_ptr, image_data_info.format == ImageDataFormat::BGR);
    } else
    {
      FlipChannelsWithoutNorm(dst_image, reinterpret_cast<u_char *>(dst_ptr),
                              image_data_info.format == ImageDataFormat::BGR);
    }
  } else
  {
    // 5. transpose flip and norm
    if (do_norm_)
    {
      TransposeAndFilpWithNorm(dst_image, dst_ptr, image_data_info.format == ImageDataFormat::BGR);
    } else
    {
      TransposeAndFilpWithoutNorm(dst_image, reinterpret_cast<u_char *>(dst_ptr),
                                  image_data_info.format == ImageDataFormat::BGR);
    }
  }

  return scale;
}

void ImageProcessingCpuResizePad::FlipChannelsWithNorm(const cv::Mat &image,
                                                       float         *dst_ptr,
                                                       bool           flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  for (int r = 0; r < rows; ++r)
  {
    size_t offset = (r * cols) * 3;
    for (int c = 0; c < cols; ++c)
    {
      size_t idx       = offset + c * 3;
      dst_ptr[idx + 0] = (image.data[idx + r_idx] - mean_[r_idx]) / val_[r_idx];
      dst_ptr[idx + 1] = (image.data[idx + g_idx] - mean_[g_idx]) / val_[g_idx];
      dst_ptr[idx + 2] = (image.data[idx + b_idx] - mean_[b_idx]) / val_[b_idx];
    }
  }
}

void ImageProcessingCpuResizePad::FlipChannelsWithoutNorm(const cv::Mat &image,
                                                          u_char        *dst_ptr,
                                                          bool           flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  for (int r = 0; r < rows; ++r)
  {
    size_t offset = r * cols * 3;
    for (int c = 0; c < cols; ++c)
    {
      size_t idx       = offset + c * 3;
      dst_ptr[idx + 0] = image.data[idx + r_idx];
      dst_ptr[idx + 1] = image.data[idx + g_idx];
      dst_ptr[idx + 2] = image.data[idx + b_idx];
    }
  }
}

void ImageProcessingCpuResizePad::TransposeAndFilpWithNorm(const cv::Mat &image,
                                                           float         *dst_ptr,
                                                           bool           flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  const int single_channel_pixel_size = rows * cols;
  for (int r = 0; r < rows; ++r)
  {
    uchar *pixel_ptr    = image.data + r * image.step;
    int    offset_start = cols * r;
    for (int c = 0; c < cols; ++c)
    {
      int offset_idx = offset_start + c;

      dst_ptr[offset_idx + 0 * single_channel_pixel_size] =
          (pixel_ptr[r_idx] - mean_[r_idx]) / val_[r_idx];

      dst_ptr[offset_idx + 1 * single_channel_pixel_size] =
          (pixel_ptr[g_idx] - mean_[g_idx]) / val_[g_idx];

      dst_ptr[offset_idx + 2 * single_channel_pixel_size] =
          (pixel_ptr[b_idx] - mean_[b_idx]) / val_[b_idx];

      pixel_ptr += 3;
    }
  }
}

void ImageProcessingCpuResizePad::TransposeAndFilpWithoutNorm(const cv::Mat &image,
                                                              u_char        *dst_ptr,
                                                              bool           flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  const int single_channel_pixel_size = rows * cols;
  for (int r = 0; r < rows; ++r)
  {
    uchar *pixel_ptr    = image.data + r * image.step;
    int    offset_start = cols * r;
    for (int c = 0; c < cols; ++c)
    {
      int offset_idx = offset_start + c;

      dst_ptr[offset_idx + 0 * single_channel_pixel_size] = pixel_ptr[r_idx];
      dst_ptr[offset_idx + 1 * single_channel_pixel_size] = pixel_ptr[g_idx];
      dst_ptr[offset_idx + 2 * single_channel_pixel_size] = pixel_ptr[b_idx];

      pixel_ptr += 3;
    }
  }
}

std::shared_ptr<IImageProcessing> CreateCpuImageProcessingResizePad(
    ImageProcessingPadMode    pad_mode,
    ImageProcessingPadValue   pad_value,
    bool                      do_transpose,
    bool                      do_norm,
    const std::vector<float> &mean,
    const std::vector<float> &val,
    const std::vector<float> &pad_color)
{
  return std::make_shared<ImageProcessingCpuResizePad>(pad_mode, pad_value, do_transpose, do_norm,
                                                       mean, val);
}

struct ImageProcessingCpuResizePadParams {
  ImageProcessingPadMode  pad_mode;
  ImageProcessingPadValue pad_value;
  bool                    do_transpose;
  bool                    do_norm;
  std::vector<float>      mean;
  std::vector<float>      val;
  std::vector<float>      pad_color;
};

class Detection2DPreprocessCpuFactory : public IImageProcessingFactory {
public:
  Detection2DPreprocessCpuFactory(const ImageProcessingCpuResizePadParams &params) : params_(params)
  {}

  std::shared_ptr<IImageProcessing> Create() override
  {
    return CreateCpuImageProcessingResizePad(params_.pad_mode, params_.pad_value,
                                             params_.do_transpose, params_.do_norm, params_.mean,
                                             params_.val, params_.pad_color);
  }

private:
  const ImageProcessingCpuResizePadParams params_;
};

std::shared_ptr<IImageProcessingFactory> CreateCpuDetPreProcessFactory(
    ImageProcessingPadMode    pad_mode,
    ImageProcessingPadValue   pad_value,
    bool                      do_transpose,
    bool                      do_norm,
    const std::vector<float> &mean,
    const std::vector<float> &val,
    const std::vector<float> &pad_color)
{
  ImageProcessingCpuResizePadParams params;
  params.pad_mode     = pad_mode;
  params.pad_value    = pad_value;
  params.mean         = mean;
  params.val          = val;
  params.do_transpose = do_transpose;
  params.do_norm      = do_norm;
  params.pad_color    = pad_color;

  return std::make_shared<Detection2DPreprocessCpuFactory>(params);
}

} // namespace easy_deploy
