#include "image_processing_utils/image_processing_utils.hpp"

#include <cuda_runtime.h>

extern "C" void launch_resize_pad_norm(const unsigned char *src,
                                       int                  src_h,
                                       int                  src_w,
                                       int                  src_stride,
                                       int                  src_format_bgr,
                                       float               *dst,
                                       int                  dst_h,
                                       int                  dst_w,
                                       int                  pad_top,
                                       int                  pad_left,
                                       float                scale,
                                       float                mean0,
                                       float                mean1,
                                       float                mean2,
                                       float                val0,
                                       float                val1,
                                       float                val2,
                                       bool                 do_transpose,
                                       bool                 do_norm,
                                       int                  pad_value,
                                       float                pad_color0,
                                       float                pad_color1,
                                       float                pad_color2,
                                       cudaStream_t         stream);

namespace easy_deploy {

class ImageProcessingCudaResizePad : public IImageProcessing {
public:
  ImageProcessingCudaResizePad(ImageProcessingPadMode    pad_mode,
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
  ImageProcessingPadMode  pad_mode_;
  ImageProcessingPadValue pad_value_;
  std::vector<float>      mean_, val_, pad_color_;
  bool                    do_transpose_, do_norm_;
};

ImageProcessingCudaResizePad::ImageProcessingCudaResizePad(ImageProcessingPadMode    pad_mode,
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
      pad_color_(pad_color),
      do_transpose_(do_transpose),
      do_norm_(do_norm)
{}

float ImageProcessingCudaResizePad::Process(std::shared_ptr<IPipelineImageData> input_image_data,
                                            ITensor                            *tensor,
                                            int                                 dst_height,
                                            int                                 dst_width)
{
  tensor->SetBufferLocation(DataLocation::DEVICE);
  float *dst_ptr = tensor->Cast<float>(); // device ptr!

  const auto &image_data_info = input_image_data->GetImageDataInfo();
  const int   image_height    = image_data_info.image_height;
  const int   image_width     = image_data_info.image_width;

  const float s_w = static_cast<float>(dst_width) / image_width;
  const float s_h = static_cast<float>(dst_height) / image_height;
  int         fix_height, fix_width;
  float       scale;
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

  int top = 0, left = 0;
  switch (pad_mode_)
  {
    case LETTER_BOX:
      top  = (dst_height - fix_height) / 2;
      left = (dst_width - fix_width) / 2;
      break;
    case BOTTOM_RIGHT:
      top  = 0;
      left = 0;
      break;
    case TOP_RIGHT:
      top  = dst_height - fix_height;
      left = 0;
      break;
  }

  // 0. 输入数据上传到Device
  cv::Mat        input_image(image_height, image_width, CV_8UC3, image_data_info.data_pointer);
  unsigned char *d_input;
  cudaMalloc(&d_input, image_height * image_width * 3);
  cudaMemcpy(d_input, input_image.data, image_height * image_width * 3, cudaMemcpyHostToDevice);

  dim3 block(32, 16);
  dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

  // pad_color数组保护
  float pc0 = pad_color_.size() > 0 ? pad_color_[0] : 0.f;
  float pc1 = pad_color_.size() > 1 ? pad_color_[1] : 0.f;
  float pc2 = pad_color_.size() > 2 ? pad_color_[2] : 0.f;

  launch_resize_pad_norm(d_input, image_height, image_width, image_width * 3,
                         image_data_info.format == ImageDataFormat::BGR ? 1 : 0, dst_ptr,
                         dst_height, dst_width, top, left, scale, mean_[0], mean_[1], mean_[2],
                         val_[0], val_[1], val_[2], do_transpose_, do_norm_,
                         static_cast<int>(pad_value_), pc0, pc1, pc2, nullptr);
  cudaFree(d_input);
  cudaDeviceSynchronize();

  return scale;
}

std::shared_ptr<IImageProcessing> CreateCudaImageProcessingResizePad(
    ImageProcessingPadMode    pad_mode,
    ImageProcessingPadValue   pad_value,
    bool                      do_transpose,
    bool                      do_norm,
    const std::vector<float> &mean,
    const std::vector<float> &val,
    const std::vector<float> &pad_color)
{
  return std::make_shared<ImageProcessingCudaResizePad>(pad_mode, pad_value, do_transpose, do_norm,
                                                        mean, val, pad_color);
}

} // namespace easy_deploy
