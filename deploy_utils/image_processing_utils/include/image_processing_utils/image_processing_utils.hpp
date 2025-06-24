#pragma once

#include "deploy_core/base_detection.hpp"

namespace easy_deploy {

enum ImageProcessingPadMode { LETTER_BOX = 0, BOTTOM_RIGHT = 1, TOP_RIGHT = 2 };
enum ImageProcessingPadValue { EDGE = 0, CONSTANT = 1 };

class IImageProcessing {
public:
  virtual float Process(std::shared_ptr<IPipelineImageData> input_image_data,
                        ITensor                            *tensor,
                        int                                 dst_height,
                        int                                 dst_width) = 0;
};

class IImageProcessingFactory {
public:
  virtual std::shared_ptr<IImageProcessing> Create() = 0;
};

std::shared_ptr<IImageProcessing> CreateCpuImageProcessingResizePad(
    ImageProcessingPadMode    pad_mode     = ImageProcessingPadMode::BOTTOM_RIGHT,
    ImageProcessingPadValue   pad_value    = ImageProcessingPadValue::EDGE,
    bool                      do_transpose = true,
    bool                      do_norm      = true,
    const std::vector<float> &mean         = {0, 0, 0},
    const std::vector<float> &val          = {255, 255, 255},
    const std::vector<float> &pad_color    = {0, 0, 0});

std::shared_ptr<IImageProcessingFactory> CreateCpuImageProcessingResizePadFactory(
    ImageProcessingPadMode    pad_mode     = ImageProcessingPadMode::BOTTOM_RIGHT,
    ImageProcessingPadValue   pad_value    = ImageProcessingPadValue::EDGE,
    bool                      do_transpose = true,
    bool                      do_norm      = true,
    const std::vector<float> &mean         = {0, 0, 0},
    const std::vector<float> &val          = {255, 255, 255},
    const std::vector<float> &pad_color    = {0, 0, 0});


std::shared_ptr<IImageProcessing> CreateCudaImageProcessingResizePad(
    ImageProcessingPadMode    pad_mode     = ImageProcessingPadMode::BOTTOM_RIGHT,
    ImageProcessingPadValue   pad_value    = ImageProcessingPadValue::EDGE,
    bool                      do_transpose = true,
    bool                      do_norm      = true,
    const std::vector<float> &mean         = {0, 0, 0},
    const std::vector<float> &val          = {255, 255, 255},
    const std::vector<float> &pad_color    = {0, 0, 0});

} // namespace easy_deploy
