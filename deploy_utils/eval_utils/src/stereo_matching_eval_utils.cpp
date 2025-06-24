#include "eval_utils/stereo_matching_eval_utils.hpp"

#include "common_utils/fs_utils.hpp"
#include "common_utils/progress_bar.hpp"
#include "common_utils/fs_utils.hpp"

#include <fstream>

namespace easy_deploy {

#define EVAL_STEREO_CHECK(state, fmt, ...)               \
  {                                                      \
    if (!(state))                                        \
    {                                                    \
      char _msg[1024];                                   \
      FormatMsg(_msg, sizeof(_msg), fmt, ##__VA_ARGS__); \
      throw std::runtime_error(_msg);                    \
    }                                                    \
  }

cv::Mat read_pfm(const std::string &path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    std::cerr << "Failed to open file: " << path << std::endl;
    return {};
  }

  std::string tag;
  file >> tag;

  EVAL_STEREO_CHECK(tag == "Pf", "Unknown PFM format: %s", tag.c_str());
  int channels = 1;

  int width, height;
  file >> width >> height;

  float scale;
  file >> scale;
  // 消耗换行符
  file.get();

  bool littleEndian = scale < 0.0f;
  EVAL_STEREO_CHECK(littleEndian, "Big endian PFM not supported here.");

  scale = std::abs(scale);

  int                nPixels = width * height * channels;
  std::vector<float> data(nPixels);
  file.read(reinterpret_cast<char *>(data.data()), nPixels * sizeof(float));

  // PFM存储是“从左下角开始”，我们需要flipY
  cv::Mat image(height, width, CV_32FC1, data.data());
  cv::Mat image_flipped;
  cv::flip(image, image_flipped, 0); // 上下翻转

  // 复制出来，否则data释放后Mat内容就无效
  return image_flipped.clone();
}

using DatasetPathType  = std::tuple<std::string, std::string, std::string>;
using DatasetFrameType = std::tuple<cv::Mat, cv::Mat, cv::Mat>;

std::vector<DatasetPathType> read_dataset_pathes(const std::string &sceneflow_val_txt_path)
{
  std::ifstream infile(sceneflow_val_txt_path);
  EVAL_STEREO_CHECK(infile, "Failed to open file : %s", sceneflow_val_txt_path.c_str());

  std::vector<DatasetPathType> pathes;

  std::string left_img_path, right_img_path, gt_disp_path;
  while (infile >> left_img_path >> right_img_path >> gt_disp_path)
  {
    fs::path val_dataset_dir = fs::path(sceneflow_val_txt_path).parent_path();

    pathes.push_back({val_dataset_dir / left_img_path, val_dataset_dir / right_img_path,
                      val_dataset_dir / gt_disp_path});
  }

  return pathes;
}

DatasetFrameType read_dataset_frame(const std::string &left_img_path,
                                    const std::string &right_img_path,
                                    const std::string &disp_gt_path)
{
  cv::Mat3b left  = cv::imread(left_img_path, cv::IMREAD_COLOR);
  cv::Mat3b right = cv::imread(right_img_path, cv::IMREAD_COLOR);
  EVAL_STEREO_CHECK(!left.empty() && !right.empty(), "Failed to read image: %s, or %s",
                    left_img_path.c_str(), right_img_path.c_str());

  cv::Mat1f gt_disp = read_pfm(disp_gt_path);
  EVAL_STEREO_CHECK(!gt_disp.empty(), "Failed to read disp gt : %s", disp_gt_path.c_str());

  return {left, right, gt_disp};
}

void eval_accuracy_sceneflow_stereo_matching(const std::shared_ptr<BaseStereoMatchingModel> &model,
                                             const std::string &sceneflow_val_txt_path)
{
  auto frame_pathes = read_dataset_pathes(sceneflow_val_txt_path);

  model->InitPipeline();

  BlockQueue<std::pair<std::shared_future<cv::Mat>, cv::Mat>> bq(100);

  auto func_producer = [&]() {
    for (const auto &pathes : frame_pathes)
    {
      const auto &[left_path, right_path, disp_gt_path] = pathes;
      const auto [left_img, right_img, disp_gt] =
          read_dataset_frame(left_path, right_path, disp_gt_path);

      std::shared_future<cv::Mat> fut = model->ComputeDispAsync(left_img, right_img);
      EVAL_STEREO_CHECK(fut.valid(), "Failed to call compute disp async API!");

      bq.BlockPush(std::pair<std::shared_future<cv::Mat>, cv::Mat>{fut, disp_gt});
    }
    bq.SetNoMoreInput();
  };

  std::thread producer_thread(func_producer);

  size_t total_pixels = 0;
  double total_epe    = 0.0;
  size_t cur_idx      = 0;
  while (true)
  {
    auto package = bq.Take();
    if (!package.has_value())
    {
      break;
    }

    auto [fut, disp_gt] = package.value();
    auto pred_disp      = fut.get();
    EVAL_STEREO_CHECK(pred_disp.size() == disp_gt.size(), "Predicted/GT disp size mismatch!");

    cv::Mat1b mask = (disp_gt > 0) & (disp_gt < 192);
    cv::Mat1f mask_f;
    mask.convertTo(mask_f, CV_32F); // 这里，mask的0/255被转换为0.f/255.f，要提前变成0/1
    mask_f /= 255.0f;               // 现在mask_f是0/1 float

    cv::Mat1f abs_diff;
    cv::absdiff(pred_disp, disp_gt, abs_diff);
    double sum_epe         = cv::sum(abs_diff.mul(mask_f))[0];
    int    valid_pixel_num = cv::countNonZero(mask);

    if (valid_pixel_num == 0)
      continue;

    total_epe += sum_epe;
    total_pixels += valid_pixel_num;
    cur_idx++;
    progress_bar(cur_idx, frame_pathes.size());
  }

  double mean_epe = total_epe / total_pixels;
  std::cout << "\r\nFinished SceneFlow validation, EPE: " << mean_epe << std::endl;

  bq.Disable();
  producer_thread.join();
}

} // namespace easy_deploy
