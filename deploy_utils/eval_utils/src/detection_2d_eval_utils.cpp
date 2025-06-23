#include "eval_utils/detection_2d_eval_utils.hpp"

#include "common_utils/fs_utils.hpp"
#include "common_utils/progress_bar.hpp"
#include "common_utils/image_drawer.hpp"

#include <nlohmann/json.hpp>
#include <fstream>

namespace easy_deploy {

static void WriteResultToJson(const std::string &file_path, const std::vector<BBox2D> &results)
{
  nlohmann::json j;
  for (auto &bbox : results)
  {
    nlohmann::json item;
    item["x1"]    = bbox.x - bbox.w / 2;
    item["y1"]    = bbox.y - bbox.h / 2;
    item["x2"]    = bbox.x + bbox.w / 2;
    item["y2"]    = bbox.y + bbox.h / 2;
    item["conf"]  = bbox.conf;
    item["label"] = bbox.cls;
    j.push_back(item);
  }
  std::ofstream ofs(file_path);
  ofs << j.dump(4);
  ofs.close();
}

static void generate_coco_result(const std::shared_ptr<BaseDetectionModel> &model,
                                 const std::string                         &coco_val_dir_path,
                                 const std::string                         &save_result_tmp_path)
{
  static const std::unordered_set<std::string> valid_ext{".jpg", ".png", ".jpeg", ".bmp"};

  fs::remove_all(save_result_tmp_path);
  fs::create_directory(save_result_tmp_path);

  auto                  all_files = get_files_in_directory(coco_val_dir_path);
  std::vector<fs::path> valid_files;
  for (const auto &file : all_files)
  {
    const auto file_ext = file.extension().string();
    if (valid_ext.count(file_ext) > 0)
    {
      valid_files.push_back(file);
    }
  }

  std::cout << "Start inference on COCO2017-VAL dataset, dataset dir : " << coco_val_dir_path
            << "\r\nTotal images : " << valid_files.size() << std::endl;

  BlockQueue<std::tuple<cv::Mat, fs::path>> bq(100);

  auto func_read_images = [&]() {
    for (const auto &file : valid_files)
    {
      std::string img_file = file.string();
      std::string img_name = file.filename().stem().string();
      cv::Mat     img      = cv::imread(img_file);
      if (img.empty())
      {
        std::cerr << "Failed to open image : " << img_file << std::endl;
        continue;
      }
      bq.BlockPush(std::tuple<cv::Mat, fs::path>{img, file});
    }
    bq.SetNoMoreInput();
  };

  auto read_image_thread = std::thread(func_read_images);

  size_t cur_idx = 0;
  while (true) {
    auto package = bq.Take();
    if (!package.has_value()) {
      break;
    }

    auto [image, file] = package.value();

    std::vector<BBox2D> det_results;
    bool ok = model->Detect(image, det_results, 0.001, false);
    if (!ok)
    {
      std::cerr << "Failed to inference on image : " << file.string() << std::endl;
      continue;
    }

    std::string out_path = fs::path(save_result_tmp_path) / (file.filename().stem().string() + ".json");
    WriteResultToJson(out_path, det_results);

    progress_bar(++cur_idx, valid_files.size());
  }
  std::cout << "\r\n";

  bq.Disable();
  read_image_thread.join();
}

static void eval_detection_result_with_python(const std::string &save_result_tmp_path,
                                              const std::string &coco_annotations_path)
{
  const std::string eval_script_cmd =
      "python3 /workspace/build/scripts/eval_utils/eval_coco2017_val.py --gt_json=" +
      coco_annotations_path + " --det_folder=" + save_result_tmp_path;

  auto ret = std::system(eval_script_cmd.c_str());
  if (ret == 0)
  {
    std::cout << "Eval detection_2d algorithm on coco2017-val successfully!" << std::endl;
  } else
  {
    std::cout << "Failed to eval detection_2d algorithm, feel free to open an issue for this!"
              << std::endl;
  }
}

void eval_accuracy_coco_detection_2d(const std::shared_ptr<BaseDetectionModel> &model,
                                     const std::string                         &coco_val_dir_path,
                                     const std::string &coco_annotations_path)
{
  const std::string save_result_tmp_paht = "/tmp/detection_coco_val_result/";
  generate_coco_result(model, coco_val_dir_path, save_result_tmp_paht);

  eval_detection_result_with_python(save_result_tmp_paht, coco_annotations_path);
}

} // namespace easy_deploy
