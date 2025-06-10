#pragma once

#include "deploy_core/base_detection.hpp"
#include "my_fixture.hpp"

namespace easy_deploy {

void eval_accuracy_coco_detection_2d(const std::shared_ptr<BaseDetectionModel> &model,
                                     const std::string                         &coco_val_dir_path,
                                     const std::string &coco_annotations_path);

class EvalAccuracyDetection2DFixture {
public:
  using SetUpReturnType = std::tuple<std::shared_ptr<BaseDetectionModel>, std::string, std::string>;

  virtual ~EvalAccuracyDetection2DFixture() = default;
  // 子类重写SetUp即可
  virtual SetUpReturnType SetUp() = 0;
};

#define RegisterEvalAccuracyDetection2D(FIXTURE_CLASS)                                    \
  static struct BOOST_JOIN(FixtureReg_, __LINE__) {                                       \
    BOOST_JOIN(FixtureReg_, __LINE__)()                                                   \
    {                                                                                     \
      FixtureRegistry::instance().add(#FIXTURE_CLASS, []() {                              \
        std::unique_ptr<EvalAccuracyDetection2DFixture> fixture(new FIXTURE_CLASS);       \
        auto [model, coco_val_dir_path, coco_annotations_path] = fixture->SetUp();        \
        eval_accuracy_coco_detection_2d(model, coco_val_dir_path, coco_annotations_path); \
      });                                                                                 \
    }                                                                                     \
  } BOOST_JOIN(fixture_reg_obj_, __LINE__);
// BOOST_JOIN宏兼容性
#ifndef BOOST_JOIN
#define BOOST_JOIN(X, Y) BOOST_DO_JOIN(X, Y)
#define BOOST_DO_JOIN(X, Y) BOOST_DO_JOIN2(X, Y)
#define BOOST_DO_JOIN2(X, Y) X##Y
#endif

} // namespace easy_deploy
