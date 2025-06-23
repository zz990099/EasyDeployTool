#pragma once

#include "deploy_core/base_stereo.hpp"
#include "my_fixture.hpp"

namespace easy_deploy {

void eval_accuracy_sceneflow_stereo_matching(const std::shared_ptr<BaseStereoMatchingModel> &model,
                                             const std::string &sceneflow_val_txt_path);

class EvalAccuracyStereoMatchingFixture {
public:
  using SetUpReturnType = std::tuple<std::shared_ptr<BaseStereoMatchingModel>, std::string>;

  virtual ~EvalAccuracyStereoMatchingFixture() = default;
  // 子类重写SetUp即可
  virtual SetUpReturnType SetUp() = 0;
};

#define RegisterEvalAccuracyStereoMatching(FIXTURE_CLASS)                              \
  static struct BOOST_JOIN(FixtureReg_, __LINE__) {                                    \
    BOOST_JOIN(FixtureReg_, __LINE__)()                                                \
    {                                                                                  \
      FixtureRegistry::instance().add(#FIXTURE_CLASS, []() {                           \
        std::unique_ptr<EvalAccuracyStereoMatchingFixture> fixture(new FIXTURE_CLASS); \
        auto [model, sceneflow_val_txt_path] = fixture->SetUp();                       \
        eval_accuracy_sceneflow_stereo_matching(model, sceneflow_val_txt_path);        \
      });                                                                              \
    }                                                                                  \
  } BOOST_JOIN(fixture_reg_obj_, __LINE__);
// BOOST_JOIN宏兼容性
#ifndef BOOST_JOIN
#define BOOST_JOIN(X, Y) BOOST_DO_JOIN(X, Y)
#define BOOST_DO_JOIN(X, Y) BOOST_DO_JOIN2(X, Y)
#define BOOST_DO_JOIN2(X, Y) X##Y
#endif

} // namespace easy_deploy
