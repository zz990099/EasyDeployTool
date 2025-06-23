#pragma once

#include <string>
#include <functional>
#include <iostream>

namespace easy_deploy {

struct FixtureCase {
  std::string           name;
  std::function<void()> func;
};

struct FixtureRegistry {
  static FixtureRegistry &instance()
  {
    static FixtureRegistry inst;
    return inst;
  }
  void add(const std::string &name, std::function<void()> func)
  {
    testcases_.emplace_back(FixtureCase{name, func});
  }
  int run(const std::string &filter = "")
  {
    int failed = 0;
    for (auto &c : testcases_)
    {
      if (!filter.empty() && c.name.find(filter) == std::string::npos)
        continue;
      std::cout << "[ RUN ] " << c.name << std::endl;
      try
      {
        c.func();
        std::cout << "[ OK ] " << c.name << std::endl;
      } catch (const std::exception &e)
      {
        std::cout << "[FAILED] " << c.name << " : " << e.what() << std::endl;
        failed++;
      }
    }
    return failed;
  }

private:
  std::vector<FixtureCase> testcases_;
};

#define EVAL_MAIN()                                 \
  int main(int argc, char **argv)                   \
  {                                                 \
    std::string filter;                             \
    for (int i = 1; i < argc; ++i)                  \
    {                                               \
      std::string arg(argv[i]);                     \
      if (arg.find("--filter=") == 0)               \
        filter = arg.substr(strlen("--filter="));   \
    }                                               \
    return FixtureRegistry::instance().run(filter); \
  }

} // namespace easy_deploy
