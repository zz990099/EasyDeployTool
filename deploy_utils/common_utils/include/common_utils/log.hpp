#pragma once

#include <cstdio>
#include <cstdarg>
#include <mutex>

namespace easy_deploy {

class ILogger {
public:
  // 使用格式串+可变参数模板
  template <typename... Args>
  void log_debug(const char *fmt, Args &&...args) noexcept
  {
    do_log_debug(fmt, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void log_info(const char *fmt, Args &&...args) noexcept
  {
    do_log_info(fmt, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void log_warn(const char *fmt, Args &&...args) noexcept
  {
    do_log_warn(fmt, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void log_error(const char *fmt, Args &&...args) noexcept
  {
    do_log_error(fmt, std::forward<Args>(args)...);
  }

  virtual ~ILogger() = default;

protected:
  // 留给派生类具体实现（纯虚，参数包转发）
  virtual void do_log_debug(const char *fmt, ...) noexcept = 0;
  virtual void do_log_info(const char *fmt, ...) noexcept  = 0;
  virtual void do_log_warn(const char *fmt, ...) noexcept  = 0;
  virtual void do_log_error(const char *fmt, ...) noexcept = 0;
};

class GlobalLogger {
public:
  static GlobalLogger &instance()
  {
    static GlobalLogger inst;
    return inst;
  }

  ILogger *GetLogger() noexcept;

  void SetLogger(ILogger *logger) noexcept;

private:
  GlobalLogger();

  ~GlobalLogger();

  ILogger *logger_;
};

#define REGISTER_EasyDeploy_LOGGER(LoggerDerivedType)                             \
  namespace {                                                                     \
  struct LoggerDerivedType##_Registrar {                                          \
    LoggerDerivedType##_Registrar()                                               \
    {                                                                             \
      ::easy_deploy::GlobalLogger::instance().SetLogger(new LoggerDerivedType()); \
    }                                                                             \
  };                                                                              \
  static LoggerDerivedType##_Registrar g_##LoggerDerivedType##_registrar;         \
  }

inline void FormatMsg(char *buf, size_t buflen, const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, buflen, fmt, args);
  va_end(args);
}

#define EasyDeployLog(level, fmt, ...)                                  \
  do                                                                    \
  {                                                                     \
    auto *logger = ::easy_deploy::GlobalLogger::instance().GetLogger(); \
    if (logger)                                                         \
    {                                                                   \
      logger->log_##level(fmt, ##__VA_ARGS__);                          \
    }                                                                   \
  } while (0)

#ifdef ENABLE_DEBUG_OUTPUT
#define LOG_DEBUG(fmt, ...) EasyDeployLog(debug, fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) \
  do                        \
  {                         \
  } while (0)
#endif

#define LOG_INFO(fmt, ...) EasyDeployLog(info, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) EasyDeployLog(warn, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) EasyDeployLog(error, fmt, ##__VA_ARGS__)

} // namespace easy_deploy
