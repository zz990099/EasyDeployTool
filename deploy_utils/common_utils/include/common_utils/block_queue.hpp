#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace easy_deploy {

/**
 * @brief A thread-safe blocking queue with shutdown/disable semantics.
 */
template <typename T>
class BlockQueue {
public:
  explicit BlockQueue(size_t max_size) : max_size_(max_size)
  {}

  // 禁拷贝、禁赋值
  BlockQueue(const BlockQueue &)            = delete;
  BlockQueue &operator=(const BlockQueue &) = delete;

  /**
   * @brief Push a obj into the queue. Will block the thread if the queue is full.
   * Return false if push is disabled.
   */
  template <typename U>
  bool BlockPush(U &&obj) noexcept;

  /**
   * @brief Push a obj into the queue. If full, remove oldest and insert.
   * Return false if push is disabled.
   */
  template <typename U>
  bool CoverPush(U &&obj) noexcept;

  /**
   * @brief Remove and return the front element. Will block if empty and not disabled/no more input.
   * Return std::nullopt if take is disabled, or no more input and queue is empty.
   */
  std::optional<T> Take() noexcept;

  /**
   * @brief Remove and return front element if any; else return std::nullopt.
   */
  std::optional<T> TryTake() noexcept;

  /**
   * @brief Return current queue size.
   */
  size_t Size() noexcept;

  /**
   * @brief Return if queue is empty.
   */
  bool Empty() noexcept;

  /**
   * @brief Disable both push and take. Wake up all threads.
   */
  void Disable() noexcept;

  /**
   * @brief Clear all elements and disable both push/take.
   */
  void DisableAndClear() noexcept;

  /**
   * @brief Disable push only. Wake up producers.
   */
  void DisablePush() noexcept;

  /**
   * @brief Re-enable push. Wake up waiting producers.
   */
  void EnablePush() noexcept;

  /**
   * @brief Disable take only. Wake up consumers.
   */
  void DisableTake() noexcept;

  /**
   * @brief Re-enable take. Wake up waiting consumers.
   */
  void EnableTake() noexcept;

  /**
   * @brief Set "NoMoreInput", i.e. producers不会再推送, 通知所有消费者。
   */
  void SetNoMoreInput() noexcept;

  /**
   * @brief Get max size.
   */
  size_t GetMaxSize() const noexcept
  {
    return max_size_;
  }

  ~BlockQueue() noexcept
  {
    Disable();
  }

private:
  size_t                  max_size_;
  std::queue<T>           q_;
  bool                    push_enabled_{true};
  bool                    take_enabled_{true};
  bool                    no_more_input_{false};
  std::mutex              mtx_;
  std::condition_variable cv_producer_;
  std::condition_variable cv_consumer_;
};

// ========== Implementation ==========

template <typename T>
template <typename U>
bool BlockQueue<T>::BlockPush(U &&obj) noexcept
{
  std::unique_lock<std::mutex> lk(mtx_);
  cv_producer_.wait(lk, [this] { return q_.size() < max_size_ || !push_enabled_; });
  if (!push_enabled_)
    return false;
  q_.push(std::forward<U>(obj));
  cv_consumer_.notify_one();
  return true;
}

template <typename T>
template <typename U>
bool BlockQueue<T>::CoverPush(U &&obj) noexcept
{
  std::unique_lock<std::mutex> lk(mtx_);
  if (!push_enabled_)
    return false;
  if (q_.size() == max_size_)
    q_.pop();
  q_.push(std::forward<U>(obj));
  cv_consumer_.notify_one();
  return true;
}

template <typename T>
std::optional<T> BlockQueue<T>::Take() noexcept
{
  std::unique_lock<std::mutex> lk(mtx_);
  cv_consumer_.wait(lk, [this] { return !q_.empty() || !take_enabled_ || no_more_input_; });
  if (!take_enabled_ || (no_more_input_ && q_.empty()))
    return std::nullopt;
  T obj = std::move(q_.front());
  q_.pop();
  cv_producer_.notify_one();
  return obj;
}

template <typename T>
std::optional<T> BlockQueue<T>::TryTake() noexcept
{
  std::unique_lock<std::mutex> lk(mtx_);
  if (q_.empty())
    return std::nullopt;
  T obj = std::move(q_.front());
  q_.pop();
  cv_producer_.notify_one();
  return obj;
}

template <typename T>
size_t BlockQueue<T>::Size() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  return q_.size();
}

template <typename T>
bool BlockQueue<T>::Empty() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  return q_.empty();
}

template <typename T>
void BlockQueue<T>::Disable() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  push_enabled_  = false;
  take_enabled_  = false;
  no_more_input_ = true;
  cv_producer_.notify_all();
  cv_consumer_.notify_all();
}

template <typename T>
void BlockQueue<T>::DisableAndClear() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  push_enabled_  = false;
  take_enabled_  = false;
  no_more_input_ = true;
  while (!q_.empty()) q_.pop();
  cv_producer_.notify_all();
  cv_consumer_.notify_all();
}

template <typename T>
void BlockQueue<T>::DisablePush() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  push_enabled_ = false;
  cv_producer_.notify_all();
  cv_consumer_.notify_all();
}

template <typename T>
void BlockQueue<T>::EnablePush() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  push_enabled_ = true;
  cv_producer_.notify_all();
}

template <typename T>
void BlockQueue<T>::DisableTake() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  take_enabled_ = false;
  cv_producer_.notify_all();
  cv_consumer_.notify_all();
}

template <typename T>
void BlockQueue<T>::EnableTake() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  take_enabled_ = true;
  cv_consumer_.notify_all();
}

template <typename T>
void BlockQueue<T>::SetNoMoreInput() noexcept
{
  std::lock_guard<std::mutex> lk(mtx_);
  no_more_input_ = true;
  cv_consumer_.notify_all();
}

} // namespace easy_deploy
