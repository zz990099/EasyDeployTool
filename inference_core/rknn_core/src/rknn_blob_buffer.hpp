#pragma once

#include <assert.h>
#include <string.h>
#include <string>
#include <rknn_api.h>
#include "deploy_core/blob_buffer.hpp"

namespace easy_deploy {

template <typename Type>
inline Type CumVector(const std::vector<Type> &vec)
{
  Type ret = 1;
  for (const auto &nn : vec)
  {
    ret *= nn;
  }

  return ret;
}

template <typename Type>
inline std::string VisualVec(const std::vector<Type> &vec)
{
  std::string ret;
  for (const auto &v : vec)
  {
    ret += std::to_string(v) + " ";
  }
  return ret;
}

class RknnTensor : public ITensor {
public:
  const std::string &GetName() const noexcept override
  {
    return name_;
  }

  void *RawPtr() override
  {
    assert(current_location_ != DataLocation::UNKOWN);
    return buffer_on_host_;
  }

  void SetBufferLocation(DataLocation location) override
  {
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[RknnTensor] `SetBufferLocation` Got invalid location: UNKOWN !");
  }

  void ToLocation(DataLocation location) override
  {
    assert(current_location_ != DataLocation::UNKOWN);
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[RknnTensor] `ToLocation` Got invalid location: UNKOWN !");
  }

  DataLocation GetBufferLocation() const noexcept override
  {
    assert(current_location_ != DataLocation::UNKOWN);
    return current_location_;
  }

  void ZeroCopy(ITensor *tensor) override
  {
    CHECK_STATE_THROW(tensor != nullptr, "[RknnTensor] `ZeroCopy` Got invalid tensor: nullptr !");
    auto location = tensor->GetBufferLocation();
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[RknnTensor] `ZeroCopy` Got invalid tensor location: UNKOWN !");
    if (location == DataLocation::DEVICE)
    {
      tensor->ToLocation(DataLocation::HOST);
    }
    auto raw_ptr = tensor->RawPtr();
    CHECK_STATE_THROW(raw_ptr != nullptr,
                      "[RknnTensor] `ZeroCopy` Got invalid tensor raw_ptr: nullptr !");

    buffer_on_host_ = raw_ptr;
  }

  void DeepCopy(ITensor *tensor) override
  {
    CHECK_STATE_THROW(tensor != nullptr, "[RknnTensor] `DeepCopy` Got invalid tensor: nullptr !");
    auto location = tensor->GetBufferLocation();
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[RknnTensor] `DeepCopy` Got invalid tensor location: UNKOWN !");
    if (location == DataLocation::DEVICE)
    {
      tensor->ToLocation(DataLocation::HOST);
    }
    auto raw_ptr = tensor->RawPtr();
    CHECK_STATE_THROW(raw_ptr != nullptr,
                      "[RknnTensor] `DeepCopy` Got invalid tensor raw_ptr: nullptr !");

    buffer_on_host_ = self_maintain_buffer_host_.get();
    memcpy(buffer_on_host_, raw_ptr, GetTensorByteSize());
  }

  const std::vector<size_t> &GetDefaultShape() const noexcept override
  {
    return default_shape_;
  }

  const std::vector<size_t> &GetShape() const noexcept override
  {
    return current_shape_;
  }

  void SetShape(const std::vector<size_t> &shape) override
  {
    size_t this_shape_byte_size = byte_size_per_element_ * CumVector(shape);
    CHECK_STATE_THROW(this_shape_byte_size <= GetBufferMaxByteSize(),
                      "[RknnTensor] `SetShape` Got invalid shape: exceeds max byte size !");
    current_shape_ = shape;
  }

  size_t GetBufferMaxByteSize() const noexcept override
  {
    return byte_size_per_element_ * CumVector(default_shape_);
  }

  size_t GetTensorByteSize() const noexcept override
  {
    return byte_size_per_element_ * CumVector(current_shape_);
  }

  size_t GetElementByteSize() const noexcept override
  {
    return byte_size_per_element_;
  }

public:
  std::string         name_;
  void               *buffer_on_host_{nullptr};
  const DataLocation  current_location_{DataLocation::HOST};
  std::vector<size_t> current_shape_;
  std::vector<size_t> default_shape_;
  size_t              byte_size_per_element_;

  std::unique_ptr<u_char[]> self_maintain_buffer_host_{nullptr};
};

} // namespace easy_deploy
