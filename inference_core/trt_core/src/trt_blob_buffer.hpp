#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <cstring>

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

template <typename T>
class CudaMemoryDeleter {
public:
  void operator()(T *ptr)
  {
    cudaFree(ptr);
  }
};

class TrtTensor : public ITensor {
public:
  const std::string &GetName() const noexcept override
  {
    return name_;
  }

  void *RawPtr() override
  {
    assert(current_location_ != DataLocation::UNKOWN);
    return current_location_ == DataLocation::HOST ? buffer_on_host_ : buffer_on_device_;
  }

  void SetBufferLocation(DataLocation location) override
  {
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[TrtTensor] `SetBufferLocation` Got invalid location: UNKOWN !");
    current_location_ = location;
  }

  void ToLocation(DataLocation location) override
  {
    assert(current_location_ != DataLocation::UNKOWN);
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[TrtTensor] `ToLocation` Got invalid location: UNKOWN !");

    if (current_location_ == location)
      return;
    if (current_location_ == DataLocation::HOST)
    {
      cudaMemcpy(buffer_on_device_, buffer_on_host_, GetTensorByteSize(), cudaMemcpyHostToDevice);
    } else
    {
      cudaMemcpy(buffer_on_host_, buffer_on_device_, GetTensorByteSize(), cudaMemcpyDeviceToHost);
    }
  }

  DataLocation GetBufferLocation() const noexcept override
  {
    assert(current_location_ != DataLocation::UNKOWN);
    return current_location_;
  }

  void ZeroCopy(ITensor *tensor) override
  {
    CHECK_STATE_THROW(tensor != nullptr, "[TrtTensor] `ZeroCopy` Got invalid tensor: nullptr !");
    auto location = tensor->GetBufferLocation();
    auto raw_ptr  = tensor->RawPtr();
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[TrtTensor] `ZeroCopy` Got invalid tensor location: UNKOWN !");
    CHECK_STATE_THROW(raw_ptr != nullptr,
                      "[TrtTensor] `ZeroCopy` Got invalid tensor raw_ptr: nullptr !");

    if (location == DataLocation::DEVICE)
    {
      cudaPointerAttributes attr;
      cudaError_t           status = cudaPointerGetAttributes(&attr, raw_ptr);
      CHECK_STATE_THROW(
          status == cudaSuccess && attr.type == cudaMemoryType::cudaMemoryTypeDevice,
          "[TrtTensor] `ZeroCopy` Got invalid tensor raw_ptr: NOT valid cuda buffer on device !");
      buffer_on_device_ = raw_ptr;
    } else
    {
      buffer_on_host_ = raw_ptr;
    }

    current_location_ = location;
  }

  void DeepCopy(ITensor *tensor) override
  {
    CHECK_STATE_THROW(tensor != nullptr, "[TrtTensor] `DeepCopy` Got invalid tensor: nullptr !");
    auto location = tensor->GetBufferLocation();
    auto raw_ptr  = tensor->RawPtr();
    CHECK_STATE_THROW(location != DataLocation::UNKOWN,
                      "[TrtTensor] `DeepCopy` Got invalid tensor location: UNKOWN !");
    CHECK_STATE_THROW(raw_ptr != nullptr,
                      "[TrtTensor] `DeepCopy` Got invalid tensor raw_ptr: nullptr !");

    if (location == DataLocation::DEVICE)
    {
      cudaPointerAttributes attr;
      cudaError_t           status = cudaPointerGetAttributes(&attr, raw_ptr);
      CHECK_STATE_THROW(
          status == cudaSuccess && attr.type == cudaMemoryType::cudaMemoryTypeDevice,
          "[TrtTensor] `ZeroCopy` Got invalid tensor raw_ptr: NOT valid cuda buffer on device !");
      buffer_on_device_ = self_maintain_buffer_device_.get();
      cudaMemcpy(buffer_on_device_, raw_ptr, GetTensorByteSize(), cudaMemcpyDeviceToDevice);
    } else
    {
      buffer_on_host_ = self_maintain_buffer_host_.get();
      memcpy(buffer_on_host_, raw_ptr, GetTensorByteSize());
    }
    current_location_ = location;
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
                      "[TrtTensor] `SetShape` Got invalid shape: exceeds max byte size !");
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
  void               *buffer_on_device_{nullptr};
  void               *buffer_on_host_{nullptr};
  DataLocation        current_location_{DataLocation::HOST};
  std::vector<size_t> current_shape_;
  std::vector<size_t> default_shape_;
  size_t              byte_size_per_element_;

  std::unique_ptr<void, std::function<void(void *)>> self_maintain_buffer_device_{nullptr};
  std::unique_ptr<u_char[]>                          self_maintain_buffer_host_{nullptr};
};

} // namespace easy_deploy
