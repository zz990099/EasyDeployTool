#pragma once

#include <memory>
#include <vector>

#include "deploy_core/common.hpp"

namespace easy_deploy {

class ITensor {
public:
  template <typename T>
  T *Cast()
  {
    return static_cast<T *>(RawPtr());
  }

  void Reset()
  {
    SetBufferLocation(DataLocation::HOST);
    SetShape(GetDefaultShape());
  }

  virtual const std::string &GetName() const noexcept = 0;

  virtual void *RawPtr() = 0;

  virtual void SetBufferLocation(DataLocation location) = 0;

  virtual void ToLocation(DataLocation location) = 0;

  virtual DataLocation GetBufferLocation() const noexcept = 0;

  virtual void ZeroCopy(ITensor *tensor) = 0;

  virtual void DeepCopy(ITensor *tensor) = 0;

  virtual const std::vector<size_t> &GetDefaultShape() const noexcept = 0;

  virtual const std::vector<size_t> &GetShape() const noexcept = 0;

  virtual void SetShape(const std::vector<size_t> &shape) = 0;

  virtual size_t GetBufferMaxByteSize() const noexcept = 0;

  virtual size_t GetTensorByteSize() const noexcept = 0;

  virtual size_t GetElementByteSize() const noexcept = 0;

  virtual ~ITensor() = default;
};

class BlobsTensor {
public:
  BlobsTensor(std::unordered_map<std::string, std::unique_ptr<ITensor>> &&tensor_map)
      : tensor_map_(std::move(tensor_map))
  {}

  BlobsTensor(const BlobsTensor &other)            = delete;
  BlobsTensor &operator=(const BlobsTensor &other) = delete;

  ITensor *GetTensor(const std::string &blob_name)
  {
    if (tensor_map_.find(blob_name) == tensor_map_.end())
    {
      throw std::runtime_error("[BlobsTensor] Tensor NOT found : " + blob_name);
    }
    return tensor_map_.at(blob_name).get();
  }

  size_t Size() const noexcept
  {
    return tensor_map_.size();
  }

  void Reset()
  {
    for (auto &p_name_tensor : tensor_map_)
    {
      p_name_tensor.second->Reset();
    }
  }

private:
  std::unordered_map<std::string, std::unique_ptr<ITensor>> tensor_map_;
};

} // namespace easy_deploy
