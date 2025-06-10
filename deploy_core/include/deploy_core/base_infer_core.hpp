#pragma once

#include <memory>
#include <thread>
#include <vector>
#include <unordered_set>

#include "common_utils/block_queue.hpp"
#include "deploy_core/async_pipeline.hpp"

namespace easy_deploy {

enum InferCoreType { ONNXRUNTIME, TENSORRT, RKNN, NOT_PROVIDED };

/**
 * @brief `IRotInferCore` is abstract interface class which defines all pure virtual functions
 * that the derived class should implement, e.g., `PreProcess`, `Inference` and `PostProcess`.
 *
 */
class IRotInferCore {
public:
  /**
   * @brief `AllocBlobsBuffer` is a common interface that user could get a brand new buffer
   * instance by. This pure virtual function is implemented by actual inference core, which
   * may take a while to process. Use pre-allocated buffer instance in mem buffer pool could
   * get better performance. See `BaseInferCore`.
   *
   * @return std::unique_ptr<BlobsTensor> A brand new buffer instance allocated by inference
   * core.
   */
  virtual std::unique_ptr<BlobsTensor> AllocBlobsBuffer() = 0;

  /**
   * @brief Get the core type.
   *
   * @return InferCoreType
   */
  virtual InferCoreType GetType()
  {
    return InferCoreType::NOT_PROVIDED;
  }

  /**
   * @brief Return the name of inference core.
   *
   * @return std::string
   */
  virtual std::string GetName()
  {
    return "";
  }

protected:
  virtual ~IRotInferCore() = default;

  /**
   * @brief `PreProcess` stage of the inference process. Return true if this is stage is not
   * needed in the actual inference core implementation. Return false if something went wrong
   * while doing processing. The pipeline will drop the package if `PreProcess` returns false.
   *
   * @param buffer a common "pipeline" package ptr.
   * @return true
   * @return false
   */
  virtual bool PreProcess(std::shared_ptr<IPipelinePackage> buffer) = 0;

  /**
   * @brief `Inference` stage of the inference process. Return false if something went wrong
   * while doing processing. The pipeline will drop the package if `Inference` returns false.
   *
   * @param buffer a common "pipeline" package ptr.
   * @return true
   * @return false
   */
  virtual bool Inference(std::shared_ptr<IPipelinePackage> buffer) = 0;

  /**
   * @brief `PostProcess` stage of the inference process. Return false if something went wrong
   * while doing processing. The pipeline will drop the package if `PostProcess` returns false.
   *
   * @param buffer a common "pipeline" package ptr.
   * @return true
   * @return false
   */
  virtual bool PostProcess(std::shared_ptr<IPipelinePackage> buffer) = 0;
};

/**
 * @brief A simple implementation of mem buffer pool. Using `BlockQueue` to deploy a
 * producer- consumer model. It will allocate buffer using `AllocBlobsBuffer` method of
 * `IRotInferCore` and provides `BlobsTensor` ptr when `Alloc` method is called. The "Alloced"
 * buffer will return back to mem buffer pool while the customed deconstruction method of shared_ptr
 * ptr is called.
 *
 */
class MemBufferPool {
public:
  MemBufferPool(IRotInferCore *infer_core, const size_t pool_size)
      : pool_size_(pool_size), dynamic_pool_(pool_size)
  {
    for (size_t i = 0; i < pool_size; ++i)
    {
      auto blobs_tensor = infer_core->AllocBlobsBuffer();
      dynamic_pool_.BlockPush(blobs_tensor.get());
      static_pool_.emplace(std::move(blobs_tensor));
    }
  }

  std::shared_ptr<BlobsTensor> Alloc(bool block)
  {
    // customed deconstruction method
    auto func_dealloc = [&](BlobsTensor *buf) {
      buf->Reset();
      this->dynamic_pool_.BlockPush(buf);
    };

    auto buf = block ? dynamic_pool_.Take() : dynamic_pool_.TryTake();
    return buf.has_value() ? std::shared_ptr<BlobsTensor>(buf.value(), func_dealloc) : nullptr;
  }

  void Release()
  {
    if (dynamic_pool_.Size() != pool_size_)
    {
      LOG_ERROR("[MemBufPool] does not maintain all bufs when release func called!");
    }
    static_pool_.clear();
  }

  int RemainSize()
  {
    return dynamic_pool_.Size();
  }

  ~MemBufferPool()
  {
    Release();
  }

private:
  const size_t                                     pool_size_;
  BlockQueue<BlobsTensor *>                        dynamic_pool_;
  std::unordered_set<std::unique_ptr<BlobsTensor>> static_pool_;
};

/**
 * @brief A dummy class to help `BaseInferCore` inherit from `BaseAsyncPipeline` to generate
 * async pipeline framework.
 *
 */
class _DummyInferCoreGenReulstType {
public:
  bool operator()(const std::shared_ptr<IPipelinePackage> & /*package*/)
  {
    return true;
  }
};

/**
 * @brief `BaseInferCore` inherits `IRotInferCore` and `BaseAsyncPipeline`. `IRotInferCore`
 * defines all pure virtual methods of the abstract function of the inference core.
 * `BaseAsyncPipeline` provides a set of methods to help user build and utilize a async
 * inference pipeline. See `BaseAsyncPipeline` defination.
 *
 * @note The inheritance relationship between class A and class B is modified by protected.
 * And `BaseInferCore` only makes the `GetPipelineContext` method public, which means the
 * derived class of `BaseInferCore` is not supported to deploy async pipeline inference
 * process. It should be used by specific algorithms in its entirety.
 *
 */
class BaseInferCore : public IRotInferCore,
                      protected BaseAsyncPipeline<bool, _DummyInferCoreGenReulstType> {
protected:
  BaseInferCore();
  typedef std::shared_ptr<IPipelinePackage> ParsingType;

public:
  using BaseAsyncPipeline::GetPipelineContext;

  /**
   * @brief This function provides a sync inference process which is completely independent
   * of the async inference pipeline. Through, it depends on the three stage virtual methods
   * defined in `IRotInferCore`. Return false if something went wrong while inference.
   *
   * @param buffer
   * @param batch_size default=1, multi-batch inference may not be supported.
   * @return true
   * @return false
   */
  bool SyncInfer(BlobsTensor *tensors, const int batch_size = 1);

  /**
   * @brief Get the pre-allocated blobs buffer shared pointer. The returned pointer is a
   * smart pointer which will automatically return to the pool when it is released.
   *
   * @param block whether to block the thread if the pool is empty.
   * @return std::shared_ptr<BlobsTensor>
   */
  std::shared_ptr<BlobsTensor> GetBuffer(bool block);

  /**
   * @brief Release the sources in base class.
   *
   * @warning The derived class should call `BaseInferCore::Release()` in its deconstruct
   * function in order to release the blobs buffer before the enviroment is destroyed.
   * Things go wrong if allocated memory released after their enviroment released on some
   * hardware.
   *
   */
  virtual void Release();

protected:
  virtual ~BaseInferCore();

  /**
   * @brief Init the base class memory pool.
   *
   * @warning Please call `Init()` at the derived class construct function`s end when the
   * runtime enviroment is setup successfully. This method will call `AllocBlobsBuffer`
   * to create a memory pool. Temporary we manually call this method to init the memory pool.
   *
   * @param mem_buf_size number of blobs buffers pre-allocated.
   */
  void Init(size_t mem_buf_size = 5);

private:
  std::unique_ptr<MemBufferPool> mem_buf_pool_{nullptr};
};

/**
 * @brief Abstract factory class of infer_core.
 *
 */
class BaseInferCoreFactory {
public:
  virtual std::shared_ptr<BaseInferCore> Create() = 0;
};

} // namespace easy_deploy
