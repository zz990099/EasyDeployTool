#include "trt_core/trt_core.hpp"

// std
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

// thirdparty
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include "trt_blob_buffer.hpp"

namespace easy_deploy {

class TensorrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override
  {
    if (severity == Severity::kINFO)
      LOG_DEBUG("[Tensorrt] : %s", msg);
    else if (severity == Severity::kERROR)
      LOG_ERROR("[Tensorrt] : %s", msg);
    else if (severity == Severity::kWARNING)
      LOG_WARN("[Tensorrt] : %s", msg);
  }
};

static const std::unordered_map<nvinfer1::DataType, size_t> map_tensor_type_byte_size_{
    {nvinfer1::DataType::kFLOAT, 4},
    {nvinfer1::DataType::kINT32, 4},
#if NV_TENSORRT_MAJOR >= 10
    {nvinfer1::DataType::kINT64, 8},
#endif
};

/**
 * @brief `TrtInferCore` is derived from `BaseInferCore` and override the abstract methods
 * of `BaseInferCore`. It wraps tensorrt engine loading and inference process.
 *
 */
class TrtInferCore : public BaseInferCore {
public:
  /**
   * @brief Construct `TrtInferCore` by providing tensorrt engine file path and blob buffer
   * pool size (defualt=5). This constructor does not need a map of blob_name and blob_shape,
   * while it will resolve model information by it self.
   *
   * @warning This constructor only should be used if the blobs shape of input model is fixed.
   * If you parse a model with dynamic blob shape, a exception will be thrown.
   *
   * @param engine_path Tensorrt engine file path.
   * @param mem_buf_size Size of buffer pool.
   */
  TrtInferCore(const std::string engine_path, const int mem_buf_size = 5);

  /**
   * @brief Construct `TrtInferCore` by providing tensorrt engine file path, max(default) blobs
   * shape and blob buffer pool size (defualt=5). If your model is parsed from a onnx model with
   * dynamic blob shape (e.g. blob_dim=-1), a mapping of blob_name and blob_shape should be provided
   * to help `TrtInferCore` alloc a apposite size blob buffer.
   *
   * @param engine_path Tensorrt engine file path.
   * @param blobs_shape Mapping of blob_name and blob_shape.
   * @param mem_buf_size Size of buffer pool.
   */
  TrtInferCore(const std::string                                             engine_path,
               const std::unordered_map<std::string, std::vector<uint64_t>> &blobs_shape,
               const int                                                     mem_buf_size = 5);

  /**
   * @brief Overrided from `BaseInferCore`, construct a instance of `TrtBlobBuffer` and return
   * the shared ptr of it. It is used by mem buffer pool in `BaseInferCore`, or users who wants
   * to alloc a brand new buffer.
   *
   * @return std::unique_ptr<BlobsTensor>
   */
  std::unique_ptr<BlobsTensor> AllocBlobsBuffer() override;

  /**
   * @brief Overrided from `BaseInferCore`. The `PreProcess` stage of tensorrt inference. It
   * prepares device buffers if user writes into host buffer derectly.
   *
   * @param buffer a common "pipeline" shared ptr.
   * @return true
   * @return false
   */
  bool PreProcess(std::shared_ptr<IPipelinePackage> buffer) override;

  /**
   * @brief Overrided from `BaseInferCore`. The `Inference` stage of tensorrt inference.
   *
   * @param buffer a common "pipeline" shared ptr.
   * @return true
   * @return false
   */
  bool Inference(std::shared_ptr<IPipelinePackage> buffer) override;

  /**
   * @brief Overrided from `BaseInferCore`. The `PostProcess` stage of tensorrt inference.
   * It will prepare output host buffer if user needs the output of model be accessable on host.
   *
   * @param buffer a common "pipeline" shared ptr.
   * @return true
   * @return false
   */
  bool PostProcess(std::shared_ptr<IPipelinePackage> buffer) override;

  ~TrtInferCore() override;

private:
  /**
   * @brief Load the tensorrt engine file on `engine_path`.
   *
   * @param engine_path
   */
  void LoadEngine(const std::string &engine_path);

  /**
   * @brief Automatically resolve model information.
   *
   * @param blobs_shape
   */
  void ResolveModelInformation(std::unordered_map<std::string, std::vector<uint64_t>> &blobs_shape);

private:
  // some members related to tensorrt
  static TensorrtLogger logger_;

  std::unique_ptr<nvinfer1::IRuntime>    runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};

  /**
   * @brief Due to tensorrt needs a unique inference context in every thread, we should maintain a
   * mapping of thread_id and a ptr of tensorrt context. A context will be created when there is a
   * new thread calls `Inference`. These contexts will be released when this `TrtInferCore` instance
   * is released.
   */
  std::unordered_map<std::thread::id, std::shared_ptr<nvinfer1::IExecutionContext>>
             s_map_tid2context_;
  std::mutex s_context_lck_;

  // cuda streams used in three stage.
  cudaStream_t preproces_stream_, inference_stream_, postprocess_stream_;

  // some model information mapping
  std::unordered_map<std::string, std::vector<size_t>> map_blob_name2shape_;
};

TensorrtLogger TrtInferCore::logger_{};

TrtInferCore::TrtInferCore(std::string engine_path, const int mem_buf_size)
{
  LoadEngine(engine_path);
  ResolveModelInformation(map_blob_name2shape_);

  BaseInferCore::Init(mem_buf_size);

  cudaStreamCreate(&preproces_stream_);
  cudaStreamCreate(&inference_stream_);
  cudaStreamCreate(&postprocess_stream_);
}

TrtInferCore::TrtInferCore(
    const std::string                                             engine_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &blobs_shape,
    const int                                                     mem_buf_size)
{
  LoadEngine(engine_path);
  map_blob_name2shape_ = blobs_shape;
  ResolveModelInformation(map_blob_name2shape_);

  BaseInferCore::Init(mem_buf_size);

  cudaStreamCreate(&preproces_stream_);
  cudaStreamCreate(&inference_stream_);
  cudaStreamCreate(&postprocess_stream_);
}

TrtInferCore::~TrtInferCore()
{
  BaseInferCore::Release();
}

void TrtInferCore::LoadEngine(const std::string &engine_path)
{
  initLibNvInferPlugins(nullptr, "");

  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good())
  {
    throw std::runtime_error("[TrtInferCore] Failed to read engine file!!!");
  }

  std::vector<char> data;

  file.seekg(0, file.end);
  const auto size = file.tellg();
  file.seekg(0, file.beg);

  data.resize(size);
  file.read(data.data(), size);

  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));

  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (engine_ == nullptr)
  {
    throw std::runtime_error("[TrtInferCore] Failed to create trt engine!!!");
  }
  LOG_DEBUG("[TrtInferCore] created tensorrt engine and context ! ");
}

void TrtInferCore::ResolveModelInformation(
    std::unordered_map<std::string, std::vector<uint64_t>> &blobs_shape)
{
  const int blob_number = engine_->getNbIOTensors();
  LOG_DEBUG("[TrtInferCore] model has {%d} blobs", blob_number);

  bool resolve_blob_shape = blobs_shape.empty();

  for (int i = 0; i < blob_number; ++i)
  {
    const char    *blob_name = engine_->getIOTensorName(i);
    nvinfer1::Dims dim       = engine_->getTensorShape(blob_name);

    const std::string s_blob_name(blob_name);

    if (resolve_blob_shape)
    {
      blobs_shape[s_blob_name] = std::vector<uint64_t>();
      for (int j = 0; j < dim.nbDims; ++j)
      {
        // 检查是否包含动态shape，自动解析暂不支持动态shape
        if (dim.d[j] <= 0)
        {
          throw std::runtime_error("[TrtInferCore] unsupport blob dim:" + std::to_string(dim.d[j]) +
                                   ", use explicit blob shape consturctor instead");
        }
        blobs_shape[s_blob_name].push_back(dim.d[j]);
      }

      std::string s_dim;
      for (auto d : dim.d)
      {
        s_dim += std::to_string(d) + " ";
      }
      LOG_DEBUG("[TrtInferCore] blob name : %s, dims : %s", blob_name, s_dim.c_str());
    }
  }
}

std::unique_ptr<BlobsTensor> TrtInferCore::AllocBlobsBuffer()
{
  std::unordered_map<std::string, std::unique_ptr<ITensor>> tensor_map;

  const int blob_number = engine_->getNbIOTensors();

  for (int i = 0; i < blob_number; ++i)
  {
    auto tensor = std::make_unique<TrtTensor>();

    const std::string s_blob_name      = engine_->getIOTensorName(i);
    const auto       &blob_shape       = map_blob_name2shape_[s_blob_name];
    auto              tensor_data_type = engine_->getTensorDataType(s_blob_name.c_str());
    CHECK_STATE_THROW(
        map_tensor_type_byte_size_.find(tensor_data_type) != map_tensor_type_byte_size_.end(),
        "[trt_core] Got unknown tensor data type: %d", static_cast<int32_t>(tensor_data_type));
    size_t blob_byte_size = map_tensor_type_byte_size_.at(tensor_data_type) * CumVector(blob_shape);

    tensor->current_shape_         = blob_shape;
    tensor->default_shape_         = blob_shape;
    tensor->byte_size_per_element_ = map_tensor_type_byte_size_.at(tensor_data_type);

    // alloc buffer memory
    // on device
    cudaMalloc(&tensor->buffer_on_device_, blob_byte_size);
    cudaMemset(tensor->buffer_on_device_, 0, blob_byte_size);
    tensor->self_maintain_buffer_device_ =
        std::unique_ptr<void, CudaMemoryDeleter<void>>(tensor->buffer_on_device_);
    // on host
    tensor->self_maintain_buffer_host_ = std::make_unique<u_char[]>(blob_byte_size);
    tensor->buffer_on_host_            = tensor->self_maintain_buffer_host_.get();

    tensor_map.emplace(s_blob_name, std::move(tensor));
  }

  return std::make_unique<BlobsTensor>(std::move(tensor_map));
}

bool TrtInferCore::PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit)
{
  CHECK_STATE(pipeline_unit != nullptr, "[TrtInferCore] PreProcess got invalid pipeline_unit!");
  auto blobs_tensor = pipeline_unit->GetInferBuffer();
  CHECK_STATE(blobs_tensor != nullptr, "[TrtInferCore] PreProcess got invalid blobs_tensor!");

  const int blob_number = engine_->getNbIOTensors();

  for (int i = 0; i < blob_number; ++i)
  {
    const std::string s_blob_name = engine_->getIOTensorName(i);

    auto tensor = dynamic_cast<TrtTensor *>(blobs_tensor->GetTensor(s_blob_name));
    CHECK_STATE(tensor != nullptr, "[trt_core] PreProcess got invalid tensor : %s",
                s_blob_name.c_str());

    if (tensor->current_location_ == DataLocation::HOST &&
        engine_->getTensorIOMode(s_blob_name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
    {
      cudaMemcpyAsync(tensor->buffer_on_device_, tensor->buffer_on_host_,
                      tensor->GetTensorByteSize(), cudaMemcpyHostToDevice, preproces_stream_);
    }
  }
  cudaStreamSynchronize(preproces_stream_);

  return true;
}

bool TrtInferCore::Inference(std::shared_ptr<IPipelinePackage> pipeline_unit)
{
  // Create tensorrt context if this is the first time execution of this thread.
  std::thread::id cur_thread_id = std::this_thread::get_id();
  if (s_map_tid2context_.find(cur_thread_id) == s_map_tid2context_.end())
  {
    std::shared_ptr<nvinfer1::IExecutionContext> context{engine_->createExecutionContext()};
    {
      std::unique_lock<std::mutex> u_lck(s_context_lck_);
      s_map_tid2context_.insert({cur_thread_id, context});
    }
  }
  auto context = s_map_tid2context_[cur_thread_id];

  // Get buffer ptr
  CHECK_STATE(pipeline_unit != nullptr, "[TrtInferCore] Inference got invalid pipeline_unit!");
  auto blobs_tensor = pipeline_unit->GetInferBuffer();
  CHECK_STATE(blobs_tensor != nullptr, "[TrtInferCore] Inference got invalid blobs_tensor!");

  const int blob_number = engine_->getNbIOTensors();

  for (int i = 0; i < blob_number; ++i)
  {
    const std::string s_blob_name = engine_->getIOTensorName(i);

    auto tensor = dynamic_cast<TrtTensor *>(blobs_tensor->GetTensor(s_blob_name));
    CHECK_STATE(tensor != nullptr, "[trt_core] Inference got invalid tensor : %s",
                s_blob_name.c_str());

    context->setTensorAddress(s_blob_name.c_str(), tensor->buffer_on_device_);

    if (engine_->getTensorIOMode(s_blob_name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
    {
      const auto &tensor_shape = tensor->current_shape_;

      nvinfer1::Dims dynamic_dim;
      dynamic_dim.nbDims = tensor_shape.size();
      for (size_t i = 0; i < tensor_shape.size(); ++i)
      {
        dynamic_dim.d[i] = tensor_shape[i];
      }
      CHECK_STATE(context->setInputShape(s_blob_name.c_str(), dynamic_dim),
                  "[TrtInferCore] Inference execute `context->setInputShape` failed!!!");
    }
  }
  context->enqueueV3(inference_stream_);

  cudaStreamSynchronize(inference_stream_);
  return true;
}

bool TrtInferCore::PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit)
{
  CHECK_STATE(pipeline_unit != nullptr, "[TrtInferCore] PostProcess got invalid pipeline_unit!");
  auto blobs_tensor = pipeline_unit->GetInferBuffer();
  CHECK_STATE(blobs_tensor != nullptr, "[TrtInferCore] PostProcess got invalid blobs_tensor!");

  const int blob_number = engine_->getNbIOTensors();

  for (int i = 0; i < blob_number; ++i)
  {
    const std::string s_blob_name = engine_->getIOTensorName(i);

    auto tensor = dynamic_cast<TrtTensor *>(blobs_tensor->GetTensor(s_blob_name));
    CHECK_STATE(tensor != nullptr, "[trt_core] PostProcess got invalid tensor : %s",
                s_blob_name.c_str());

    if (engine_->getTensorIOMode(s_blob_name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
      continue;

    if (tensor->current_location_ == DataLocation::HOST)
    {
      cudaMemcpyAsync(tensor->buffer_on_host_, tensor->buffer_on_device_,
                      tensor->GetTensorByteSize(), cudaMemcpyDeviceToHost, postprocess_stream_);
    }
  }

  cudaStreamSynchronize(postprocess_stream_);
  return true;
}

static bool FileSuffixCheck(const std::string &file_path, const std::string &suffix)
{
  const size_t mark = file_path.rfind('.');
  std::string  suf;
  return mark != file_path.npos &&
         (suf = file_path.substr(mark, file_path.size() - mark)) == suffix;
}

std::shared_ptr<BaseInferCore> CreateTrtInferCore(std::string model_path, const int mem_buf_size)
{
  if (!FileSuffixCheck(model_path, ".engine"))
  {
    throw std::invalid_argument("Trt infer core expects file end with `.engine`. But got " +
                                model_path + " instead");
  }

  return std::make_shared<TrtInferCore>(model_path, mem_buf_size);
}

std::shared_ptr<BaseInferCore> CreateTrtInferCore(
    std::string                                                   model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                     mem_buf_size)
{
  if (!FileSuffixCheck(model_path, ".engine"))
  {
    throw std::invalid_argument("Trt infer core expects file end with `.engine`. But got " +
                                model_path + " instead");
  }

  std::unordered_map<std::string, std::vector<uint64_t>> blobs_shape;
  for (const auto &p : input_blobs_shape)
  {
    blobs_shape.insert(p);
  }
  for (const auto &p : output_blobs_shape)
  {
    blobs_shape.insert(p);
  }

  return std::make_shared<TrtInferCore>(model_path, blobs_shape, mem_buf_size);
}

} // namespace easy_deploy
