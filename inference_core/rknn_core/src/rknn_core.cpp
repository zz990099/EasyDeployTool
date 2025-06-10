#include "rknn_core/rknn_core.hpp"
#include "rknn_blob_buffer.hpp"
#include <unordered_map>

#include <rknn_api.h>

namespace easy_deploy {

static std::unordered_map<RknnInputTensorType, rknn_tensor_type> map_type_my2rk{
    {RknnInputTensorType::RK_UINT8, RKNN_TENSOR_UINT8},
    {RknnInputTensorType::RK_INT8, RKNN_TENSOR_INT8},
    {RknnInputTensorType::RK_FLOAT16, RKNN_TENSOR_FLOAT16},
    {RknnInputTensorType::RK_FLOAT32, RKNN_TENSOR_FLOAT32},
    {RknnInputTensorType::RK_UINT32, RKNN_TENSOR_UINT32},
    {RknnInputTensorType::RK_INT32, RKNN_TENSOR_INT32},
    {RknnInputTensorType::RK_INT64, RKNN_TENSOR_INT64},
};

static std::unordered_map<rknn_tensor_type, int> map_rknn_type2size_{
    {RKNN_TENSOR_INT8, 1},    {RKNN_TENSOR_UINT8, 1}, {RKNN_TENSOR_FLOAT16, 4},
    {RKNN_TENSOR_FLOAT32, 4}, {RKNN_TENSOR_INT32, 4}, {RKNN_TENSOR_UINT32, 4},
    {RKNN_TENSOR_INT64, 8}};

static std::unordered_map<rknn_tensor_type, rknn_tensor_type> map_rknn_type2type{
    {RKNN_TENSOR_INT8, RKNN_TENSOR_UINT8},      {RKNN_TENSOR_UINT8, RKNN_TENSOR_UINT8},
    {RKNN_TENSOR_FLOAT16, RKNN_TENSOR_FLOAT32}, {RKNN_TENSOR_FLOAT32, RKNN_TENSOR_FLOAT32},
    {RKNN_TENSOR_INT32, RKNN_TENSOR_INT32},     {RKNN_TENSOR_UINT32, RKNN_TENSOR_UINT32},
    {RKNN_TENSOR_INT64, RKNN_TENSOR_INT64}};

class RknnInferCore : public BaseInferCore {
public:
  RknnInferCore(std::string                                                 model_path,
                const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
                const int                                                   mem_buf_size     = 5,
                const int                                                   parallel_ctx_num = 1);

  ~RknnInferCore() override;

  InferCoreType GetType()
  {
    return InferCoreType::RKNN;
  }

  std::string GetName()
  {
    return "rknn_core";
  }

private:
  bool PreProcess(std::shared_ptr<IPipelinePackage> buffer) override;

  bool Inference(std::shared_ptr<IPipelinePackage> buffer) override;

  bool PostProcess(std::shared_ptr<IPipelinePackage> buffer) override;

private:
  std::unique_ptr<BlobsTensor> AllocBlobsBuffer() override;

  size_t ReadModelFromFile(const std::string &model_path, void **model_data);

  void ResolveModelInformation(
      const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type);

private:
  std::vector<rknn_context>     rknn_ctx_parallel_;
  BlockQueue<rknn_context>      bq_ctx_;
  BlockQueue<std::future<bool>> bq_async_future_;

  //
  size_t blob_input_number_;
  size_t blob_output_number_;

  std::vector<rknn_tensor_attr> blob_attr_input_;
  std::vector<rknn_tensor_attr> blob_attr_output_;

  std::unordered_map<std::string, std::vector<uint64_t>> map_input_blob_name2shape_;
  std::unordered_map<std::string, std::vector<uint64_t>> map_output_blob_name2shape_;
};

RknnInferCore::RknnInferCore(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
    const int                                                   mem_buf_size,
    const int                                                   parallel_ctx_num)
    : bq_ctx_(parallel_ctx_num), bq_async_future_(parallel_ctx_num)
{
  if (parallel_ctx_num <= 0)
  {
    throw std::invalid_argument("[rknn core] Got Invalid ctx_num: " +
                                std::to_string(parallel_ctx_num));
  }

  void  *model_data           = nullptr;
  size_t model_data_byte_size = ReadModelFromFile(model_path, &model_data);
  if (model_data == nullptr)
  {
    throw std::runtime_error("[rknn_core] Failed to read model from file: " + model_path);
  }
  LOG_DEBUG("[rknn core] initilize using {%d} ctx instances", parallel_ctx_num);

  for (int i = 0; i < parallel_ctx_num; ++i)
  {
    rknn_context ctx;
    if (rknn_init(&ctx, model_data, model_data_byte_size, 0, NULL) != RKNN_SUCC)
    {
      throw std::runtime_error("[rknn_core] Failed to init rknn_ctx [ " + std::to_string(i) + " ]");
    }
    bq_ctx_.BlockPush(ctx);
    rknn_ctx_parallel_.push_back(ctx);
  }

  free(model_data);

  ResolveModelInformation(map_blob_type);

  BaseInferCore::Init(mem_buf_size);
}

size_t RknnInferCore::ReadModelFromFile(const std::string &model_path, void **model_data)
{
  FILE *fp = fopen(model_path.c_str(), "rb");
  if (fp == NULL)
  {
    printf("fopen %s fail!\n", model_path.c_str());
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  size_t file_size = ftell(fp);
  char  *data      = (char *)malloc(file_size + 1);
  data[file_size]  = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != fread(data, 1, file_size, fp))
  {
    printf("fread %s fail!\n", model_path.c_str());
    free(data);
    fclose(fp);
    return -1;
  }
  if (fp)
  {
    fclose(fp);
  }
  *model_data = data;
  return file_size;
}

RknnInferCore::~RknnInferCore()
{
  //////////////////////////// IMPORTANT /////////////////////////////////
  for (size_t i = 0; i < bq_ctx_.GetMaxSize(); ++i)
  {
    auto ctx_value = bq_ctx_.Take();
    if (!ctx_value.has_value())
    {
      LOG_ERROR("[rknn_core] Failed to get ctx from block queue!!!");
      continue;
    }
    auto ctx = ctx_value.value();
    if (rknn_destroy(ctx) != RKNN_SUCC)
    {
      LOG_ERROR("[rknn_core] In deconstructor destroy rknn ctx failed!!!");
    }
  }
}

std::unique_ptr<BlobsTensor> RknnInferCore::AllocBlobsBuffer()
{
  std::unordered_map<std::string, std::unique_ptr<ITensor>> tensor_map;

  for (size_t i = 0; i < blob_input_number_; ++i)
  {
    const auto &s_blob_name        = blob_attr_input_[i].name;
    const auto  rknn_blob_type     = blob_attr_input_[i].type;
    const auto &blob_shape         = map_input_blob_name2shape_[s_blob_name];
    auto        tensor             = std::make_unique<RknnTensor>();
    tensor->name_                  = s_blob_name;
    tensor->current_shape_         = blob_shape;
    tensor->default_shape_         = blob_shape;
    tensor->byte_size_per_element_ = map_rknn_type2size_.at(map_rknn_type2type.at(rknn_blob_type));
    tensor->self_maintain_buffer_host_ = std::make_unique<u_char[]>(tensor->GetTensorByteSize());
    tensor->buffer_on_host_            = tensor->self_maintain_buffer_host_.get();

    tensor_map.emplace(s_blob_name, std::move(tensor));
  }

  for (size_t i = 0; i < blob_output_number_; ++i)
  {
    const auto &s_blob_name    = blob_attr_output_[i].name;
    const auto  rknn_blob_type = blob_attr_output_[i].type;
    const auto &blob_shape     = map_output_blob_name2shape_[s_blob_name];

    auto tensor                        = std::make_unique<RknnTensor>();
    tensor->name_                      = s_blob_name;
    tensor->current_shape_             = blob_shape;
    tensor->default_shape_             = blob_shape;
    tensor->byte_size_per_element_     = 4; // map_rknn_type2size_.at(rknn_blob_type);
    tensor->self_maintain_buffer_host_ = std::make_unique<u_char[]>(tensor->GetTensorByteSize());
    tensor->buffer_on_host_            = tensor->self_maintain_buffer_host_.get();

    tensor_map.emplace(s_blob_name, std::move(tensor));
  }

  return std::make_unique<BlobsTensor>(std::move(tensor_map));
}

void RknnInferCore::ResolveModelInformation(
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type)
{
  rknn_input_output_num rknn_io_num;
  if (rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_IN_OUT_NUM, &rknn_io_num, sizeof(rknn_io_num)) !=
      RKNN_SUCC)
  {
    throw std::runtime_error("[rknn core] Failed to execute in_out_num `rknn_query`");
  }
  LOG_DEBUG("model input blob num: %ld \toutput blob num: %ld", rknn_io_num.n_input,
            rknn_io_num.n_output);

  blob_input_number_  = rknn_io_num.n_input;
  blob_output_number_ = rknn_io_num.n_output;
  blob_attr_input_.resize(blob_input_number_);
  blob_attr_output_.resize(blob_output_number_);

  // input blob
  for (size_t i = 0; i < blob_input_number_; ++i)
  {
    blob_attr_input_[i].index = i;
    if (rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_INPUT_ATTR, &(blob_attr_input_[i]),
                   sizeof(rknn_tensor_attr)) != RKNN_SUCC)
    {
      throw std::runtime_error("[rknn core] Failed to execute input `rknn_query`");
    }
    const std::string s_blob_name = blob_attr_input_[i].name;

    // modify type using user offered tensor type (it's a limitation of rknn)
    if (map_blob_type.find(s_blob_name) != map_blob_type.end())
    {
      blob_attr_input_[i].type = map_type_my2rk[map_blob_type.at(s_blob_name)];
    }

    if (map_rknn_type2size_.find(blob_attr_input_[i].type) == map_rknn_type2size_.end())
    {
      LOG_ERROR("[rknn_core] blob_name : %s, blob_type: %d, NOT found in `map_rknn_type2size_`!",
                s_blob_name.c_str(), blob_attr_input_[i].type);
      throw std::runtime_error("[rknn core] Failed to resolve model information!!!");
    }
    const int blob_type_byte_size = map_rknn_type2size_[blob_attr_input_[i].type];

    std::vector<uint64_t> blob_shape;
    std::string           s_blob_info = s_blob_name;
    for (size_t j = 0; j < blob_attr_input_[i].n_dims; ++j)
    {
      s_blob_info += "\t" + std::to_string(blob_attr_input_[i].dims[j]);
      blob_shape.push_back(blob_attr_input_[i].dims[j]);
    }
    LOG_DEBUG(s_blob_info.c_str());
    LOG_DEBUG("blob fmt: %s, type: %s", get_format_string(blob_attr_input_[i].fmt),
              get_type_string(blob_attr_input_[i].type));
    map_input_blob_name2shape_[s_blob_name] = blob_shape;
  }

  // output blob
  for (size_t i = 0; i < blob_output_number_; ++i)
  {
    blob_attr_output_[i].index = i;
    if (rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_OUTPUT_ATTR, &(blob_attr_output_[i]),
                   sizeof(rknn_tensor_attr)) != RKNN_SUCC)
    {
      throw std::runtime_error("[rknn core] Failed to execute output `rknn_query`");
    }
    const std::string     s_blob_name = blob_attr_output_[i].name;
    std::vector<uint64_t> blob_shape;
    std::string           s_blob_info = blob_attr_output_[i].name;
    for (size_t j = 0; j < blob_attr_output_[i].n_dims; ++j)
    {
      s_blob_info += "\t" + std::to_string(blob_attr_output_[i].dims[j]);
      blob_shape.push_back(blob_attr_output_[i].dims[j]);
    }
    LOG_DEBUG(s_blob_info.c_str());
    LOG_DEBUG("blob fmt: %s, type: %s", get_format_string(blob_attr_output_[i].fmt),
              get_type_string(blob_attr_output_[i].type));
    map_output_blob_name2shape_[s_blob_name] = blob_shape;
  }
}

bool RknnInferCore::PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit)
{
  CHECK_STATE(pipeline_unit != nullptr, "[rknn_core] Inference got invalid pipeline_unit!");
  auto blobs_tensor = pipeline_unit->GetInferBuffer();
  CHECK_STATE(blobs_tensor != nullptr, "[rknn_core] Inference got invalid blobs_tensor!");

  auto func_async_execution = [this, blobs_tensor](rknn_context ctx) -> bool {
    std::vector<rknn_input> inputs(blob_input_number_);
    for (size_t i = 0; i < blob_input_number_; ++i)
    {
      auto tensor     = blobs_tensor->GetTensor(blob_attr_input_[i].name);
      inputs[i].index = blob_attr_input_[i].index;
      inputs[i].fmt   = blob_attr_input_[i].fmt;
      inputs[i].type  = map_rknn_type2type.at(blob_attr_input_[i].type);
      inputs[i].buf   = tensor->RawPtr();
      inputs[i].size  = tensor->GetTensorByteSize();
    }
    std::vector<rknn_output> outputs(blob_output_number_);
    for (size_t i = 0; i < blob_output_number_; ++i)
    {
      auto tensor            = blobs_tensor->GetTensor(blob_attr_output_[i].name);
      outputs[i].index       = blob_attr_output_[i].index;
      outputs[i].buf         = tensor->RawPtr();
      outputs[i].size        = tensor->GetTensorByteSize();
      outputs[i].is_prealloc = true;
      outputs[i].want_float  = true;
    }

    CHECK_STATE(rknn_inputs_set(ctx, blob_input_number_, inputs.data()) == RKNN_SUCC,
                "[rknn core] Inference `rknn_inputs_set` execute failed!!!");
    CHECK_STATE(rknn_run(ctx, nullptr) == RKNN_SUCC,
                "[rknn core] Inference `rknn_run` execute failed!!!");
    CHECK_STATE(rknn_outputs_get(ctx, blob_output_number_, outputs.data(), nullptr) == RKNN_SUCC,
                "[rknn core] Inference `rknn_outputs_get` execute failed!!!");

    bq_ctx_.BlockPush(ctx);
    return true;
  };

  auto ctx = bq_ctx_.Take();
  CHECK_STATE(ctx.has_value(), "[rknn_core] Failed to get valid ctx !!!");

  bq_async_future_.BlockPush(std::async(std::launch::async, func_async_execution, ctx.value()));

  return true;
}

#define RKNN_CHECK_STATE(state, hint) \
  {                                   \
    if (!(state))                     \
    {                                 \
      LOG_ERROR((hint));              \
      bq_ctx_.BlockPush(index);       \
      return false;                   \
    }                                 \
  }

bool RknnInferCore::Inference(std::shared_ptr<IPipelinePackage> pipeline_unit)
{
  auto future = bq_async_future_.Take();
  CHECK_STATE(future.has_value(), "[rknn_core] Failed to valid future !!!");

  CHECK_STATE(future.value().get(), "[rknn_core] Failed execute rknn inference !!!");
  return true;
}

bool RknnInferCore::PostProcess(std::shared_ptr<IPipelinePackage> buffer)
{
  return true;
}

std::shared_ptr<BaseInferCore> CreateRknnInferCore(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
    const int                                                   mem_buf_size,
    const int                                                   parallel_ctx_num)
{
  return std::make_shared<RknnInferCore>(model_path, map_blob_type, mem_buf_size, parallel_ctx_num);
}

} // namespace easy_deploy
