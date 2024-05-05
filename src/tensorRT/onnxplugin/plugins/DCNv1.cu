

#include <onnxplugin/onnxplugin.hpp>
#include <common/cuda_tools.hpp>
#include <cublas_v2.h>
#include <cuda_fp16.h>

using namespace ONNXPlugin;

const int MAXTENSORDIMS = 10;

struct TensorDesc {
    int shape[MAXTENSORDIMS];
    int stride[MAXTENSORDIMS];
    int dim;
};


#define cublasCheck(op)														 \
do {																	 \
    auto ret = (op);													 \
    if (ret != CUBLAS_STATUS_SUCCESS) {											 \
        INFOF("%s fail, %d != %d", #op, ret, CUBLAS_STATUS_SUCCESS);				 \
    }																	 \
} while (0);

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }
size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
    return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}

template <typename scalar_t>
cublasStatus_t cublasGemmWrap(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const scalar_t* alpha,
    const scalar_t* A, int lda, const scalar_t* B, int ldb,
    const scalar_t* beta, scalar_t* C, int ldc);

template <>
cublasStatus_t cublasGemmWrap<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B,
    int ldb, const float* beta, float* C, int ldc) {
    //printf("goning to cublasGemmWrap\n");
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemmWrap<half>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const half* alpha, const half* A, int lda, const half* B,
    int ldb, const half* beta, half* C, int ldc) {
    //printf("goning to cublasGemmWrap\n");
    return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


struct ConvParams {
    int stride[2];
    int padding[2];
    int dilation[2];
    int groups;
    int deformable_groups;
    int im2col_step;
};

void parseJson(const char* jsonString, struct ConvParams* params) {

    const char* strideStart = strstr(jsonString, "\"stride\": [") + strlen("\"stride\": [");
    sscanf(strideStart, "%d, %d", &params->stride[0], &params->stride[1]);

    // 定位到 "padding" 字段并解析其值
    const char* paddingStart = strstr(jsonString, "\"padding\": [") + strlen("\"padding\": [");
    sscanf(paddingStart, "%d, %d", &params->padding[0], &params->padding[1]);

    // 定位到 "dilation" 字段并解析其值
    const char* dilationStart = strstr(jsonString, "\"dilation\": [") + strlen("\"dilation\": [");
    sscanf(dilationStart, "%d, %d", &params->dilation[0], &params->dilation[1]);

    // 定位到 "groups" 字段并解析其值
    const char* groupsStart = strstr(jsonString, "\"groups\": ") + strlen("\"groups\": ");
    sscanf(groupsStart, "%d", &params->groups);

    // 定位到 "deformable_groups" 字段并解析其值
    const char* defGroupsStart = strstr(jsonString, "\"deformable_groups\": ") + strlen("\"deformable_groups\": ");
    sscanf(defGroupsStart, "%d", &params->deformable_groups);

    // 定位到 "im2col_step" 字段并解析其值
    const char* im2colStart = strstr(jsonString, "\"im2col_step\": ") + strlen("\"im2col_step\": ");
    sscanf(im2colStart, "%d", &params->im2col_step);
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n, const scalar_t* __restrict__ data_im, const scalar_t* __restrict__ data_offset,
    const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col, const int width_col,
    scalar_t* __restrict__ data_col) {
    const int hw_col = height_col * width_col;
    const int data_col_step = batch_size * hw_col;
    
    CUDA_1D_KERNEL_LOOP(index, n) {
        // index index of output matrix
        int tmp_index = index;
        const int w_col = tmp_index % width_col;
        tmp_index /= width_col;
        const int h_col = tmp_index % height_col;
        tmp_index /= height_col;
        const int b_col = tmp_index % batch_size;
        const int c_im = tmp_index / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;

        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        scalar_t* __restrict__ data_col_ptr = data_col + c_col * data_col_step + index % data_col_step;
        const scalar_t* __restrict__ data_im_ptr =
            data_im + (b_col * num_channels + c_im) * height * width;
        const scalar_t* __restrict__ data_offset_ptr =
            data_offset +
            ((b_col * deformable_group + deformable_group_index) << 1) * kernel_h * kernel_w * hw_col +
            h_col * width_col + w_col;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                const int data_offset_h = (i * kernel_w + j) * hw_col << 1;
                const scalar_t offset_h = data_offset_ptr[data_offset_h];
                const int data_offset_w = data_offset_h + hw_col;
                const scalar_t offset_w = data_offset_ptr[data_offset_w];
                const scalar_t h_im = h_in + i * dilation_h + (float)offset_h;
                const scalar_t w_im = w_in + j * dilation_w + (float)offset_w;
                const scalar_t val = deformable_im2col_bilinear(data_im_ptr, height, width, h_im, w_im);
                *data_col_ptr = val;
                data_col_ptr += data_col_step;
            }
        }
    }
}

template <typename scalar_t>
void deform_conv_im2col(const scalar_t* input, const scalar_t* offset, scalar_t* column,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, cudaStream_t stream) {
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col * parallel_imgs;
    int channel_per_deformable_group = channels / deformable_group;
    //printf("goning to deformable_im2col_gpu_kernel\n");
    int job = channels * height_col * width_col;
    //printf("job = %d\n", job);
    //printf("num_kernels = %d\n", num_kernels);
    deformable_im2col_gpu_kernel<scalar_t> << <CUDATools::grid_dims(num_kernels), CUDATools::block_dims(num_kernels / 32), 0, stream >> > (
        num_kernels, input, offset, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group, parallel_imgs, channels,
        deformable_group, height_col, width_col, column);
    //printf("Done deformable_im2col_gpu_kernel\n"); 

    cudaCheckError();
}

template <class scalar_t>
__global__ void copy_permute_kernel(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src,
    int n, TensorDesc ts_src_stride, TensorDesc ts_dst_stride,
    TensorDesc ts_permute) {
    const int src_dim = ts_src_stride.dim;
    const auto src_stride = ts_src_stride.stride;
    const auto dst_stride = ts_dst_stride.stride;
    const auto permute = ts_permute.shape;
    CUDA_1D_KERNEL_LOOP(index, n) {
        size_t dst_index = index;
        size_t src_index = 0;
        for (int i = 0; i < src_dim; ++i) {
            int dim_index = dst_index / dst_stride[i];
            dst_index = dst_index % dst_stride[i];
            src_index += dim_index * src_stride[permute[i]];
        }
        dst[index] = src[src_index];
    }
}

template <class scalar_t>
void memcpyPermute(scalar_t* dst, const scalar_t* src, int* src_size, int* permute, int src_dim,
    cudaStream_t stream) {
    size_t copy_size = 1;
    TensorDesc ts_permute;
    memcpy(&(ts_permute.shape[0]), permute, src_dim * sizeof(int));

    TensorDesc ts_src_stride;
    TensorDesc ts_dst_stride;
    ts_src_stride.dim = src_dim;
    ts_dst_stride.dim = src_dim;
    int* src_stride = &(ts_src_stride.stride[0]);
    int* dst_stride = &(ts_dst_stride.stride[0]);
    int* dst_size = &(ts_dst_stride.shape[0]);
    src_stride[src_dim - 1] = 1;
    dst_stride[src_dim - 1] = 1;

    for (int i = src_dim - 1; i >= 0; --i) {
        dst_size[i] = src_size[permute[i]];
        if (i < src_dim - 1) {
            src_stride[i] = src_stride[i + 1] * src_size[i + 1];
        }
    }

    for (int i = src_dim - 1; i >= 0; --i) {
        copy_size *= dst_size[i];
        if (i < src_dim - 1) {
            dst_stride[i] = dst_stride[i + 1] * dst_size[i + 1];
        }
    }

    copy_permute_kernel<scalar_t> << <CUDATools::grid_dims(copy_size), CUDATools::block_dims(copy_size), 0, stream >> > (
        dst, src, copy_size, ts_src_stride, ts_dst_stride, ts_permute);
}

template <typename scalar_t>
void deform_conv(const scalar_t* input, const scalar_t* weight, const scalar_t* offset,
    scalar_t* output, void* workspace, int batchSize, int nInputPlane, int inputHeight,
    int inputWidth, int nOutputPlane, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream) 
{
    size_t word_size = sizeof(float);
    im2col_step = std::min(int(batchSize), im2col_step);
    //printf("inputWidth = %d\n", inputWidth);
    //printf("inputHeight = %d\n", inputHeight);
    //printf("kW = %d\n", kW);
    //printf("kH = %d\n", kH);
    //printf("padW = %d\n", padW);
    //printf("padH = %d\n", padH);
    //printf("dW = %d\n", dW);
    //printf("dH = %d\n", dH);
    //printf("dilationW = %d\n", dilationW);
    //printf("dilationH = %d\n", dilationH);
    //printf("im2col_step = %d\n", im2col_step);
    //printf("deformable_group = %d\n", deformable_group);
    //printf("group = %d\n", group);

    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long outputHW = outputHeight * outputWidth;
    long kHW = kH * kW;
    long columns_size = getAlignedSize(nInputPlane * kHW * im2col_step * outputHW * word_size);


    // column buffer for img2col
    char* workspace_ptr = reinterpret_cast<char*>(workspace);
    scalar_t* columns = reinterpret_cast<scalar_t*>(workspace_ptr);
    workspace_ptr = workspace_ptr + columns_size;


    scalar_t* output_buffer;
    if (im2col_step == 1) {
        output_buffer = output;
    }
    else {
        // output need permute when im2col_step!=1
        output_buffer = reinterpret_cast<scalar_t*>(workspace_ptr);
    }
    //printf("goning to input_elt_step\n");
    long input_elt_step = im2col_step * nInputPlane * inputHeight * inputWidth;
    long offset_elt_step = im2col_step * deformable_group * 2 * kHW * outputHW;
    long out_buffer_step = nOutputPlane * im2col_step * outputHW;
    long col_g_step = nInputPlane * kHW * im2col_step * outputHW / group;
    long weight_g_step = nOutputPlane * nInputPlane * kHW / (group * group);
    long out_buffer_g_step = out_buffer_step / group;
    int m = nOutputPlane / group;
    int n = im2col_step * outputHW;
    int k = nInputPlane * kHW / group;
    scalar_t alpha = 1.f;
    scalar_t beta = 0.f;
    //printf("goning to elt < batchSize\n");
    for (int elt = 0; elt < batchSize / im2col_step; elt++) {
        const scalar_t* input_start = input + elt * input_elt_step;
        const scalar_t* offset_start = offset + elt * offset_elt_step;

        deform_conv_im2col<scalar_t>(input_start, offset_start, columns, nInputPlane, inputHeight,
            inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
            im2col_step, deformable_group, stream);
        //printf("goning to g < group\n");
        for (int g = 0; g < group; ++g) {
            const scalar_t* weight_start = weight + g * weight_g_step;
            scalar_t* col_start = columns + g * col_g_step;
            scalar_t* out_buffer_start = output_buffer + elt * out_buffer_step + g * out_buffer_g_step;
            cublasGemmWrap<scalar_t>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, col_start,
                n, weight_start, k, &beta, out_buffer_start, n);
            //printf("Done cublasGemmWrap\n");
            cudaCheckError();
        }
    }

    if (im2col_step != 1) {
        int output_buffer_shape[5] = { batchSize / im2col_step, nOutputPlane, im2col_step,
                                      static_cast<int>(outputHeight), static_cast<int>(outputWidth) };
        int output_buffer_permute[5] = { 0, 2, 1, 3, 4 };
        memcpyPermute<scalar_t>(output, output_buffer, &output_buffer_shape[0],
            &output_buffer_permute[0], 5, stream);
    }
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t deformable_im2col_bilinear(const scalar_t* __restrict__ input,
    const int height, const int width,
    float h, float w) {
    if (h <= -1 || height <= h || w <= -1 || width <= w) {
        return 0;
    }

    const int h_low = floorf(h);
    const int w_low = floorf(w);

    input += h_low * width;
    const scalar_t v1 = (h_low >= 0 && w_low >= 0) ? input[w_low] : static_cast<scalar_t>(0.0f);
    const int w_high = w_low + 1;
    const scalar_t v2 =
        (h_low >= 0 && w_high <= width - 1) ? input[w_high] : static_cast<scalar_t>(0.0f);
    const scalar_t lw = w - w_low;
    const scalar_t v_low = fmaf(v2 - v1, lw, v1);
    input += width;
    const scalar_t v3 =
        (h_low <= height - 2 && w_low >= 0) ? input[w_low] : static_cast<scalar_t>(0.0f);
    const scalar_t v4 =
        (h_low <= height - 2 && w_high <= width - 1) ? input[w_high] : static_cast<scalar_t>(0.0f);
    const scalar_t v_high = fmaf(v4 - v3, lw, v3);
    const scalar_t lh = h - h_low;
    const scalar_t val = fmaf(v_high - v_low, lh, v_low);
    return val;
}

template <>
__device__ __forceinline__ __half deformable_im2col_bilinear(const __half* __restrict__ input,
    const int height, const int width,
    float h, float w) {
    if (h <= -1 || height <= h || w <= -1 || width <= w) {
        return 0;
    }

    const int h_low = floorf(h);
    const int w_low = floorf(w);

    input += h_low * width;
    const float v1 = (h_low >= 0 && w_low >= 0) ? __half2float(input[w_low]) : 0.0f;
    const int w_high = w_low + 1;
    const float v2 = (h_low >= 0 && w_high <= width - 1) ? __half2float(input[w_high]) : 0.0f;
    const float lw = w - w_low;
    const float v_low = fmaf(v2 - v1, lw, v1);
    input += width;
    const float v3 = (h_low <= height - 2 && w_low >= 0) ? __half2float(input[w_low]) : 0.0f;
    const float v4 =
        (h_low <= height - 2 && w_high <= width - 1) ? __half2float(input[w_high]) : 0.0f;
    const float v_high = fmaf(v4 - v3, lw, v3);
    const float lh = h - h_low;
    const float val = fmaf(v_high - v_low, lh, v_low);
    return __float2half(val);
}

unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
        // case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
        return 1;
    default:
        throw std::runtime_error("Invalid DataType.");
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}





class DCNv1 : public TRTPlugin {

private:
    nvinfer1::Dims mStride;
	nvinfer1::Dims mPadding;
	nvinfer1::Dims mDilation;
	int mGroup;
	int mDeformableGroup;

public:
    cublasHandle_t cublasHandle_ = nullptr;
    SetupPlugin(DCNv1);

    virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* cublas, nvinfer1::IGpuAllocator* /*allocator*/) noexcept override{
        cublasHandle_ = cublas;
    }

    virtual void detachFromContext() noexcept override{
        cublasHandle_ = nullptr;
    }

    std::shared_ptr<LayerConfig> new_config() {
        auto cfg = TRTPlugin::new_config();

        //cfg->supportDataType_ = {nvinfer1::DataType::kFLOAT};
        //cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
        cfg->support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
        return cfg;
    }

    virtual void config_finish() override{
        
         INFO("weights = %d", config_->weights_.size());
         for(int i = 0; i < config_->weights_.size(); ++i){
         	auto& w = config_->weights_[i];
         	if(w->type() == TRT::DataType::Float16){
         		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), float(w->at<__half>(0)));
         	}else{
         		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), w->at<float>(0));
         	}
         }
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept{

        int sizeof_dtype = getElementSize(outputs[0].type);

        int batch_size = inputs[0].dims.d[0];
        int nInputPlane = inputs[0].dims.d[1];
        int inputHeight = inputs[0].dims.d[2];
        int inputWidth = inputs[0].dims.d[3];

        int nOutputPlane = outputs[0].dims.d[1];
        int outputHeight = outputs[0].dims.d[2];
        int outputWidth = outputs[0].dims.d[3];

        int offsetHeight = inputs[1].dims.d[2];
        int offsetWidth = inputs[1].dims.d[3];
        //printf("getWorkspaceSize batch_size = %d\n", batch_size);
        //printf("getWorkspaceSize offsetHeight = %d\n", offsetHeight);
        //printf("getWorkspaceSize offsetWidth = %d\n", offsetWidth);

        //printf("getWorkspaceSize nInputPlane = %d\n", nInputPlane);
        //printf("getWorkspaceSize inputHeight = %d\n", inputHeight);
        //printf("getWorkspaceSize inputWidth = %d\n", inputWidth);
        //printf("getWorkspaceSize nOutputPlane = %d\n", nOutputPlane);
        //printf("getWorkspaceSize outputHeight = %d\n", outputHeight);
        //printf("getWorkspaceSize outputWidth = %d\n", outputWidth);

        int kW = config_->weights_[0]->size(3);
        int kH = config_->weights_[0]->size(2);
        //printf("getWorkspaceSize %d\n", inputs[2].dims);
        //printf("getWorkspaceSize kW = %d\n", kW);
        //printf("getWorkspaceSize kH = %d\n", kH);
        int im2col_step = std::min(32, batch_size);

        size_t col_size = getAlignedSize(nInputPlane * kW * kH * im2col_step * outputHeight *
            outputWidth * sizeof_dtype);

        size_t out_size = 0;
        if (im2col_step != 1)
            out_size = getAlignedSize(batch_size * nOutputPlane * outputHeight * outputWidth *
                sizeof_dtype);
        //printf("getWorkspaceSize col_size %d\n", col_size);
        //printf("getWorkspaceSize out_size %d\n", out_size);
        return col_size + out_size;
        //int kernel_size = 3;
        //int deformable_group = 1;
        //size_t im2colSize = (size_t)inputs[0].dims.d[1] * kernel_size * kernel_size * outputs[0].dims.d[2] * outputs[0].dims.d[3];
        //size_t maskSize = (size_t)inputs[0].dims.d[2] * inputs[0].dims.d[3] * kernel_size * kernel_size * deformable_group;
        //config_->workspace_size_ = (im2colSize + maskSize) * config_->max_batch_size_ * TRT::data_type_size(config_->usage_dtype_);
        //return config_->workspace_size_;
            
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept{
        //printf("getOutputDimensions\n");
        //printf("output_dims[0] = %d\n", inputs[0].d[0]->getConstantValue());
        //printf("output_dims[1] = %d\n", exprBuilder.constant(config_->weights_[0]->size(0))->getConstantValue());
        //printf("output_dims[2] = %d\n", inputs[1].d[2]->getConstantValue());
        //printf("output_dims[3] = %d\n", inputs[1].d[3]->getConstantValue());


        nvinfer1::DimsExprs output_dims;
        output_dims.nbDims = 4;
        output_dims.d[0] = inputs[0].d[0];
        output_dims.d[1] = exprBuilder.constant(config_->weights_[0]->size(0));
        //output_dims.d[1] = inputs[2].d[0];
        output_dims.d[2] = inputs[1].d[2];
        output_dims.d[3] = inputs[1].d[3];
        return output_dims;
    }


    template<typename DataType>
    int enqueue_native(cublasHandle_t handle, const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, ConvParams params, void* workspace, cudaStream_t stream) {
        
        int batch = 1;
        int channels = inputs[0].channel();
        int height = inputs[0].height();
        int width = inputs[0].width();
        int channels_out = outputs[0].channel();
        int kernel_h = weights[0].height();
        int kernel_w = weights[0].width();
        //printf("kernel_h = %d\n", kernel_h);
        //printf("kernel_w = %d\n", kernel_w);
        const void* x = inputs[0].ptr_;
        const void* offset = inputs[1].ptr_;
        const void* weight = weights[0].ptr_;
        //const void* weight = config_->weights_[0].get();
        void* output = outputs[0].ptr_;

        int dW = params.stride[0];
        int dH = params.stride[1];
        int padW = params.padding[0];
        int padH = params.padding[1];
        int dilationW = params.dilation[0];
        int dilationH = params.dilation[1];
        int group = params.groups;
        int deformable_group = params.deformable_groups;
        int im2col_step = params.im2col_step;
        //printf("dW = %d\n", dW);
        //printf("dH = %d\n", dH);
        //printf("padW = %d\n", padW);
        //printf("padH = %d\n", padH);
        //printf("dilationW = %d\n", dilationW);
        //printf("dilationH = %d\n", dilationH);
        //printf("group = %d\n", group);
        //printf("deformable_group = %d\n", deformable_group);
        //printf("im2col_step = %d\n", im2col_step);

        auto data_type = inputs[0].dtype_;
        switch (data_type) {
        case TRT::DataType::Float:
            deform_conv<float>((float*)x, (float*)weight, (float*)offset, (float*)output, workspace,
                batch, channels, height, width, channels_out, kernel_w, kernel_h,
                dH, dW, padH, padW, dilationW,
                dilationH, group, deformable_group, im2col_step, cublasHandle_, stream);

            break;
        case TRT::DataType::Float16:
            deform_conv<half>((half*)x, (half*)weight, (half*)offset, (half*)output, workspace,
                batch, channels, height, width, channels_out, kernel_w, kernel_h,
                dH, dW, padH, padW, dilationW,
                dilationH, group, deformable_group, im2col_step, cublasHandle_, stream);
            break;
        default:
            return 1;
        }

        return 0;

    }
    int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream)
    {
		//printf("goning to enqueue reload\n");

        const char* jsonString = config_->info_.c_str();
        ConvParams params;
        parseJson(jsonString, &params);
        if (config_->usage_dtype_ == TRT::DataType::Float) {
            enqueue_native<float>(cublasHandle_, inputs, outputs, weights, params, workspace, stream);
        }
        else if (config_->usage_dtype_ == TRT::DataType::Float16) {
            // enqueue_native<__half>(cublasHandle_, inputs, outputs, weights, workspace, stream);
            INFOF("not implement function");
        }
        else {
            INFOF("not implement function");
        }
        return 0;
    }
};

RegisterPlugin(DCNv1);