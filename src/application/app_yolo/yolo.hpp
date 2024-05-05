#pragma once

#ifndef YOLO_HPP
#define YOLO_HPP

#include "global_export.h"

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <tensorRT/common/trt_tensor.hpp>
#include <application/common/object_detector.hpp>

/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace Yolo{

    using namespace std;
    using namespace ObjectDetector;

    enum class TRT_EXPORT Type : int{
        V5 = 0,
        X  = 1,
        V3 = 2,
        V7 = 3
    };

    enum class TRT_EXPORT NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    TRT_EXPORT void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    class TRT_EXPORT Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    TRT_EXPORT shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );

    TRT_EXPORT const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP