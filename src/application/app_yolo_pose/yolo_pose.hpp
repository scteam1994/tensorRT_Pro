#pragma once

#ifndef YOLO_POSE_HPP
#define YOLO_POSE_HPP

#include "global_export.h"

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>


namespace YoloPose {

    using namespace std;
    using namespace cv;

    typedef tuple<Mat, Rect> Input;

    enum class TRT_EXPORT NMSMethod : int {
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    class TRT_EXPORT Infer {
    public:
        virtual shared_future<vector<vector<Point3f>>> commit(const Input& input) = 0;
        virtual vector<shared_future<vector<vector<Point3f>>>> commits(const vector<Input>& inputs) = 0;
    };

    TRT_EXPORT shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        float confidence_threshold = 0.25f, float nms_threshold = 0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 32
    );

}; // namespace HandPose

#endif // HAND_POSE_HPP