#pragma once

#ifndef HAND_POSE_HPP
#define HAND_POSE_HPP

#include "global_export.h"

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>


namespace HandPose {

    using namespace std;
    using namespace cv;

    typedef tuple<Mat, Rect> Input;

    class TRT_EXPORT Infer {
    public:
        virtual shared_future<vector<Point3f>> commit(const Input& input) = 0;
        virtual vector<shared_future<vector<Point3f>>> commits(const vector<Input>& inputs) = 0;
    };

    TRT_EXPORT shared_ptr<Infer> create_infer(const string& engine_file, int gpuid);

}; // namespace HandPose

#endif // HAND_POSE_HPP