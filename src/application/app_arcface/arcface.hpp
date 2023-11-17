#pragma once
#ifndef ARCFACE_HPP
#define ARCFACE_HPP
#include "global_export.h"
#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace Arcface{

    using namespace std;

    struct TRT_EXPORT landmarks{
        float points[10];
    };

    typedef cv::Mat_<float>           feature;
    typedef tuple<cv::Mat, landmarks> commit_input;

    class TRT_EXPORT Infer{
    public:
        virtual shared_future<feature>         commit (const commit_input& input)          = 0;
        virtual vector<shared_future<feature>> commits(const vector<commit_input>& inputs) = 0;
    };

    TRT_EXPORT cv::Mat face_alignment(const cv::Mat& image, const landmarks& landmark);
    TRT_EXPORT shared_ptr<Infer> create_infer(const string& engine_file, int gpuid=0);

}; // namespace RetinaFace

#endif // ARCFACE_HPP