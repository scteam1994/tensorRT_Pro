#pragma once
#ifndef POSE_SDK_HPP
#define POSE_SDK_HPP
#include "global_export.h"
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>
#include <builder/trt_builder.hpp>
#include "app_alphapose/alpha_pose.hpp"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "app_yolo/yolo.hpp"
#include "app_yolo/multi_gpu.hpp"

#if defined(_WIN32)
#include <Windows.h>
#include <wingdi.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")  
#endif
#include <iostream>
#endif
using namespace std;
using namespace cv;
namespace Pose {
    struct Param {
        int device_id;
        string name_alpha;
        string name_yolo;


    };
    struct Person {
        vector<Point3f> keypoints;
        Rect box;
    };
    typedef vector<Person> Result;
    class TRT_EXPORT combine_pose {
    public:
        
        //vector<string> cocolabels = {
        //"person", "bicycle", "car", "motorcycle", "airplane",
        //"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        //"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        //"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        //"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        //"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        //"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        //"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        //"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        //"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        //"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        //"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        //"scissors", "teddy bear", "hair drier", "toothbrush"
        //};

        void init(Param param);
        void combine_infer(cv::Mat& img, Result &res);
    };
}