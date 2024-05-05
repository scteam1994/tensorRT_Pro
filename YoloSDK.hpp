//#pragma once
//#ifndef YOLO_SDK_HPP
//#define YOLO_SDK_HPP
//#include "global_export.h"
//#include <stdio.h>
//#include <string.h>
//#include <common/ilogger.hpp>
//#include <functional>
//#include <stdlib.h>
//#include <opencv2/opencv.hpp>
//#include "application/app_yolo/yolo.hpp"
//#if defined(_WIN32)
//
//#include <Windows.h>
//#include <wingdi.h>
//#include <Shlwapi.h>
//#pragma comment(lib, "shlwapi.lib")  
//#endif
//#include <iostream>
//#endif
//using namespace std;
//using namespace cv;
//namespace YOLOSDK {
//    struct TRT_EXPORT Param {
//        int device_id;
//        int img_width;
//        int img_height;
//        const char* name_yolo;
//        string face_folder;
//        vector<Mat> face_vector;
//        bool show_res;
//        Param() {
//            device_id = 0;
//            img_width = 640;
//            img_height = 640;
//            name_yolo = "yolov5s-hand-person-0.979";
//            show_res = false;
//        }
//
//    };
//    struct TRT_EXPORT Result {
//        cv::Rect box;
//        float score;
//        int class_id;
//    };
//
//    class TRT_EXPORT YoloObj {
//    private:
//        shared_ptr<Yolo::Infer> engine_yolo;
//        float* bank = nullptr;
//        float* bank_d = nullptr;
//    public:
//        bool show_res = false;
//        static bool compile_yolo(int input_width, int input_height, const char* name);
//        // init engine
//
//        void init(YOLOSDK::Param& param);
//        void infer(cv::Mat& frame, YOLOSDK::Result& res);
//
//    };
//}