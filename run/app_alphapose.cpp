#include <builder/trt_builder.hpp>
#include "app_alphapose/alpha_pose.hpp"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#if defined(_WIN32)
#include <Windows.h>
#include <wingdi.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")  
#endif
#include <iostream>
using namespace std;
using namespace cv;
bool requires(const char* name);
int app_alphapose() {

    TRT::set_device(0);

    const char* name = "alpha-pose-136";

    if (not requires(name))
        return 0;
    string onnx_file = "alpha-pose-136.onnx";
    string model_file = "alpha-pose-136.FP32.trtmodel";
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(
            TRT::Mode::FP32,            // FP32、FP16、INT8
            test_batch_size,            // max_batch_size
            onnx_file,                  // source
            model_file                  // save to
        );
    }

    Mat image = imread("inference/liudehua.jpg");
    auto engine = AlphaPose::create_infer(model_file, 0);
    auto box = Rect(0, 0, 666, 1017);
    double inferenceTime(0.0), t1(0.0);
    std::vector<tuple<cv::Mat, cv::Rect>> pose_input;
    int size_b = 16;
    for (int i = 0; i < size_b; ++i) {
		pose_input.push_back(make_tuple(image, box));
	}
    while (true) {
        t1 = static_cast<double>(cv::getTickCount());
        auto keys = engine->commits(pose_input);
        keys.back().get();
        inferenceTime = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
        std::cout << "FPS:" << int(1000.0f / inferenceTime * 100)*size_b / 100.0f << std::endl;

    }




    //while (true)
    //{
    //    t1 = static_cast<double>(cv::getTickCount());
    //    auto keys = engine->commit(make_tuple(image, box)).get();
    //    inferenceTime = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
    //    std::cout << "FPS:" << int(1000.0f / inferenceTime * 100) / 100.0f << std::endl;


    //    for (int i = 0; i < keys.size(); ++i) {
    //        float x = keys[i].x;
    //        float y = keys[i].y;
    //        if (keys[i].z > 0.05) {
    //            cv::circle(image, Point(x, y), 5, Scalar(0, 255, 0), 1, 16);
    //        }
    //    }
    //}
    auto save_file = "pose.show.jpg";

    imwrite(save_file, image);
    return 0;
}
