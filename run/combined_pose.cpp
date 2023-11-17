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
using namespace std;
using namespace cv;
bool requires(const char* name);
class combine_pose {
private:
    vector<string> cocolabels = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    shared_ptr<Yolo::Infer> engine_yolo;
    shared_ptr<AlphaPose::Infer> engine_alpha;
    bool status = true;
    Yolo::BoxArray yolo_boxes;
    Mat current_frame;
public:
    void init() {
        int deviceid = 0;
        TRT::set_device(deviceid);

        const char* name_alpha = "alpha-pose-136";
        const char* name_yolo = "yolov5s";
        if (!requires(name_alpha))
            return;
        if (!requires(name_yolo))
            return;
        string onnx_file_yolo = "yolov5s.onnx";
        string model_file_yolo = "yolov5s.FP32.trtmodel";
        string onnx_file_alpha = "alpha-pose-136.onnx";
        string model_file_alpha = "alpha-pose-136.FP32.trtmodel";
        int test_batch_size = 16;
        if (not iLogger::exists(model_file_yolo)) {
            TRT::compile(
                TRT::Mode::FP32,                       // FP32、FP16、INT8
                test_batch_size,            // max batch size
                onnx_file_yolo,                  // source 
                model_file_yolo,                 // save to
                {}
            );
        }

        if (!iLogger::exists(model_file_alpha)) {
            TRT::compile(
                TRT::Mode::FP32,            // FP32、FP16、INT8
                test_batch_size,            // max_batch_size
                onnx_file_alpha,                  // source
                model_file_alpha                  // save to
            );
        }

        engine_yolo = Yolo::create_infer(
            "yolov5s.FP32.trtmodel",                // engine file
            Yolo::Type::V5,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
            0,                   // gpu id
            0.25f,                      // confidence threshold
            0.45f,                      // nms threshold
            Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
            1024,                       // max objects
            false                       // preprocess use multi stream
        );
        engine_alpha = AlphaPose::create_infer(model_file_alpha, 0);
        if (engine_yolo == nullptr) {
            INFOE("Engine_yolo is nullptr");
            return;
        }
        if (engine_alpha == nullptr) {
            INFOE("Engine_alpha is nullptr");
            return;
        }
    }
    void yolo_thread_func() {
        double inferenceTime(0.0), t1(0.0);
        //string camid = "rtsp://liuzixiang:test@12$@192.168.120.64/h264/ch1/main/av_stream";
        //cv::VideoCapture capture(0);
        //capture.set(cv::CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
        cv::Mat img = imread("inference/gril.jpg");
        
        //if (!capture.isOpened())
        //{
        //    std::cout << "Failed to open camera with index " << camid << std::endl;
        //}
        while (true)
        {
            if (!status) {
				continue;
			}
            
            inferenceTime = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
            t1 = static_cast<double>(cv::getTickCount());
            std::cout << "FPS:" << int(1000.0f / inferenceTime * 100) / 100.0f << std::endl;
            

            //capture >> current_frame;
            current_frame = img.clone();

            if (current_frame.empty()) {
                cout << "frame is empty" << endl;
                continue;
            }
            INFO("start yolo");
            yolo_boxes = engine_yolo->commit(current_frame).get();
            INFO("End yolo");
            status = false;
            inferenceTime = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
            cout << "yolo time:" << inferenceTime << endl;
            
        }
        //capture.release();

    }

    void alpha_thread_func() {
        double inferenceTime2(0.0), t2(0.0), t1(0.0);
        while (true) {
            if (status) {
                continue;
            }
            cv::Mat frame;
            frame = current_frame.clone();
            Yolo::BoxArray boxes = yolo_boxes;
            status = true;
            
            t2 = static_cast<double>(cv::getTickCount());
            int person_num = 0;
            for (auto& obj : boxes) {
                if (obj.class_label == 0) {
                    person_num++;
                    cv::Rect box = Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top);
                    INFO("start alpha");
                    auto keys = engine_alpha->commit(make_tuple(frame, box)).get();
                    INFO("End alpha");
                    for (int i = 0; i < keys.size(); ++i) {
                        float x = keys[i].x;
                        float y = keys[i].y;
 /*                       if (keys[i].z > 0.05) {
                            cv::circle(current_frame, Point(x, y), 5, Scalar(0, 0, 255), 2);
                        }*/
                    }
                }
            }
            inferenceTime2 = (static_cast<double>(cv::getTickCount()) - t2) / cv::getTickFrequency() * 1000;
            cout<< "inference time:"<<inferenceTime2<<endl;
            cout << "person num:" << person_num << endl;
            
            //cout << "alpha" << endl;
            //cv::imshow("result", current_frame);
            //cv::waitKey(1);
        }

    }

    void combine_infer() {
        thread yolo_thread(&combine_pose::yolo_thread_func, this);
        yolo_thread.detach();
        //thread alpha_thread(&combine_pose::alpha_thread_func, this);
        //alpha_thread.join();
        alpha_thread_func();
    }

};

int combine_infer() {

    int deviceid = 0;
    TRT::set_device(deviceid);

    const char* name_alpha = "alpha-pose-136";
    const char* name_yolo = "yolov5s";
    if (!requires(name_alpha))
        return 0;
    if (!requires(name_yolo))
        return 0;
    string onnx_file_yolo = "yolov5s.onnx";
    string model_file_yolo = "yolov5s.FP32.trtmodel";
    string onnx_file_alpha = "alpha-pose-136.onnx";
    string model_file_alpha = "alpha-pose-136.FP32.trtmodel";
    int test_batch_size = 16;
    if (not iLogger::exists(model_file_yolo)) {
        TRT::compile(
            TRT::Mode::FP32,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file_yolo,                  // source 
            model_file_yolo,                 // save to
            {}
        );
    }

    if (!iLogger::exists(model_file_alpha)) {
        TRT::compile(
            TRT::Mode::FP32,            // FP32、FP16、INT8
            test_batch_size,            // max_batch_size
            onnx_file_alpha,                  // source
            model_file_alpha                  // save to
        );
    }

    auto engine_yolo = Yolo::create_infer(
        model_file_yolo,                // engine file
        Yolo::Type::V5,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
        deviceid,                   // gpu id
        0.25f,                      // confidence threshold
        0.45f,                      // nms threshold
        Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                       // max objects
        false                       // preprocess use multi stream
    );
    auto engine_alpha = AlphaPose::create_infer(model_file_alpha, 0);
    if (engine_yolo == nullptr) {
        INFOE("Engine_yolo is nullptr");
        return 0;
    }
    if (engine_alpha == nullptr) {
        INFOE("Engine_alpha is nullptr");
        return 0;
    }
    double inferenceTime(0.0), t1(0.0);
    string camid = "rtsp://liuzixiang:test@12$@192.168.120.64/h264/ch1/main/av_stream";
    cv::VideoCapture capture(0);
    capture.set(cv::CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));

    t1 = static_cast<double>(cv::getTickCount());
    if (!capture.isOpened())
    {
        std::cout << "Failed to open camera with index " << camid << std::endl;
    }
    cv::Mat frame = imread("inference/gril.jpg");
    auto t2 = static_cast<double>(cv::getTickCount());
    while (true) {
        auto t1 = static_cast<double>(cv::getTickCount());
        cv::Mat img = frame.clone();
        if (img.empty()) {
			continue;
		}
        auto boxes = engine_yolo->commit(img).get();
        float inference_yolo_time = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
        cout << "yolo time:" << inference_yolo_time << endl;
        t1 = static_cast<double>(cv::getTickCount());
        int person_num = 0;
        for (auto& obj : boxes) {
            if (obj.class_label == 0) {
                person_num++;
                cv::Rect box = Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top);
                auto keys = engine_alpha->commit(make_tuple(img, box)).get();
                for (int i = 0; i < keys.size(); ++i) {
                    float x = keys[i].x;
                    float y = keys[i].y;
                    //if (keys[i].z > 0.05) {
                    //    cv::circle(img, Point(x, y), 5, Scalar(0, 0, 255), 2);
                    //}
                }
            }
        }
        float inference_average_time = (static_cast<double>(cv::getTickCount()) - t2) / cv::getTickFrequency() * 1000;
        cout << "inference time:" << inference_average_time << endl;
        std::cout << "FPS:" << int(1000.0f / inference_average_time * 100) / 100.0f << std::endl;
        cout << "person_num: "<<person_num << endl;
        t2 = static_cast<double>(cv::getTickCount());
        float inference_alpha_time = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
        cout << "alpha time:" << inference_alpha_time << endl;
    }
    //  cv::imshow("result", img);
    //  cv::waitKey(0);
    return 0;
}

int combine_infer_main() {
    combine_infer();
    //combine_pose combine;
    //combine.init();
    //combine.combine_infer();

    return 0;
}



