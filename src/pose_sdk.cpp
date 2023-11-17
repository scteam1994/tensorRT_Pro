#pragma once
#include "pose_sdk.hpp"
bool requires(const char* name);
class combine_pose {
private:
    shared_ptr<Yolo::Infer> engine_yolo;
    shared_ptr<AlphaPose::Infer> engine_alpha;
    bool show_res = false;
public:
    void init(Pose::Param param) {
        int deviceid = param.device_id;
        TRT::set_device(param.device_id);

        const char* name_alpha = "alpha-pose-136";
        const char* name_yolo = "yolov5s";
        if (!requires(name_alpha))
            return;
        if (!requires(name_yolo))
            return;
        string onnx_file_yolo = iLogger::format("%s.onnx", name_yolo);
        string model_file_yolo = iLogger::format("%s.FP32.trtmodel", name_yolo);
        string onnx_file_alpha = iLogger::format("%s.onnx", name_alpha);
        string model_file_alpha = iLogger::format("%s.FP32.trtmodel", name_alpha);
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


    void combine_infer(cv::Mat& frame, Pose::Result& res) {

        if (frame.empty()) {
            INFOE("Frame is empty");
            return;
        }
        int w = frame.cols;
        int h = frame.rows;
        Yolo::BoxArray boxes = engine_yolo->commit(frame).get();
        int person_num = 0;
        for (auto& obj : boxes) {
            if (obj.class_label == 0) {
                Pose::Person res_tmp;
                person_num++;
                int x = obj.left>0?obj.left:0;
                int y = obj.top>0?obj.top:0;
                int width = (obj.right - obj.left)> (w - x) ? (w - x) : (obj.right - obj.left);
                int height = (obj.bottom - obj.top) > (h - y) ? (h - y) : (obj.bottom - obj.top);
                cv::Rect box = Rect(x, y, width, height);
                vector<Point3f> keys = engine_alpha->commit(make_tuple(frame, box)).get();
                res_tmp.box = box;
                res_tmp.keypoints = keys;
                if (show_res) {
                    for (int i = 0; i < keys.size(); ++i) {
                        float x = keys[i].x;
                        float y = keys[i].y;
                        if (keys[i].z > 0.05) {
                            cv::circle(frame, Point(x, y), 5, Scalar(0, 255, 0), 1, 16);
                        }
                    }
                }
                res.push_back(res_tmp);
            }
        }
    }

};