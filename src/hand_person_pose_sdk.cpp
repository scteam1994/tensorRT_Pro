#pragma once
#include <builder/trt_builder.hpp>
#include <common/ilogger.hpp>
#include "app_yolo/multi_gpu.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include<logger.h> // add file:   ../TensorRT-8.4.2.4/samples/common/logger.cpp
#include "hand_person_pose_sdk.hpp"
#include <mutex>
#include <shared_mutex>
int frame_w;
int frame_h;
bool requires(const char* name);
std::shared_mutex mtx;
int cnt = 0;
namespace Pose {

    void draw(cv::Mat& img, const std::vector<cv::Point3f>& kps, std::vector<std::tuple<int, int>> kp_tuples, std::vector<cv::Scalar> colors,int prefix_len = 0) {
        if (kps.size() == 0) {
            return;
        }
        int thick = 2;
        for (int i = 0; i < kp_tuples.size(); i++) {
            int pt1_index = prefix_len + std::get<0>(kp_tuples[i]);
            int pt2_index = prefix_len + std::get<1>(kp_tuples[i]);
            if (kps[pt1_index].z > 0.01 && kps[pt2_index].z > 0.01) {
				cv::line(img, cv::Point(kps[pt1_index].x, kps[pt1_index].y), cv::Point(kps[pt2_index].x, kps[pt2_index].y), colors[i], thick);
			}
            //cv::line(img, cv::Point(kps[pt1_index].x, kps[pt1_index].y), cv::Point(kps[pt2_index].x, kps[pt2_index].y), colors[i], thick);
          /*  cv::putText(img, std::to_string(i), cv::Point(kps[pt1_index].x, kps[pt1_index].y), cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);*/
        }

    }
    void generateRandomColors(int n, std::vector<cv::Scalar>& colors) {
        // 设置随机数种子
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < n; ++i) {
            // 生成三个随机的颜色通道值
            int blue = std::rand() % 256;
            int green = std::rand() % 256;
            int red = std::rand() % 256;

            // 使用cv::Scalar创建颜色
            cv::Scalar color(blue, green, red);
            colors.push_back(color);
        }
    }
    bool SortBox(const Yolo::Box& a, const Yolo::Box& b) {
        //按最接近中心
        return (a.right + a.left -frame_w) * (a.right + a.left - frame_w) + (a.bottom + a.top -frame_h) * (a.bottom + a.top - frame_h) < (b.right + b.left - frame_w) * (b.right + b.left - frame_w) + (b.bottom + b.top - frame_h) * (b.bottom + b.top - frame_h);
        //按面积最大
        //return (a.right-a.left)*(a.bottom-a.top) > (b.right-b.left)*(b.bottom-b.top);
        //按平均
        //float x = (a.right + a.left - frame_w)* (a.right + a.left - frame_w) + (a.bottom + a.top - frame_h) * (a.bottom + a.top - frame_h) + (a.right - a.left) * (a.bottom - a.top);
        //float y = (b.right + b.left - frame_w)* (b.right + b.left - frame_w) + (b.bottom + b.top - frame_h) * (b.bottom + b.top - frame_h) + (b.right - b.left) * (b.bottom - b.top);
        //return x > y;
    }

    bool SortBoxKpts(const vector<Point3f>& a, vector<Point3f>& b) {
		//按最接近中心
		return (a[1].x + a[0].x - frame_w) * (a[1].x + a[0].x - frame_w) + (a[1].y + a[0].y - frame_h) * (a[1].y + a[0].y - frame_h) < (b[1].x + b[0].x - frame_w) * (b[1].x + b[0].x - frame_w) + (b[1].y + b[0].y - frame_h) * (b[1].y + b[0].y - frame_h);
		//按面积最大
		//return (a[1].x - a[0].x) * (a[1].y - a[0].y) > (b[1].x - b[0].x) * (b[1].y - b[0].y);
		//按平均
		//float x = (a[1].x + a[0].x - frame_w) * (a[1].x + a[0].x - frame_w) + (a[1].y + a[0].y - frame_h) * (a[1].y + a[0].y - frame_h) + (a[1].x - a[0].x) * (a[1].y - a[0].y);
		//float y = (b[1].x + b[0].x - frame_w) * (b[1].x + b[0].x - frame_w) + (b[1].y + b[0].y - frame_h) * (b[1].y + b[0].y - frame_h) + (b[1].x - b[0].x) * (b[1].y - b[0].y);
		//return x > y;

    }

    void Combine_pose::YoloBox2cvrect (Yolo::Box& src, cv::Rect& dst) {
        int x, y, width, height;
        //防止越界
        width = (src.right - src.left);
        height = (src.bottom - src.top);
        x = src.left > 0 ? src.left : 0;
        y = src.top > 0 ? src.top : 0;
        width = x + width < frame_w ? width : frame_w - x;
        height = y + height < frame_h ? height : frame_h - y;
        dst = Rect(x, y, width, height);
    }
    Yolo::BoxArray Combine_pose::find_hand_box(const Yolo::BoxArray & boxes, const vector<Point3f> person, int& single, float confidence_thresh) {
        Yolo::BoxArray result;
        //9+2,10+2
        float norm_bot_l = (person[11].x - person[9].x) * (person[11].x - person[9].x) + (person[11].y - person[9].y) * (person[11].y - person[9].y);
        float norm_bot_r = (person[12].x - person[10].x) * (person[12].x - person[10].x) + (person[12].y - person[10].y) * (person[12].y - person[10].y);
        //single : -1->none,0->all, 1->right,2->left
        //person id :9->left 10->right
        if (boxes.size() == 0) {
            single = -1;
            //kalman_filter_lh->resetFilter();
            //kalman_filter_rh->resetFilter();
            return result;
        }
        else if (boxes.size() == 1) {
            float c_x = (boxes[0].left + boxes[0].right) / 2;
            float c_y = (boxes[0].top + boxes[0].bottom) / 2;
            float dist_l = (c_x - person[11].x) * (c_x - person[11].x)  + (c_y - person[11].y) * (c_y - person[11].y) ;
            dist_l /= norm_bot_l;
            float dist_r = (c_x - person[12].x) * (c_x - person[12].x) + (c_y - person[12].y) * (c_y - person[12].y) ;
            dist_r /= norm_bot_r;
            // find min dist

            if (dist_l > dist_r && dist_r<dist_thresh) {
                //right
                //std::cout << "right" << dist_r << std::endl;
                single = 1;
                result.push_back(boxes[0]);
                //kalman_filter_lh->resetFilter();
            }
            else if (dist_r > dist_l && dist_l < dist_thresh) {
                //left
                //std::cout << "left" << dist_l << std::endl;
                single = 2;
                result.push_back(boxes[0]);
                //kalman_filter_rh->resetFilter();
            }
            else
            {
                single = -1;
                //kalman_filter_lh->resetFilter();
                //kalman_filter_rh->resetFilter();
                return result;
            }
            return result;
        }

        int left_hand_idx = 0, right_hand_idx = 0;
        int secomd_min_left_idx = 0;
        int secomd_min_right_idx = 0;
        float min_left_dist = 999999.f;
        float min_right_dist = 999999.f;
        float secomd_min_left_dist = 999999.f;
        float secomd_min_right_dist = 999999.f;

        for (int i = 0; i < boxes.size(); i++) {
            float c_x = (boxes[i].left + boxes[i].right) / 2;
            float c_y = (boxes[i].top + boxes[i].bottom) / 2;


            float dist_l = (c_x - person[11].x) * (c_x - person[11].x)  + (c_y - person[11].y) * (c_y - person[11].y) ;
            dist_l /= norm_bot_l;
            if (dist_l < min_left_dist) {
                min_left_dist = dist_l;
                left_hand_idx = i;
            }
            else if (dist_l< secomd_min_left_dist && dist_l!= min_left_dist)
            {
                secomd_min_left_dist = dist_l;
                secomd_min_left_idx = i;
            }
            float dist_r = (c_x - person[12].x) * (c_x - person[12].x)  + (c_y - person[12].y) * (c_y - person[12].y);
            dist_r /= norm_bot_r;
            if (dist_r < min_right_dist) {
                secomd_min_right_dist = min_right_dist;
                min_right_dist = dist_r;
                secomd_min_right_idx = right_hand_idx;
                right_hand_idx = i;
            }
            else if (dist_r < secomd_min_right_dist && dist_r != min_right_dist)
            {
                secomd_min_right_dist = dist_r;
                secomd_min_right_idx = i;
            }



        }
        //std::cout << "left" << min_left_dist << std::endl;
        //std::cout << "right" << min_right_dist << std::endl;


        //左右手都是一只，这一值距离左右都最近
        if (right_hand_idx == left_hand_idx)
        {
            if (min_left_dist > min_right_dist)
            {
                min_left_dist = secomd_min_left_dist;
                left_hand_idx = secomd_min_left_idx;
            }
            else
            {
                min_right_dist = secomd_min_right_dist;
                right_hand_idx = secomd_min_right_idx;
            }
        }

        if (min_left_dist > dist_thresh && min_right_dist > dist_thresh)
        {
            single = -1;
            //kalman_filter_lh->resetFilter();
            //kalman_filter_rh->resetFilter();
        }
        else if (min_left_dist < dist_thresh && min_right_dist > dist_thresh)
        {
            //left
            single = 2;
            result.push_back(boxes[left_hand_idx]);
            //kalman_filter_rh->resetFilter();
        }
        else if (min_left_dist > dist_thresh && min_right_dist < dist_thresh)
        {
            //right
            single = 1;
            result.push_back(boxes[right_hand_idx]);
            //kalman_filter_lh->resetFilter();
        }
        else
        {
            single = 0;
            result.push_back(boxes[left_hand_idx]);
            result.push_back(boxes[right_hand_idx]);
            
        }
        return result;
    }

    Yolo::BoxArray filterBoxesByClassLabel(const Yolo::BoxArray& boxes, int targetLabel, float confidence_thresh) {
        Yolo::BoxArray filteredBoxes;
        std::copy_if(boxes.begin(), boxes.end(), std::back_inserter(filteredBoxes),
            [targetLabel, confidence_thresh](const Yolo::Box& box) {
                return box.class_label == targetLabel && box.confidence > confidence_thresh;
            });
        return filteredBoxes;
    }

    void Combine_pose::init(Pose::Param param) {
        int deviceid = param.device_id;
        max_person_num = param.max_person_num;
        TRT::set_device(param.device_id);
        if (!requires(param.name_handpose))
            return;
        if (!requires(param.name_yolo))
            return;
        if (!requires(param.name_bodypose))
            return;
        // if show res
        _show = param.code_show;
        smooth = param.smooth;
        kalman = param.kalman;
        cache_size = param.cache_size;
        moumentent_thresh = param.moumentent_thresh;
        dist_thresh = param.dist_thresh;
        body_pkt_num = param.body_pkt_num;
        hand_pkt_num = param.hand_pkt_num;
        hand_rec_active = param.hand_rec_active;
        auto mode = TRT::Mode(param.mode);
        auto mode_name = TRT::mode_string(mode);
        string onnx_file_yolo = iLogger::format("%s.onnx", param.name_yolo);
        string model_file_yolo = iLogger::format("%s.%s.trtmodel", param.name_yolo, mode_name);
        string onnx_file_hand = iLogger::format("%s.onnx", param.name_handpose);
        string model_file_hand = iLogger::format("%s.%s.trtmodel", param.name_handpose, mode_name);
        string onnx_file_body = iLogger::format("%s.onnx", param.name_bodypose);
        string model_file_body = iLogger::format("%s.%s.trtmodel", param.name_bodypose, mode_name);
        int test_batch_size = 16;
        auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {

            INFO("Int8 %d / %d", current, count);

            for (int i = 0; i < files.size(); ++i) {
                auto image = cv::imread(files[i]);
                Yolo::image_to_tensor(image, tensor, Yolo::Type::V5, i);
            }
            };
        if (not iLogger::exists(model_file_yolo)) {

            TRT::compile(
				mode,                       // FP32、FP16、INT8
				test_batch_size,            // max batch size
				onnx_file_yolo,                  // source 
				model_file_yolo,                 // save to
                {},
				int8process,
                "inference"
			);


        }

        if (!iLogger::exists(model_file_hand)) {
            TRT::compile(
                mode,            // FP32、FP16、INT8
                test_batch_size,            // max_batch_size
                onnx_file_hand,                  // source
                model_file_hand,                  // save to
                {},
                int8process,
                "inference"
            );
        }

        if (!iLogger::exists(model_file_body)) {
            TRT::compile(
                mode,            // FP32、FP16、INT8
                test_batch_size,            // max_batch_size
                onnx_file_body,                  // source
                model_file_body,                  // save to
                {},
                int8process,
                "inference"
            );
        }

        engine_yolo = Yolo::create_infer(
            model_file_yolo,                // engine file
            Yolo::Type::V5,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
            0,                   // gpu id
            0.1f,                      // confidence threshold
            0.45f,                      // nms threshold
            Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
            1024,                       // max objects
            false                       // preprocess use multi stream
        );
        engine_hand = AlphaPose::create_infer(model_file_hand, 0);

        engine_body = YoloPose::create_infer(model_file_body, 0);
        if (engine_yolo == nullptr) {
            INFOE("Engine_yolo is nullptr");
            return;
        }
        if (engine_hand == nullptr) {
            INFOE("Engine_hand is nullptr");
            return;
        }
        if (engine_body == nullptr) {
            INFOE("Engine_body is nullptr");
            return;
        }

        kalman_filter_lh = new KalmanFilter2D(hand_pkt_num);
        kalman_filter_rh = new KalmanFilter2D(hand_pkt_num);
        kalman_filter_b = new KalmanFilter2D(body_pkt_num+2);
        generateRandomColors(body_pkt_num, colors_body);
        generateRandomColors(hand_pkt_num, colors_hand);

    }
    void Combine_pose::change_param(Param param) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        max_person_num = param.max_person_num;
        _show = param.code_show;
        smooth = param.smooth;
        kalman = param.kalman;
        cache_size = param.cache_size;
        moumentent_thresh = param.moumentent_thresh;
        dist_thresh = param.dist_thresh;
        hand_rec_active = param.hand_rec_active;

        lock.unlock();
    }
    void Combine_pose::smooth_label(std::vector<cv::Point3f>& mean_keypoints, std::vector<std::vector<cv::Point3f>>& smooth_cache, vector<Point3f> _keypoints, int height, int cache_size, float moumentent_thresh) {
        if (smooth_cache.size() == 0)
        {
            smooth_cache.push_back(_keypoints);
        }
        else
        {
            cv::Point2f moumentent;
            moumentent.x = _keypoints[0].x - smooth_cache[smooth_cache.size() - 1][0].x;
            moumentent.y = _keypoints[0].y - smooth_cache[smooth_cache.size() - 1][0].y;
            moumentent.x = moumentent.x / height; //height不容易变
            moumentent.y = moumentent.y / height;
            //if (_keypoints[0].z==1) {
            //    std::cout << "moumentent:" << moumentent.x * moumentent.x + moumentent.y * moumentent.y << std::endl;
            //}
            
            if (moumentent.x * moumentent.x + moumentent.y * moumentent.y < moumentent_thresh)
            {
                smooth_cache.push_back(_keypoints);
            }
            else
            {
                smooth_cache.clear();
            }
        }


        if (smooth_cache.size() > cache_size)
        {
            smooth_cache.erase(smooth_cache.begin());
        }


        for (int i = 0; i < smooth_cache.size(); i++) //cache_size
        {
            for (int j = 0; j < smooth_cache[i].size(); j++)//17
            {
                mean_keypoints[j].x += smooth_cache[i][j].x;
                mean_keypoints[j].y += smooth_cache[i][j].y;
            }

        }
        for (int i = 0; i < mean_keypoints.size(); i++)
        {
            mean_keypoints[i].x /= smooth_cache.size();//cache_size
            mean_keypoints[i].y /= smooth_cache.size();
        }
    }
    void Combine_pose::combine_infer_hand(cv::Mat& frame, Pose::Result& res) {
        if (frame.empty()) {
            INFOE("Frame is empty");
            return;
        }
        frame_w = frame.cols;
        frame_h = frame.rows;
        auto hand_boxes_f = engine_yolo->commit(frame);
        auto body_boxes_keys_yolopose_f = engine_body->commit(make_tuple(frame, cv::Rect(0, 0, frame_w, frame_h)));
        vector<vector<Point3f>> person_boxes_keys = body_boxes_keys_yolopose_f.get();
        if (person_boxes_keys.size() == 0)
        {
            if (kalman)
            {
                kalman_filter_b->resetFilter();
                kalman_filter_rh->resetFilter();
                kalman_filter_lh->resetFilter();
            }
            return;
        }
        if (person_boxes_keys.size() > max_person_num) {
            std::sort(person_boxes_keys.begin(), person_boxes_keys.end(), SortBoxKpts);
            //保留前max_person_num个
            person_boxes_keys.erase(person_boxes_keys.begin() + max_person_num, person_boxes_keys.end());
        }
        Yolo::BoxArray hand_boxes_raw = filterBoxesByClassLabel(hand_boxes_f.get(), 0, confidence_thresh_hand);
        for (auto person_box_key : person_boxes_keys)
        {
            Pose::Person res_tmp;
            cv::Rect inputbox;
            inputbox.x = person_box_key[0].x>0?person_box_key[0].x:0;
            inputbox.y = person_box_key[0].y>0?person_box_key[0].y:0;
            inputbox.width = person_box_key[1].x - person_box_key[0].x;
            inputbox.height = person_box_key[1].y - person_box_key[0].y;
            inputbox.width = inputbox.x + inputbox.width < frame_w ? inputbox.width : frame_w - inputbox.x;
            inputbox.height = inputbox.y + inputbox.height < frame_h ? inputbox.height : frame_h - inputbox.y;
            res_tmp.body_box = inputbox;
            if (smooth)
            {
                vector<Point3f> mean_body_keypoints(person_box_key.size());
                for (int i = 0; i < mean_body_keypoints.size(); i++)
                {
                    mean_body_keypoints[i].x = 0;
                    mean_body_keypoints[i].y = 0;
                    mean_body_keypoints[i].z = 0;
                }
                smooth_label(mean_body_keypoints, smooth_cache_b, person_box_key, (person_box_key[1].y - person_box_key[0].y), cache_size, moumentent_thresh);
                res_tmp.body_keypoints = mean_body_keypoints;
            }
            else if (kalman)
            {
                kalman_filter_b->predict();
                kalman_filter_b->correct(person_box_key);
                res_tmp.body_keypoints = person_box_key;
            }
            else
            {
                
                res_tmp.body_keypoints = person_box_key;
            }
            vector<Point3f> person_key_tmp(person_box_key.begin() + 2, person_box_key.end());
            res_tmp.body_keypoints = person_key_tmp;
			int single;//single : -1->none,0->all, 1->right,2->left
			Yolo::BoxArray hand_boxes = find_hand_box(hand_boxes_raw, person_box_key, single, confidence_thresh_hand);

            //开始处理hand
            if (single == -1) {
                if (kalman)
                {
                    kalman_filter_rh->resetFilter();
                    kalman_filter_lh->resetFilter();
                }
                res.push_back(res_tmp);
                continue;

            }
            else
            {
                shared_future<vector<Point3f>> R_hand_keypoints;
                shared_future<vector<Point3f>> L_hand_keypoints;
                if (single == 0) {
                    cv::Rect inputbox_r;
                    YoloBox2cvrect(hand_boxes[1], inputbox_r);
                    R_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_r));

                    cv::Rect inputbox_l;
                    YoloBox2cvrect(hand_boxes[0], inputbox_l);
                    L_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_l));
                    res_tmp.R_hand_box = inputbox_r;
                    res_tmp.L_hand_box = inputbox_l;

                    if (smooth)
                    {
                        auto kptsr = R_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_R_hand_keypoints(kptsr.size());
                        for (int i = 0; i < mean_R_hand_keypoints.size(); i++)
                        {
                            mean_R_hand_keypoints[i].x = 0;
                            mean_R_hand_keypoints[i].y = 0;
                            mean_R_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_R_hand_keypoints, smooth_cache_rh, kptsr, inputbox_r.height, cache_size, moumentent_thresh);
                        res_tmp.R_hand_keypoints = mean_R_hand_keypoints;
                        auto kptsl = L_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_L_hand_keypoints(kptsl.size());
                        for (int i = 0; i < mean_L_hand_keypoints.size(); i++)
                        {
                            mean_L_hand_keypoints[i].x = 0;
                            mean_L_hand_keypoints[i].y = 0;
                            mean_L_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_L_hand_keypoints, smooth_cache_lh, kptsl, inputbox_l.height, cache_size, moumentent_thresh);
                        res_tmp.L_hand_keypoints = mean_L_hand_keypoints;
                    }
                    else if (kalman) {

                        kalman_filter_lh->predict();
                        kalman_filter_rh->predict();
                        auto kptsl = L_hand_keypoints.get();
                        auto kptsr = R_hand_keypoints.get();
                        kalman_filter_lh->correct(kptsl);
                        kalman_filter_rh->correct(kptsr);
                        res_tmp.L_hand_keypoints = kptsl;
                        res_tmp.R_hand_keypoints = kptsr;

                    }
                    else
                    {
                        res_tmp.R_hand_keypoints = R_hand_keypoints.get();
                        res_tmp.L_hand_keypoints = L_hand_keypoints.get();
                    }

                }
                if (single == 1) {
                    //right
                    cv::Rect inputbox_r;
                    YoloBox2cvrect(hand_boxes[0], inputbox_r);
                    R_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_r));
                    res_tmp.R_hand_box = inputbox_r;
                    if (smooth)
                    {
                        auto kptsr = R_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_R_hand_keypoints(kptsr.size());
                        for (int i = 0; i < mean_R_hand_keypoints.size(); i++)
                        {
                            mean_R_hand_keypoints[i].x = 0;
                            mean_R_hand_keypoints[i].y = 0;
                            mean_R_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_R_hand_keypoints, smooth_cache_rh, kptsr, inputbox_r.height, cache_size, moumentent_thresh);
                        res_tmp.R_hand_keypoints = mean_R_hand_keypoints;
                    }
                    else if (kalman) {
                        kalman_filter_rh->predict();
                        kalman_filter_lh->resetFilter();
                        auto kptsr = R_hand_keypoints.get();
                        kalman_filter_rh->correct(kptsr);
                        res_tmp.R_hand_keypoints = kptsr;

                    }
                    else
                    {
                        res_tmp.R_hand_keypoints = R_hand_keypoints.get();
                    }
                }

                if (single == 2) {
                    //left
                    cv::Rect inputbox_l;
                    YoloBox2cvrect(hand_boxes[0], inputbox_l);
                    L_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_l));
                    res_tmp.L_hand_box = inputbox_l;

                    if (smooth)
                    {
                        auto kptsl = L_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_L_hand_keypoints(kptsl.size());
                        for (int i = 0; i < mean_L_hand_keypoints.size(); i++)
                        {
                            mean_L_hand_keypoints[i].x = 0;
                            mean_L_hand_keypoints[i].y = 0;
                            mean_L_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_L_hand_keypoints, smooth_cache_lh, kptsl, inputbox_l.height, cache_size, moumentent_thresh);
                        res_tmp.L_hand_keypoints = mean_L_hand_keypoints;
                    }
                    else if (kalman) {
                        kalman_filter_lh->predict();
                        auto kptsl = L_hand_keypoints.get();
                        kalman_filter_lh->correct(kptsl);
                        kalman_filter_rh->resetFilter();
                        res_tmp.L_hand_keypoints = kptsl;

                    }
                    else
                    {
                        res_tmp.L_hand_keypoints = L_hand_keypoints.get();
                    }


                }

            }
            res.push_back(res_tmp);
        
        }
        for (auto& res_tmp : res)
        {
            switch (_show) {
            case hand_l_lp:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                break;
            case hand_r_lp:
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                break;
            case person_lp:
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body);
                break;
            case hand_l_box:
                if (res_tmp.L_hand_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                break;
            case hand_r_box:
                if (res_tmp.R_hand_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                break;
            case person_box:
                if (res_tmp.body_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.body_box, Scalar(0, 255, 0), 2);
                break;
            case all_kp:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body);
                break;
            case all_box:
                if (res_tmp.L_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.R_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.body_box.width != 0) {
                    cv::rectangle(frame, res_tmp.body_box, Scalar(255, 0, 0), 2);
                }
                break;
            case all:

                if (res_tmp.R_hand_box.width != 0) {
                    
                    //Rect box_ = res_tmp.R_hand_box;
                    //float rate = 0.1f;
                    //float pad_width = box_.width * (1 + 2 * rate);
                    //float pad_height = box_.height * (1 + 2 * rate);
                    //float scalex = 256 / pad_width;
                    //float scaley = 192 / pad_height;
                    //float i2d[6], d2i[6];
                    //i2d[0] = scalex;  i2d[1] = 0;      i2d[2] = -(box_.x + box_.width * 0.5) * scalex + 256 * 0.5;
                    //i2d[3] = 0;      i2d[4] = scaley;  i2d[5] = -(box_.y + box_.height * 0.5) * scaley + 192 * 0.5;
                    //cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
                    //cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
                    //cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
                    //cv::Mat imgshow;
                    //cv::warpAffine(frame, imgshow, m2x3_i2d, cv::Size(256, 192), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
                    //cv::imshow("hand", imgshow);
                    //
                    //std::ostringstream ss;
                    //ss << std::setw(3) << std::setfill('0') << cnt;
                    //cv::imwrite("handimg/hand" + ss.str() + ".jpg", imgshow);
                    //cnt++;
                    cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.body_box.width != 0) {
                    cv::rectangle(frame, res_tmp.body_box, Scalar(255, 0, 0), 2);
                }
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body);
                if (res_tmp.L_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                }
                break;
            case none:
                break;
            default:
                break;
            }
        }
    }

    void Combine_pose::combine_infer_hand_depreacated(cv::Mat& frame, Pose::Result& res) {
        if (frame.empty()) {
            INFOE("Frame is empty");
            return;
        }
        int r_hand_empty_count = 0;
        int l_hand_empty_count = 0;
        frame_w = frame.cols;
        frame_h = frame.rows;
        bool yolo_pose_find_person = false;
        vector<vector<Point3f>> body_keys_yolopose;
        Yolo::BoxArray raw_boxes = engine_yolo->commit(frame).get();
        if (raw_boxes.size() == 0) {
			return;
		}
        //只保留boxes中class_label==0的元素
        int person_label = static_cast<int>(yolo_detect_label::person);
        int hand_label = static_cast<int>(yolo_detect_label::hand);
        std::unique_lock<std::shared_mutex> lock(mtx);
        Yolo::BoxArray person_boxes = filterBoxesByClassLabel(raw_boxes, person_label, confidence_thresh_body);//person conf 0.5
        person_boxes.clear();
        //if (person_boxes.size() == 0) {
        //    body_keys_yolopose = engine_body->commit(make_tuple(frame, cv::Rect(0, 0, frame_w, frame_h))).get();
        //    if (body_keys_yolopose.size() > 0)
        //    {
        //        if (body_keys_yolopose[0].z > confidence_thresh_body)
        //        {
        //            Yolo::Box person = Yolo::Box(body_keys_yolopose[0].x, body_keys_yolopose[0].y, body_keys_yolopose[1].x - body_keys_yolopose[0].x,
        //                body_keys_yolopose[1].y - body_keys_yolopose[0].y, body_keys_yolopose[0].z, 1);
        //            person_boxes.push_back(person);
        //            yolo_pose_find_person = true;
        //        }
        //        else
        //        {
        //            return;
        //        }
        //    }
        //}
        int person_num = person_boxes.size();
        if (person_num > max_person_num) {
            std::sort(person_boxes.begin(), person_boxes.end(), SortBox);
            //保留前max_person_num个
            person_boxes.erase(person_boxes.begin() + max_person_num, person_boxes.end());
		}
        for (auto& obj : person_boxes) {


            Pose::Person res_tmp;
            cv::Rect inputbox;
            YoloBox2cvrect(obj, inputbox);
            res_tmp.body_box = inputbox;
            res_tmp.body_area = inputbox.width * inputbox.height; //todo multiperson
            vector<Point3f> body_keys;
            if (yolo_pose_find_person) {
                body_keys = body_keys_yolopose[0];
            }
            else
            {
                body_keys = engine_body->commit(make_tuple(frame, inputbox)).get()[0];
            }
            
            if (body_keys.size() == 0) {

                kalman_filter_b->resetFilter();
                res.push_back(res_tmp);
                continue;
            }
            if (smooth)
            {
                vector<Point3f> mean_body_keypoints(body_keys.size());
                for (int i = 0; i < mean_body_keypoints.size(); i++)
                {
                    mean_body_keypoints[i].x = 0;
                    mean_body_keypoints[i].y = 0;
                    mean_body_keypoints[i].z = 0;
                }
                smooth_label(mean_body_keypoints, smooth_cache_b, body_keys, inputbox.height, cache_size, moumentent_thresh);
                res_tmp.body_keypoints = mean_body_keypoints;
            }
            else if (kalman) {
                kalman_filter_b->predict();
                kalman_filter_b->correct(body_keys);
                res_tmp.body_keypoints = body_keys;
            }
            else
            {
                res_tmp.body_keypoints = body_keys;
            }

            if (!hand_rec_active) {
                continue;
            }
            int single;//single : -1->none,0->all, 1->right,2->left
            
            Yolo::BoxArray hand_boxes = find_hand_box(filterBoxesByClassLabel(raw_boxes, hand_label, confidence_thresh_hand), body_keys, single, confidence_thresh_hand);

            //开始处理hand
            if (single == -1) {
                r_hand_empty_count += 1;
                l_hand_empty_count += 1;
                if (kalman)
                {
                    if (r_hand_empty_count > 2) {
                        r_hand_empty_count = 0;
                        kalman_filter_rh->resetFilter();
                    }
                    else
                    {
                        auto kptsr = kalman_filter_rh->predict();
                        res_tmp.R_hand_keypoints = kptsr;
                    }
                    if (l_hand_empty_count > 2) {
                        l_hand_empty_count = 0;
                        kalman_filter_lh->resetFilter();
                    }
                    else
                    {
                        auto kptsl = kalman_filter_lh->predict();
                        res_tmp.L_hand_keypoints = kptsl;
                    }

                }
                res.push_back(res_tmp);
                continue;

            }
            else
            {
                shared_future<vector<Point3f>> R_hand_keypoints;
                shared_future<vector<Point3f>> L_hand_keypoints;
                if (single == 0) {
                    r_hand_empty_count = 0;
                    l_hand_empty_count = 0;
                    cv::Rect inputbox_r;
                    YoloBox2cvrect(hand_boxes[1], inputbox_r);
                    R_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_r));

                    cv::Rect inputbox_l;
                    YoloBox2cvrect(hand_boxes[0], inputbox_l);
                    L_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_l));
                    res_tmp.R_hand_box = inputbox_r;
                    res_tmp.L_hand_box = inputbox_l;

                    if (smooth)
                    {
                        auto kptsr = R_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_R_hand_keypoints(kptsr.size());
                        for (int i = 0; i < mean_R_hand_keypoints.size(); i++)
                        {
                            mean_R_hand_keypoints[i].x = 0;
                            mean_R_hand_keypoints[i].y = 0;
                            mean_R_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_R_hand_keypoints, smooth_cache_rh, kptsr, inputbox_r.height, cache_size, moumentent_thresh);
                        res_tmp.R_hand_keypoints = mean_R_hand_keypoints;
                        auto kptsl = L_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_L_hand_keypoints(kptsl.size());
                        for (int i = 0; i < mean_L_hand_keypoints.size(); i++)
                        {
                            mean_L_hand_keypoints[i].x = 0;
                            mean_L_hand_keypoints[i].y = 0;
                            mean_L_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_L_hand_keypoints, smooth_cache_lh, kptsl, inputbox_l.height, cache_size, moumentent_thresh);
                        res_tmp.L_hand_keypoints = mean_L_hand_keypoints;
                    }
                    else if (kalman) {
                        
                        kalman_filter_lh->predict();
                        kalman_filter_rh->predict();
                        auto kptsl = L_hand_keypoints.get();
                        auto kptsr = R_hand_keypoints.get();
                        kalman_filter_lh->correct(kptsl);
                        kalman_filter_rh->correct(kptsr);
                        res_tmp.L_hand_keypoints = kptsl;
                        res_tmp.R_hand_keypoints = kptsr;
                    
                    }
                    else
                    {
                        res_tmp.R_hand_keypoints = R_hand_keypoints.get();
                        res_tmp.L_hand_keypoints = L_hand_keypoints.get();
                    }

                }
                if (single == 1 ) {
                    //right
                    r_hand_empty_count = 0;
                    l_hand_empty_count += 1;
                    cv::Rect inputbox_r;
                    YoloBox2cvrect(hand_boxes[0], inputbox_r);
                    R_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_r));
                    res_tmp.R_hand_box = inputbox_r;
                    if (smooth)
                    {
                        auto kptsr = R_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_R_hand_keypoints(kptsr.size());
                        for (int i = 0; i < mean_R_hand_keypoints.size(); i++)
                        {
                            mean_R_hand_keypoints[i].x = 0;
                            mean_R_hand_keypoints[i].y = 0;
                            mean_R_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_R_hand_keypoints, smooth_cache_rh, kptsr, inputbox.height, cache_size, moumentent_thresh);
                        res_tmp.R_hand_keypoints = mean_R_hand_keypoints;
                    }
                    else if (kalman) {
                        kalman_filter_rh->predict();
                        if (l_hand_empty_count > 3) {
                            l_hand_empty_count = 0;
                            kalman_filter_lh->resetFilter();
						}
                        else
                        {
                            auto kptsl = kalman_filter_lh->predict();
                            res_tmp.L_hand_keypoints = kptsl;
                        }
                        
                        auto kptsr = R_hand_keypoints.get();
                        kalman_filter_rh->correct(kptsr);
                        res_tmp.R_hand_keypoints = kptsr;
                        
                    }
                    else
                    {
                        res_tmp.R_hand_keypoints = R_hand_keypoints.get();
                    }
                }
                
                if (single == 2 ) {
                    //left
                    r_hand_empty_count += 1;
                    l_hand_empty_count = 0;
                    cv::Rect inputbox_l;
                    YoloBox2cvrect(hand_boxes[0], inputbox_l);
                    L_hand_keypoints = engine_hand->commit(make_tuple(frame, inputbox_l));
                    res_tmp.L_hand_box = inputbox_l;
                    
                    if (smooth)
                    {
                        auto kptsl = L_hand_keypoints.get();
                        std::vector<cv::Point3f> mean_L_hand_keypoints(kptsl.size());
                        for (int i = 0; i < mean_L_hand_keypoints.size(); i++)
                        {
                            mean_L_hand_keypoints[i].x = 0;
                            mean_L_hand_keypoints[i].y = 0;
                            mean_L_hand_keypoints[i].z = 0;
                        }
                        smooth_label(mean_L_hand_keypoints, smooth_cache_lh, kptsl, inputbox.height, cache_size, moumentent_thresh);
                        res_tmp.L_hand_keypoints = mean_L_hand_keypoints;
                    }
                    else if (kalman) {
                        kalman_filter_lh->predict();
                        auto kptsl = L_hand_keypoints.get();
                        kalman_filter_lh->correct(kptsl);

                        if (r_hand_empty_count >3) {
							r_hand_empty_count = 0;
							kalman_filter_rh->resetFilter();
						}
                        else
                        {
							auto kptsr = kalman_filter_rh->predict();
							res_tmp.R_hand_keypoints = kptsr;
						}
                        res_tmp.L_hand_keypoints = kptsl;
                        
                    }
                    else
                    {
                        res_tmp.L_hand_keypoints = L_hand_keypoints.get();
                    }
               

                }

            }


            lock.unlock();

            res.push_back(res_tmp);
        }

        for (auto& res_tmp : res)
        {
            switch (_show) {
            case hand_l_lp:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                break;
            case hand_r_lp:
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                break;
            case person_lp:
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body, 2);
                break;
            case hand_l_box:
                if (res_tmp.L_hand_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                break;
            case hand_r_box:
                if (res_tmp.R_hand_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                break;
            case person_box:
                if (res_tmp.body_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.body_box, Scalar(0, 255, 0), 2);
                break;
            case all_kp:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body, 2);
                break;
            case all_box:
                if (res_tmp.L_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.R_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.body_box.width != 0) {
                    cv::rectangle(frame, res_tmp.body_box, Scalar(255, 0, 0), 2);
                }
                break;
            case all:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body, 2);
                if (res_tmp.L_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.R_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                    Rect box_ = res_tmp.R_hand_box;
                    float rate = 0.1f;
                    float pad_width = box_.width * (1 + 2 * rate);
                    float pad_height = box_.height * (1 + 2 * rate);
                    float scale = min(256 / pad_width, 256 / pad_height);
                    float i2d[6], d2i[6];
                    i2d[0] = scale;  i2d[1] = 0;      i2d[2] = -(box_.x + box_.width * 0.5) * scale + 256 * 0.5;
                    i2d[3] = 0;      i2d[4] = scale;  i2d[5] = -(box_.y + box_.height * 0.5) * scale + 256 * 0.5;
                    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
                    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
                    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
                    cv::Mat imgshow;
                    cv::warpAffine(frame, imgshow, m2x3_i2d, cv::Size(256, 256), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
                    cv::imshow("hand", imgshow);
                    //d2i[0] 
                    //Rect r_o;
                    //r_o.x = M_inv[0, 0] * j + M_inv[0, 1] * k + M_inv[0, 2]
                }
                if (res_tmp.body_box.width != 0) {
                    cv::rectangle(frame, res_tmp.body_box, Scalar(255, 0, 0), 2);
                }
                break;
            case none:
                break;
            default:
                break;
            }
        }
    }
    

    void Combine_pose::Alphapose136_infer(cv::Mat& frame, Pose::Result& res) {
        if (frame.empty()) {
            INFOE("Frame is empty");
            return;
        }
        frame_w = frame.cols;
        frame_h = frame.rows;
        auto raw_boxes = engine_yolo->commit(frame).get();
        if (raw_boxes.size() == 0) {
            return;
        }
        //只保留boxes中class_label==0的元素
        int person_label = static_cast<int>(yolo_detect_label::person);
        Yolo::BoxArray person_boxes = filterBoxesByClassLabel(raw_boxes, person_label, confidence_thresh_body);//person conf 0.5
        if (person_boxes.size() > max_person_num) {
            std::sort(person_boxes.begin(), person_boxes.end(), SortBox);
            //保留前max_person_num个
            person_boxes.erase(person_boxes.begin() + max_person_num, person_boxes.end());
        }

        for (auto& obj : person_boxes)
        {
            Pose::Person res_tmp;
            cv::Rect inputbox;
            YoloBox2cvrect(obj, inputbox);
            res_tmp.body_box = inputbox;
            res_tmp.body_area = inputbox.width * inputbox.height; //todo multiperson
            vector<Point3f> whole_keys;
            whole_keys = engine_hand->commit(make_tuple(frame, inputbox)).get();

            if (whole_keys.size() == 0) {

                kalman_filter_b->resetFilter();
                res.push_back(res_tmp);
                continue;
            }

            // 0-25: body, 26-93: face, 94-114: left hand, 115-135: right hand
            for (int i = 94; i < 136; i++)
            {
                if (whole_keys[i].z < hand_thresh)
                {
					whole_keys[i].x = -1;
					whole_keys[i].y = -1;
					whole_keys[i].z = -1;
				}
            }

            
            if (smooth)
            {
                vector<Point3f> mean_body_keypoints(whole_keys.size());
                for (int i = 0; i < mean_body_keypoints.size(); i++)
                {
                    mean_body_keypoints[i].x = 0;
                    mean_body_keypoints[i].y = 0;
                    mean_body_keypoints[i].z = 0;
                }
                smooth_label(mean_body_keypoints, smooth_cache_b, whole_keys, inputbox.height, cache_size, moumentent_thresh);
            }
            else if (kalman) {
                kalman_filter_b->predict();
                kalman_filter_b->correct(whole_keys);
            }

            vector<Point3f> body_keys;
            vector<Point3f> L_hand_keypoints;
            vector<Point3f> R_hand_keypoints;
            vector<Point3f> face_keypoints;

            
            body_keys.insert(body_keys.end(), whole_keys.begin(), whole_keys.begin() + 26);
            face_keypoints.insert(face_keypoints.end(), whole_keys.begin() + 26, whole_keys.begin() + 94);
            L_hand_keypoints.insert(L_hand_keypoints.end(), whole_keys.begin() + 94, whole_keys.begin() + 115);
            R_hand_keypoints.insert(R_hand_keypoints.end(), whole_keys.begin() + 115, whole_keys.begin() + 136);
            res_tmp.body_keypoints = body_keys;
            res_tmp.L_hand_keypoints = L_hand_keypoints;
            res_tmp.R_hand_keypoints = R_hand_keypoints;
            res_tmp.face_keypoints = face_keypoints; // no use for now
            res.push_back(res_tmp);
            


        }
        for (auto& res_tmp : res)
        {
            switch (_show) {
            case hand_l_lp:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                break;
            case hand_r_lp:
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                break;
            case person_lp:
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body);
                break;
            case hand_l_box:
                if (res_tmp.L_hand_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                break;
            case hand_r_box:
                if (res_tmp.R_hand_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                break;
            case person_box:
                if (res_tmp.body_box.width == 0) {
                    break;
                }
                cv::rectangle(frame, res_tmp.body_box, Scalar(0, 255, 0), 2);
                break;
            case all_kp:
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body);
                break;
            case all_box:
                if (res_tmp.L_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.L_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.R_hand_box.width != 0) {
                    cv::rectangle(frame, res_tmp.R_hand_box, Scalar(0, 255, 0), 2);
                }
                if (res_tmp.body_box.width != 0) {
                    cv::rectangle(frame, res_tmp.body_box, Scalar(255, 0, 0), 2);
                }
                break;
            case all:


                if (res_tmp.body_box.width != 0) {
                    cv::rectangle(frame, res_tmp.body_box, Scalar(255, 0, 0), 2);
                }
                draw(frame, res_tmp.L_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.R_hand_keypoints, hand_kp_tuples, colors_hand);
                draw(frame, res_tmp.body_keypoints, bd_kp_tuples, colors_body);
                break;
            case none:
                break;
            default:
                break;
            }
        }
    }

    void Combine_pose::start() {
        INFO("start");
    }
};