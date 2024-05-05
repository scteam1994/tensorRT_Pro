#pragma once
#ifndef POSE_SDK_HPP
#define POSE_SDK_HPP
#include "global_export.h"
#include <stdio.h>
#include <string.h>
#include <functional>
#include "application/app_handpose/handpose.hpp"
#include "application/app_yolo/yolo.hpp"
#include "application/app_yolo_pose/yolo_pose.hpp"
#include "application/app_alphapose/alpha_pose.hpp"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "application/tools/kalman_kpt.hpp"
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

    const std::vector<std::tuple<int, int>> hand_kp_tuples = { {0, 1},
                                        {1, 2},
                                        {2, 3},
                                        {3, 4},

                                        {0, 5},
                                        {5, 6},
                                        {6, 7},
                                        {7, 8},

                                        {0, 9},
                                        {9, 10},
                                        {10, 11},
                                        {11, 12},

                                        {0, 13},
                                        {13, 14},
                                        {14, 15},
                                        {15, 16},

                                        {0, 17},
                                        {17, 18},
                                        {18, 19},
                                        {19, 20} };

    const std::vector<std::tuple<int, int>> bd_kp_tuples = { {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {5, 11}, {7, 9}, {6, 8}, {6, 12}, {8, 10}, {12, 14}, {14, 16}, {11, 13}, {13, 15} };

    const std::vector<std::tuple<int, int>> bd_kp_tuples_alpha = { {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {5, 11}, {7, 9},
        {6, 8}, {6, 12}, {8, 10},
        {12, 14}, {14, 16}, {11, 13}, {13, 15},
        {12,19},{11,19},
        {16,25},{25,21},{25,23},
		{15,24},{24,20},{24,22}
    };
    enum show_part {
        hand_l_lp = 0,
        hand_r_lp,
        person_lp,
        hand_l_box,
        hand_r_box,
        person_box,
        all_kp,
        all_box,
        all,
        none=100
    };

    enum infer_backend {
		AlphaPose136 = 0,
		HandBodyPose = 1
	};
    struct Param {
        int device_id;//gpu id
        const char* name_handpose;
        const char* name_yolo;
        const char* name_bodypose;
        int max_person_num;
        int mode;
        float box_scale = 1.0f;//暂时废弃，改在cuda kernel中写死
        bool smooth;
        bool kalman;//仅单人
        int cache_size;//smooth使用
        float moumentent_thresh;//smooth使用
        float dist_thresh;//找双手时使用
        int body_pkt_num = 17;//绑定模型
        int hand_pkt_num = 21;//绑定模型
        show_part code_show;
        int hand_rec_active;//0-> 不手势识别 1->右手识别 2->左手识别,3->双手识别
        Param() {
            device_id = 0;
			name_handpose = "hand-pose-res50-256";
			name_yolo = "yolov5s-hand-person-0.979";
            name_bodypose = "yolov8n-pose";
            max_person_num = 1; //hot
            smooth = false; //hot
            kalman = false; //hot
            cache_size = 10; //hot
            moumentent_thresh = 0.5; //hot
            dist_thresh = 0.4; //hot
            code_show = all_kp; //hot
            hand_rec_active = 3; //hot
            mode = 1;
        }

    };

    enum yolo_detect_label : int {
        hand = 1,
        person = 0
    };//yolo model label dict
    struct Person {
        vector<Point3f> L_hand_keypoints;
        vector<Point3f> R_hand_keypoints;
        vector<Point3f> face_keypoints;
        Rect L_hand_box;
        Rect R_hand_box;
        vector<Point3f> body_keypoints;
        Rect body_box;
        int body_area;
    };
    typedef vector<Person> Result;
    class TRT_EXPORT Combine_pose {
    private:

        shared_ptr<Yolo::Infer> engine_yolo;
        shared_ptr<AlphaPose::Infer> engine_hand;
        shared_ptr<YoloPose::Infer> engine_body;
        //param
        int max_person_num;
        float box_scale = 1.0f;//暂时废弃，改在cuda kernel中写死
        bool smooth = false;
        bool kalman = false;//仅单人
        int cache_size = 10;//smooth使用
        float moumentent_thresh = 0.5;//smooth使用
        float dist_thresh = 0.4;//找双手时使用
        float confidence_thresh_body = 0.5;
        float confidence_thresh_hand = 0.1;
        float hand_thresh = 0.1;
        int body_pkt_num = 17;//绑定模型
        int hand_pkt_num = 21;//绑定模型
        show_part _show;
        int hand_rec_active = 0;
        std::vector<cv::Scalar> colors_hand;
        std::vector<cv::Scalar> colors_body;
        int r_hand_empty_count = 0;
        int l_hand_empty_count = 0;
        void YoloBox2cvrect(Yolo::Box& src, cv::Rect& dst);
        Yolo::BoxArray find_hand_box(const Yolo::BoxArray& raw_boxes, const vector<Point3f> person, int& single, float confidence_thresh);
        std::vector<std::vector<cv::Point3f>> smooth_cache_lh;
        std::vector<std::vector<cv::Point3f>> smooth_cache_rh;
        std::vector<std::vector<cv::Point3f>> smooth_cache_b;
        KalmanFilter2D* kalman_filter_lh;
        KalmanFilter2D* kalman_filter_rh;
        KalmanFilter2D* kalman_filter_b;
        void init(Param param);
    public:
        Combine_pose(Param param) {
            init(param);
        };
        ~Combine_pose() {
            delete kalman_filter_lh;
            delete kalman_filter_rh;
            delete kalman_filter_b;
        };
        
        void change_param(Param param);
        void set_show(bool show) { if (show) _show = show_part::all; else { _show = show_part::none; }; };
        void get_show(bool& show) { if (_show != show_part::none) show = true; else show = false; };
        void set_max_person_num(int num) { this->max_person_num = num; };
        void combine_infer_hand_depreacated(cv::Mat& img, Result &res);
        void combine_infer_hand(cv::Mat& img, Result& res); 
        void Alphapose136_infer(cv::Mat& img, Result& res);
        void start();
        void smooth_label(std::vector<cv::Point3f>& mean_keypoints, std::vector<std::vector<cv::Point3f>>& smooth_cache, vector<Point3f> _keypoints, int height, int cache_size, float moumentent_thresh);
    };

}