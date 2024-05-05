#pragma once
#ifndef FACE_SDK_HPP
#define FACE_SDK_HPP
#include "global_export.h"
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "application/app_arcface/arcface.hpp"
#include "application/app_scrfd/scrfd.hpp"
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
namespace Face {
    void compare_face(const float* feature1, const float* bank, float* res, int bank_size, int feature_size);
    struct TRT_EXPORT Param {
        int device_id;
        int img_width;
        int img_height;
        const char* name_arcface;
        const char* name_scrfd;
        string face_folder;
        vector<Mat> face_vector;
        bool show_res;
        Param() {
			device_id = 0;
			img_width = 640;
			img_height = 640;
			name_arcface = "arcface_iresnet50";
			name_scrfd = "scrfd_2.5g_bnkps";
			face_folder = "face_bank";
            show_res = false;
		}

    };
    struct TRT_EXPORT Person {
        string ID;
        FaceDetector::Box face;
        Arcface::landmarks landmarks;
        Arcface::feature embedding;
        float distance;
    };
    typedef vector<Person> Result;

    class TRT_EXPORT Face_rec {
    private:
        shared_ptr<Scrfd::Infer> engine_scrfd;
        shared_ptr<Arcface::Infer> engine_arcface;
        
        float* bank = nullptr;
        float* bank_d = nullptr;
        


    public:
        int feature_size = 512;
        bool show_res = false;
        int bank_size;
        vector<string> bank_id;
        int update_bank_size() {
            bank_size = bank_id.size();
            return bank_size;
        }

        static bool compile_scrfd(int input_width, int input_height, const char* name);
        static bool compile_arcface(const char* name);

        // init engine

        void init(Face::Param& param);
        void register_face(cv::Mat& img, string& ID);
        void combine_infer(cv::Mat& frame, Face::Result& res);
        bool reset(Face::Param& param);


        // standalone functions
        // create face bank input is a folder Path of images
        void create_face_bank(std::string img_folder);
        // create face bank input is a vector of images mats
        void create_face_bank(vector<Mat> mat);
        // compare face with face bank, need to call init first and then call init create_face_bank
        void compare_face_bank_gpu(Arcface::feature& out, float* res); 
        // compare face pair, no need to call create_face_bank, but need to call init
        bool compare_face_pair(Mat& img1, Mat& img2, float& dist);
        // get embedding of a single face, no need to call create_face_bank, but need to call init
        bool get_embedding(Mat& img, Arcface::feature& out);
        // detect face, no need to call create_face_bank, but need to call init
        bool detect_face(Mat& img, FaceDetector::BoxArray& faces);

        
    };
}