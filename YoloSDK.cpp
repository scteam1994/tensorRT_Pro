//#pragma once
//#include "YoloSDK.hpp"
//#include <common/preprocess_kernel.cuh>
//#include<algorithm>
//#include <random>
//#include <sstream>
//#include <filesystem>
//#include <builder/trt_builder.hpp>
//
//
//using namespace YOLOSDK;
//
//
//bool YoloObj::compile_yolo(const char* name) {
//    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {
//
//        INFO("Int8 %d / %d", current, count);
//
//        for (int i = 0; i < files.size(); ++i) {
//            auto image = cv::imread(files[i]);
//            Yolo::image_to_tensor(image, tensor, type, i);
//        }
//        };
//    string onnx_file = iLogger::format("%s.onnx", name);
//    string model_file = iLogger::format("%s.%s.trtmodel", name, TRT::mode_string(mode));
//    int test_batch_size = 16;
//    TRT::Mode mode = TRT::Mode::FP32;
//    if (not iLogger::exists(name)) {
//
//        if (not iLogger::exists(model_file)) {
//            bool ok = TRT::compile(
//                mode,            // FP32、FP16、INT8
//                test_batch_size,            // max batch size
//                onnx_file,                  // source
//                model_file,                // saveto
//                {},
//                int8process,
//                "inference"
//            );
//
//            if (!ok) return false;
//    }
//    
//    string onnx_file = iLogger::format("%s.onnx", name);
//    string model_file = iLogger::format("%s.%s.trtmodel", name, TRT::mode_string(mode));
//    int test_batch_size = 1;
//
//    if (not iLogger::exists(model_file)) {
//        bool ok = TRT::compile(
//            mode,            // FP32、FP16、INT8
//            test_batch_size,            // max batch size
//            onnx_file,                  // source
//            model_file                  // saveto
//        );
//
//        if (!ok) return false;
//    }
//
//    return true;
//}
//
//void Face_rec::create_face_bank(std::string img_folder) {
//    if (fs::is_directory(img_folder)) {
//        fs::directory_iterator begin(img_folder);
//        fs::directory_iterator end;
//        this->bank_size = std::distance(begin, end);
//        if (this->bank_size == 0) {
//            INFO("Face bank folder is empty");
//            return;
//        }
//        this->bank = (float*)malloc(sizeof(float) * feature_size * bank_size);
//
//        bank_size = 0;
//        for (const auto& entry : fs::directory_iterator(img_folder)) {
//            this->bank_id.push_back(uuid::generate_uuid_v4());
//            cout << "正在录入" << entry.path().filename().string() << " ID" << this->bank_id.back() << endl;
//            string img = entry.path().filename().string();
//            Mat img_mat = imread(img_folder + "/" + img);
//            if (img_mat.empty()) {
//                INFOE("图片%s无法读取。", img.c_str());
//                continue;
//            }
//            Arcface::feature embedding;
//            if (get_embedding(img_mat, embedding)) {
//                memcpy(bank + bank_size * feature_size, embedding.ptr<float>(0), sizeof(float) * feature_size);
//                bank_size++;
//            }
//            else {
//                INFOE("图片%s无法提取特征。", img.c_str());
//            }
//            INFO("已读取%d张图片。", bank_size);
//        }
//        if (bank_size != update_bank_size()) {
//            INFOW("Face bank size does not match. This may cause unknown error.Please rebuild face bank");
//            return;
//        }
//
//        checkCudaRuntime(cudaMalloc((void**)&bank_d, sizeof(float) * bank_size * feature_size));
//        checkCudaRuntime(cudaMemcpy(bank_d, bank, sizeof(float) * bank_size * feature_size, cudaMemcpyHostToDevice));
//
//    }
//    else {
//        INFO("Face bank folder is not a directory");
//    }
//
//
//    return;
//
//}
//
//void Face_rec::create_face_bank(vector<Mat> mat) {
//    this->bank_size = mat.size();
//    this->bank = (float*)malloc(sizeof(float) * feature_size * bank_size);
//    bank_size = 0;
//    for (int i = 0; i < mat.size(); ++i) {
//        this->bank_id.push_back(uuid::generate_uuid_v4());
//        cout << "正在录入" << this->bank_id.back() << endl;
//        Mat img_mat = mat[i];
//        if (img_mat.empty()) {
//            INFOE("图片无法读取。");
//            continue;
//        }
//        Arcface::feature embedding;
//        if (get_embedding(img_mat, embedding)) {
//            memcpy(bank + bank_size * feature_size, embedding.ptr<float>(0), sizeof(float) * feature_size);
//            bank_size++;
//        }
//        else {
//            INFOE("图片无法提取特征。");
//        }
//        INFO("已读取%d张图片。", bank_size);
//        if (bank_size != update_bank_size()) {
//            INFOW("Face bank size does not match. This may cause unknown error.Please rebuild face bank");
//            return;
//        }
//        checkCudaRuntime(cudaMalloc((void**)&bank_d, sizeof(float) * bank_size * feature_size));
//        checkCudaRuntime(cudaMemcpy(bank_d, bank, sizeof(float) * bank_size * feature_size, cudaMemcpyHostToDevice));
//
//    }
//
//    return;
//
//}
//
//void Face_rec::init(Face::Param& param) {
//    this->show_res = param.show_res;
//    TRT::set_device(param.device_id);
//    if (!compile_scrfd(param.img_width, param.img_height, param.name_scrfd))
//        return;
//    string model_file_scrfd = iLogger::format("%s.%dx%d.%s.trtmodel", param.name_scrfd, param.img_width, param.img_height, TRT::mode_string(TRT::Mode::FP32));
//    engine_scrfd = Scrfd::create_infer(model_file_scrfd, 0, 0.7);
//
//    if (engine_scrfd == nullptr) {
//        INFOE("Engine_scrfd is nullptr");
//        return;
//    }
//
//    if (!compile_arcface(param.name_arcface))
//        return;
//    string model_file_arcface = iLogger::format("%s.FP32.trtmodel", param.name_arcface);
//    engine_arcface = Arcface::create_infer(model_file_arcface, 0);
//
//    if (engine_arcface == nullptr) {
//        INFOE("Engine_arcface is nullptr");
//        return;
//    }
//
//
//    if (param.face_folder != "") {
//        create_face_bank(param.face_folder);
//    }
//    else if (param.face_vector.size() != 0) {
//        create_face_bank(param.face_vector);
//    }
//    else {
//        INFO("Face bank is empty, Please create face bank before use");
//    }
//
//}
//
//void Face_rec::register_face(cv::Mat& img, string& ID) {
//    if (img.empty()) {
//        INFOE("Frame is empty");
//        return;
//    }
//    auto faces = engine_scrfd->commit(img).get();
//    int person_num = 0;
//    for (int i = 0; i < faces.size(); ++i) {
//        Mat crop;
//        auto face = faces[i];
//        tie(crop, face) = Scrfd::crop_face_and_landmark(img, face);
//
//        Arcface::landmarks landmarks;
//        memcpy(landmarks.points, face.landmark, sizeof(landmarks.points));
//        Arcface::feature out = engine_arcface->commit(make_tuple(crop, landmarks)).get();
//        if (out.empty()) {
//            INFOE("Feature is empty");
//            return;
//        }
//        if (ID == nullptr) {
//
//            string id = uuid::generate_uuid_v4();
//            INFO("未指定ID，自动生成ID：%s", id.c_str());
//            this->bank_id.push_back(id);
//            this->bank = (float*)realloc(bank, sizeof(float) * feature_size * bank_size);
//            memcpy(bank + bank_size * feature_size, out.ptr<float>(0), sizeof(float) * feature_size);
//            cout << "已录入" << id << endl;
//            ID = id;
//
//        }
//        else {
//            this->bank_id.push_back(ID);
//            this->bank = (float*)realloc(bank, sizeof(float) * feature_size * bank_size);
//            memcpy(bank + bank_size * feature_size, out.ptr<float>(0), sizeof(float) * feature_size);
//            cout << "已录入" << ID << endl;
//
//        }
//        bank_size++;
//    }
//    if (bank_size != update_bank_size()) {
//        INFOW("Face bank size does not match. This may cause unknown error.Please rebuild face bank");
//        return;
//    }
//    bank_d = nullptr;
//    checkCudaRuntime(cudaMalloc((void**)&bank_d, sizeof(float) * bank_size * feature_size));
//    checkCudaRuntime(cudaMemcpy(bank_d, bank, sizeof(float) * bank_size * feature_size, cudaMemcpyHostToDevice));
//
//
//    return;
//}
//
//void Face_rec::compare_face_bank_gpu(Arcface::feature& out, float* res) {
//    if (bank_d == nullptr) {
//        INFOE("Face bank is empty, Please create face bank before use");
//        return;
//    }
//    // cv::Mat to float*
//    float* output = out.ptr<float>(0);
//    float* res_d = nullptr;
//    float* output_d = nullptr;
//    checkCudaRuntime(cudaMalloc((void**)&output_d, sizeof(float) * feature_size));
//    checkCudaRuntime(cudaMalloc((void**)&res_d, sizeof(float) * bank_size));
//    checkCudaRuntime(cudaMemcpy(res_d, res, sizeof(float) * bank_size, cudaMemcpyHostToDevice));
//    checkCudaRuntime(cudaMemcpy(output_d, output, sizeof(float) * feature_size, cudaMemcpyHostToDevice));
//
//    compare_face(output_d, bank_d, res_d, bank_size, feature_size);
//    checkCudaRuntime(cudaMemcpy(res, res_d, sizeof(float) * bank_size, cudaMemcpyDeviceToHost));
//
//}
//
//
//
//void Face_rec::combine_infer(cv::Mat& frame, Face::Result& res) {
//    if (frame.empty()) {
//        INFOE("Frame is empty");
//        return;
//    }
//    if (bank_size == 0) {
//        INFOE("Face bank is empty, Please create face bank before use");
//        return;
//    }
//    auto faces = engine_scrfd->commit(frame).get();
//    int person_num = 0;
//    for (int i = 0; i < faces.size(); ++i) {
//        Mat crop;
//        auto face = faces[i];
//        tie(crop, face) = Scrfd::crop_face_and_landmark(frame, face);
//
//        Arcface::landmarks landmarks;
//        memcpy(landmarks.points, face.landmark, sizeof(landmarks.points));
//        Arcface::feature out = engine_arcface->commit(make_tuple(crop, landmarks)).get();
//        float* distances = (float*)malloc(sizeof(float) * bank_size);
//        compare_face_bank_gpu(out, distances);
//        if (distances == nullptr) {
//            INFOE("distances compare failed");
//            return;
//        }
//        int minPosition = min_element(distances, distances + bank_size) - distances;
//        Face::Person p;
//        p.ID = this->bank_id[minPosition];
//        p.face = face;
//        p.landmarks = landmarks;
//        p.embedding = out;
//        p.distance = distances[minPosition];
//        res.push_back(p);
//        if (show_res) {
//            cv::rectangle(frame, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), cv::Scalar(0, 255, 0), 2, 8, 0);
//            cv::putText(frame, p.ID, cv::Point(face.left, face.top), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
//            cv::putText(frame, to_string(p.distance), cv::Point(face.left, face.bottom), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
//        }
//    }
//
//}
//
//bool Face_rec::compare_face_pair(Mat& img1, Mat& img2, float& dist) {
//    Arcface::feature emb1, emb2;
//    if (get_embedding(ref(img1), ref(emb1)) && get_embedding(ref(img2), ref(emb2))) {
//        for (int i = 0; i < feature_size; ++i) {
//            dist += (emb1.ptr<float>(0)[i] - emb2.ptr<float>(0)[i]) * (emb1.ptr<float>(0)[i] - emb2.ptr<float>(0)[i]);
//        }
//        return true;
//    }
//    else
//    {
//        return false;
//    }
//}
//
//bool Face_rec::get_embedding(Mat& img, Arcface::feature& out) {
//    Arcface::landmarks landmarks;
//
//    auto faces = engine_scrfd->commit(img).get();
//
//    if (faces.size() == 0) {
//        INFOE("No face detected");
//        return false;
//    }
//    if (faces.size() > 1) {
//        INFOE("More than one face detected");
//        return false;
//    }
//    Mat crop;
//    auto face = faces[0];
//    tie(crop, face) = Scrfd::crop_face_and_landmark(img, face);
//    memcpy(landmarks.points, face.landmark, sizeof(landmarks.points));
//    out = engine_arcface->commit(make_tuple(crop, landmarks)).get();
//    return true;
//}
//
//bool Face_rec::detect_face(Mat& img, FaceDetector::BoxArray& res) {
//    auto box = engine_scrfd->commit(img).get();
//    res = box;
//    if (res.size() == 0) {
//        INFOE("No face detected");
//        return false;
//    }
//
//    return true;
//}
//
//bool Face_rec::reset(Face::Param& param) {
//    if (bank != nullptr) {
//        free(bank);
//        bank = nullptr;
//    }
//    bank_size = 0;
//    bank_id.clear();
//    if (engine_scrfd != nullptr) {
//        engine_scrfd.reset();
//    }
//    if (engine_arcface != nullptr) {
//        engine_arcface.reset();
//    }
//
//    Face_rec::init(param);
//    return true;
//}