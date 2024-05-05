# REFERENCE
原项目https://github.com/shouxieai/tensorRT_Pro

# WORK
1. 合并并打包了yolo和alphapose
2. 合并并打包了scrfd和arcface
3. 使用cuda实现了一个人脸库对比的函数
4. 增加了yolov8-pose和alphapose+DCN的支持，增加了一个mmdet的DCN的cuda实现
5. 目前只实现了windows visual studio 编译


## face sdk
api
```c++
    void init(Face::Param& param);  // 初始化
    void register_face(cv::Mat& img, string& ID); // 添加单个人脸到人脸库
    void combine_infer(cv::Mat& frame, Face::Result& res); // 检测人脸并返回结果
    bool reset(Face::Param& param);// 重置

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

    // 中英结合，凑合看吧
```
## pose sdk
不算个sdk，只是个demo，从这里开始新学的windows编译动态库，好在之前搞过linux的，所以没啥问题，不过还是花了不少时间，不过总算搞定了，以后应该不会这么麻烦了。
```c++

int combine_infer_main() {
    // single thread infer
    combine_infer();
    
    // multi thread infer

    combine_pose combine;
    combine.init();
    combine.combine_infer();

    return 0;
}
```

在src/combine_pose.cpp中示例了如何多线程和单线程进行推理，多线程demo中尝试了流水线异步infer，但是不知道为什么没有效果，添加了异步之后yolo的infer时间变得很长，总fps没有什么变化，不知道是不是和机器有关还是说和原项目中的 class InferImpl 中的机制有关，如果有人能运行起来这个项目，麻烦告诉我一下，谢谢。
我已经试过取消infer_controller.hpp中的std::unique_lock<std::mutex> l(jobs_lock_);，没有效果。
