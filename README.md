# REFERENCE
原项目https://github.com/shouxieai/tensorRT_Pro

# WORK
1. 合并并打包了yolo和alphapose
2. 合并并打包了scrfd和arcface
3. 使用cuda实现了一个人脸库对比的函数
4. 原本lean中的依赖移动到3rdparty中
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

