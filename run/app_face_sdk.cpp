#include "face_sdk.hpp"
#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#include <iostream>
#if defined(_WIN32)
#include <Windows.h>
#include <wingdi.h>
#include <Shlwapi.h>
#pragma comment(lib, "shlwapi.lib")  
#endif

using namespace std;
using namespace cv;
int app_face_sdk() {

    Face::Param param;
    //param.device_id = 0;
    //param.img_width = 640;
    //param.img_height = 640;
    //param.name_arcface = "arcface_iresnet50";
    //param.name_scrfd = "scrfd_2.5g_bnkps";
    //param.face_folder = "face_bank";
    Face::Face_rec face_rec;
    face_rec.init(param);
    Mat img = imread("inference/yq.jpg");
    vector<Face::Person> res;
    // res list 中每个元素是一个人脸，包含人脸的id，人脸的位置，人脸的特征向量，人脸的距离
    // 图片中可能有多个人脸，每个人脸都会在res中有一个对应的元素
    face_rec.combine_infer(ref(img), ref(res));
    cout << "res_size:" << res.size() << endl;
    for (int i = 0; i < res.size(); i++) {
        cout << "ID:" << res[i].ID << endl;
        cout << "distance:" << res[i].distance << endl;
        cout << "face:" << res[i].face.left << res[i].face.top << res[i].face.right << res[i].face.bottom << endl;
        //cout << "embedding" << res[i].embedding<< endl;
    }
    //string ID = "liudehua";
    string ID;
    face_rec.register_face(img,ID);
    cout << "ID:" <<ID << endl;
    cout << "bank_size:" << face_rec.update_bank_size() << endl;
    return 0;
}
