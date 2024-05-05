#include "kalman_kpt.hpp"
void mat2vec(const cv::Mat& mat, std::vector<cv::Point3f>& vec) {
    int n = 0;
    for (auto& i : vec) 
    {
        i.x =  mat.at<float>(n * 3, 0);
        i.y =  mat.at<float>(n * 3 + 1, 0);
        i.z =  mat.at<float>(n * 3 + 2, 0);
        n++;
    }
}
KalmanFilter2D::KalmanFilter2D(int vector_length) : vector_length_(vector_length) {
    kf_ = cv::KalmanFilter(vector_length * 3, vector_length * 3, 0);
    state_ = cv::Mat::zeros(vector_length * 3, 1, CV_32F);
    cv::randn(state_, cv::Scalar::all(0), cv::Scalar::all(0.1));
    kf_.statePost = state_;

    //kf_.statePost = state_;
    cv::setIdentity(kf_.transitionMatrix);

    // Initialize measurement matrix H
    cv::setIdentity(kf_.measurementMatrix);

    // Initialize process noise covariance matrix Q
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-5));

    // Initialize measurement noise covariance matrix R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));

    // Initialize error covariance matrix P
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
}
std::vector<cv::Point3f> KalmanFilter2D::predict() {
    cv::Mat prediction = kf_.predict();
    std::vector<cv::Point3f> measurement(vector_length_);
    mat2vec(prediction, measurement);
    return measurement;
}
void KalmanFilter2D::resetFilter() {
    cv::randn(state_, cv::Scalar::all(0), cv::Scalar::all(0.1));
    kf_.statePost = state_;
}

void KalmanFilter2D::correct(std::vector<cv::Point3f>& measurement) {
    cv::Mat measurement_mat(vector_length_ * 3, 1, CV_32F);
    for (int i = 0; i < vector_length_; ++i) {
        measurement_mat.at<float>(i * 3, 0) = measurement[i].x;
        measurement_mat.at<float>(i * 3 + 1, 0) = measurement[i].y;
        measurement_mat.at<float>(i * 3 + 2, 0) = measurement[i].z;
    }
    auto p = kf_.correct(measurement_mat);
    mat2vec(p, measurement);
    
}

