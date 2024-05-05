#ifndef KALMAN_FILTER_2D_H
#define KALMAN_FILTER_2D_H

#include <opencv2/opencv.hpp>

class KalmanFilter2D {
public:
    KalmanFilter2D(int vector_length);
    void resetFilter();
    std::vector<cv::Point3f> predict();
    void correct(std::vector<cv::Point3f>& measurement);

private:
    cv::KalmanFilter kf_;
    int vector_length_;
    cv::Mat state_;
};

#endif // KALMAN_FILTER_2D_H