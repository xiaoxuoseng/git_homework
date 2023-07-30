#pragma once
#include "opencv2/opencv.hpp"

cv::Mat Stitch(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &map1, const cv::Mat &map2);
void myRemap(const cv::Mat& img, cv::Mat& dst, const cv::Mat& map1, const cv::Mat& map2, const int heigth, const int width);
void imgMergeMax(const cv::Mat & img1, const cv::Mat & img2, cv::Mat &dst);
void imgMergeOpticalFlow(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& dst);
void normaliziBrightness(cv::Mat& img1, const cv::Mat& img2);
