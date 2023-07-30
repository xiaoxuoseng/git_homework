#include "stitch.h"
using namespace std;

cv::Mat Stitch(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &map1, const cv::Mat &map2) {
    int height_out = 2880;
    int width_out = height_out * 2;
    cv::Mat output(height_out, width_out, CV_8UC3);
    output.setTo(0);

    // TODO
    cv::Mat remap_img1(height_out, width_out, CV_8UC3); 
    cv::Mat remap_img2(height_out, width_out, CV_8UC3);
    std::vector<cv::Mat> split_maps;
    cv::split(map1, split_maps);
    myRemap(img1, remap_img1, split_maps[0], split_maps[1], height_out, width_out);
    cv::imwrite("remap_img1.jpg", remap_img1);

    split_maps.clear();
    cv::split(map2, split_maps);
    myRemap(img2, remap_img2, split_maps[0], split_maps[1], height_out, width_out);
    cv::imwrite("remap_img2.jpg", remap_img2);

    //imgMergeMax(remap_img1, remap_img2, output);
    imgMergeOpticalFlow(remap_img1, remap_img2, output);
    return output;
}

void myRemap(const cv::Mat& img, cv::Mat& dst, const cv::Mat& map1, const cv::Mat& map2, const int heigth, const int width)
{
    cout << "Entry my remap" << endl;
    cv::Mat res = -1 * cv::Mat::ones(cv::Size(width, heigth), CV_8UC3);
    cv::Mat resize_map1 = cv::Mat::zeros(cv::Size(width, heigth), CV_8UC3);
    cv::Mat resize_map2 = cv::Mat::zeros(cv::Size(width, heigth), CV_8UC3);
    cv::resize(map1, resize_map1, cv::Size(width, heigth)); // , 0, 0, cv::INTER_CUBIC
    cv::resize(map2, resize_map2, cv::Size(width, heigth));
    for (int i = 0; i < heigth; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int u = (int)(resize_map1.at<float>(i, j) * img.rows);
            int v = (int)(resize_map2.at<float>(i, j) * img.cols);
            if (u < img.rows && v < img.cols && u > 0 && v > 0)
            {
                res.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(v, u);
                //cout << img.at<cv::Vec3b>(v, u) << endl;
            }    
        }
    }
    dst = res;
}

void imgMergeMax(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& dst)
{
    cv::Mat grey1, grey2;
    cv::Vec3b v_zero(0, 0, 0);
    cv::cvtColor(img1, grey1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, grey2, cv::COLOR_BGR2GRAY);
    for (int i = 0; i < img1.rows; i++)
    {
        for (int j = 0; j < img1.cols; j++)
        {
            cv::Vec3b pixel =  grey1.at<uint8_t>(i, j) > grey2.at<uint8_t>(i, j) ? img1.at<cv::Vec3b>(i, j) : img2.at<cv::Vec3b>(i, j);
            dst.at<cv::Vec3b>(i, j) = pixel;
        }
    }
}

void imgMergeOpticalFlow(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& dst)
{
    double cut_rate = 1.2 / 5.0;
    int x1 = cut_rate * img1.cols, x2 = (1 - cut_rate) * img1.cols;

    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img1_gray, corners, 500, 0.01, 10);

    std::vector<cv::Point2f> corners2;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1_gray, img2_gray, corners, corners2, status, err);

    cv::Mat transform = cv::estimateRigidTransform(corners, corners2, false);
    cv::Mat img2_aligned;
    cv::warpAffine(img2, img2_aligned, transform, img2.size());

    cv::Mat temp;
    cv::hconcat(img1(cv::Range::all(), cv::Range(0, x1)), img2_aligned(cv::Range::all(), cv::Range(x1, img1.cols)), temp);
    cv::hconcat(temp(cv::Range::all(), cv::Range(0, x2)), img1(cv::Range::all(), cv::Range(x2, img1.cols)), dst);
    //imgMergeMax(img1, img2_aligned, dst);
}

void normaliziBrightness(cv::Mat& img1, const cv::Mat& img2)
{
    cv::Scalar mean1 = cv::mean(img1);
    cv::Scalar mean2 = cv::mean(img2);
    cv::Vec3s diff;
    for (int i = 0; i < 3; i++)
    {
        diff[i] = static_cast<char>(mean2[i] - mean1[i]);
        //cout << diff[i] << mean2[i] - mean1[i] << endl;
    }

    //cout << diff[1] << "\t" << mean2[1] - mean1[1] << endl;
    for (int i = 0; i < img1.rows; i++)
    {
        for (int j = 0; j < img1.cols; j++)
        {
            //cout << img1.at<cv::Vec3b>(i, j) << endl;
            for (int c = 0; c < 3; c++)
            {
                cout << static_cast<char>(img1.at<cv::Vec3b>(i, j)[c]) << "\t" << diff[c] << endl;
                char sub_diff = img1.at<cv::Vec3b>(i, j)[c] + diff[c];
                cout << sub_diff << endl;
                if (sub_diff < 0)
                    img1.at<cv::Vec3b>(i, j)[c] = 0;
                else
                    img1.at<cv::Vec3b>(i, j)[c] = sub_diff;
            } 
           // cout << img1.at<cv::Vec3b>(i, j) << endl;
        }
    }
}

