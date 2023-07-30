#include <iostream>
#include <string>
#include <map>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int batch = 1, channel = 3, cols = 288, rows = 512;

void get_input(float * fg_input, float * bg_input, float * mask)
{
	string fg_path = "../data/jump.jpg";
	string bg_path = "../data/seaside.jpg";
	string mask_path = "../data/mask.jpg";
	cv::Mat mask_gray;

	cv::Mat fg_raw = cv::imread(fg_path);
	cv::Mat bg_raw = cv::imread(bg_path);
	cv::Mat mask_raw = cv::imread(mask_path);

	cv::cvtColor(mask_raw, mask_gray, cv::COLOR_BGR2GRAY);

	for (int c = 0; c < channel; ++c) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				fg_input[c * rows * cols + i * cols + j] = static_cast<float>((fg_raw.at<cv::Vec3b>(i, j))[c]) / 255;
				bg_input[c * rows * cols + i * cols + j] = static_cast<float>((bg_raw.at<cv::Vec3b>(i, j))[c]) / 255;
				if (c == 1) mask[i * cols + j] = static_cast<float>(mask_gray.at<uchar>(i, j)) / 255;
			}	
		}
	}
}

void convertResult2CV(MNN::Tensor* output, cv::Mat &dst)
{
	float* foutput = (float*)malloc(sizeof(float) * batch * channel * cols * rows);
	cv::Mat cvdst(rows, cols, CV_8UC3);

	float* ptr = output->host<float>();
	for (int i = 0; i < output->elementSize(); i++) {
		foutput[i] = ptr[i];
	}

	for (int c = 0; c < channel; ++c) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				cvdst.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(foutput[c * rows * cols + i * cols + j] * 255);
			}
		}
	}
	dst = cvdst;
}

int main()
{
	cv::Mat cvOutput;
	shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile("../model/picture_fusion.FP32.mnn"));
    MNN::ScheduleConfig config;
	config.type = MNNForwardType::MNN_FORWARD_CPU;
	config.numThread = 4;
	auto session = net->createSession(config);

	map<string, MNN::Tensor*> tensor_input = net->getSessionInputAll(session);
	float* fg_input = (float*)malloc(sizeof(float) * batch * channel * rows * cols);
	float* bg_input = (float*)malloc(sizeof(float) * batch * channel * rows * cols);
	float* mask_input = (float*)malloc(sizeof(float) * batch * rows * cols);
	get_input(fg_input, bg_input, mask_input);

	std::vector<int> dims = {channel, rows, cols};
	std::vector<int> mask_dims = { batch, rows, cols };
	MNN::Tensor* fg_tensor = MNN::Tensor::create<float>(dims, fg_input, MNN::Tensor::CAFFE);
	MNN::Tensor* bg_tensor = MNN::Tensor::create<float>(dims, bg_input, MNN::Tensor::CAFFE);
	MNN::Tensor* mask_tensor = MNN::Tensor::create<float>(mask_dims, mask_input, MNN::Tensor::CAFFE);

	net->resizeTensor(tensor_input["fg_input"], dims);
    net->resizeTensor(tensor_input["bg_input"], dims);
    net->resizeTensor(tensor_input["alpha_input"], mask_dims);
	net->resizeSession(session);

	tensor_input["fg_input"]->copyFromHostTensor(fg_tensor);
	tensor_input["bg_input"]->copyFromHostTensor(bg_tensor);
	tensor_input["alpha_input"]->copyFromHostTensor(mask_tensor);

	MNN::ErrorCode error = net->runSession(session);

	cv::Mat dst;
	auto output = net->getSessionOutput(session, NULL);
    std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, output->getDimensionType())); //nchw
    output->copyToHostTensor(outputUser.get());

	convertResult2CV(output, dst);
	cv::imwrite("output.jpg", dst);
	free(fg_input);
	free(bg_input);
	free(mask_input);
}

