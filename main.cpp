#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <iomanip>

#define FOLDER_LEFT "img/2011_09_26/2011_09_26_drive_0014_sync/image_02/data/" //Chemin relatif par rapport au fichier
#define FOLDER_RIGHT "img/2011_09_26/2011_09_26_drive_0014_sync/image_03/data/" //Chemin relatif par rapport au fichier
#define NB_FRAME 313

cv::Mat computeDisparity(cv::Mat &rectifiedL, cv::Mat &rectifiedR)
{
	double min, max;
	cv::Mat disparity = cv::Mat(rectifiedL.rows, rectifiedL.cols, CV_16S)
		, out = cv::Mat(rectifiedL.rows, rectifiedL.cols, CV_8UC1);
	cv::Ptr<cv::StereoBM> SBM = cv::StereoBM::create(0, 21);

	SBM->compute(rectifiedL, rectifiedR, disparity);
	cv::minMaxLoc(disparity, &min, &max);
	disparity.convertTo(out, CV_8UC1, 255 / (max - min));

	return out;
}

void callBackTrackBar(int pos, void* userdata)
{
	cv::Mat left, right, disparity;
	std::ostringstream oss;

	oss << std::setfill('0');
	oss.str("");
	oss << FOLDER_LEFT << std::setw(10) << pos << ".png";
	left = cv::imread(oss.str(), 0);
	oss.str("");
	oss << FOLDER_RIGHT << std::setw(10) << pos << ".png";
	right = cv::imread(oss.str(), 0);

	disparity = computeDisparity(left, right);
	cv::imshow("Window", disparity);
	cv::waitKey(1);
}

int main(void)
{
	cv::namedWindow("Window", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Window", NULL, NB_FRAME, callBackTrackBar);

	cv::waitKey();

	return 0;
}