#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

#define FOLDER_LEFT "img/2011_09_26/2011_09_26_drive_0014_sync/image_02/data/" //Chemin relatif par rapport au fichier
#define FOLDER_RIGHT "img/2011_09_26/2011_09_26_drive_0014_sync/image_03/data/" //Chemin relatif par rapport au fichier
#define NB_FRAME 314

cv::Mat computeDisparity(cv::Mat &rectifiedL, cv::Mat &rectifiedR)
{
	cv::Ptr<cv::StereoBM> SBM = cv::StereoBM::create();
	SBM->setBlockSize(21);
	SBM->setNumDisparities(112);
	SBM->setPreFilterSize(5);
	SBM->setPreFilterCap(1);
	SBM->setMinDisparity(0);
	SBM->setTextureThreshold(5);
	SBM->setUniquenessRatio(5);
	SBM->setSpeckleWindowSize(0);
	SBM->setSpeckleRange(20);
	SBM->setDisp12MaxDiff(64);

	cv::Mat disparity, out;

	SBM->compute(rectifiedL, rectifiedR, disparity);
	cv::normalize(disparity, out, 0.1, 255, CV_MINMAX, CV_8U);

	return out;
}

int main(void)
{
	cv::Mat left, right, disparity;
	std::ostringstream oss;

	oss << std::setfill('0');
	cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
	for (int i = 0; i < NB_FRAME; ++i)
	{
		oss.str("");
		oss << FOLDER_LEFT << std::setw(10) << i << ".png";
		left = cv::imread(oss.str(), 0);
		oss.str("");
		oss << FOLDER_RIGHT << std::setw(10) << i << ".png";
		right = cv::imread(oss.str(), 0);

		disparity = computeDisparity(left, right);
		cv::imshow("Disparity", disparity);
		cv::waitKey(25);
	}
	return 0;
}