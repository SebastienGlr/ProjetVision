#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <iomanip>

#define FOLDER_LEFT "img/2011_09_26/2011_09_26_drive_0014_sync/image_02/data/" //Left Camera
#define FOLDER_RIGHT "img/2011_09_26/2011_09_26_drive_0014_sync/image_03/data/" //Right Camera
#define NB_FRAME 313

std::string getImagePath(const char* folder, int ind)
{
	std::ostringstream oss;

	oss << std::setfill('0');
	oss << folder << std::setw(10) << ind << ".png";
	return oss.str();
}

void computeObjectMouvement()
{
	cv::Mat image_next, image_prev;
	std::vector<cv::Point> features_next, features_prev;
	std::vector<uchar> status;
	std::vector<float> err;
	
	image_next = cv::imread(getImagePath(FOLDER_LEFT, 0), 0); //First picture

	cv::goodFeaturesToTrack(image_next, features_next, 50, 0.01, 10);

	for (int i = 1; i < NB_FRAME+1; i++)
	{
		image_prev = image_next.clone();
		features_prev = features_next;
		image_next = cv::imread(getImagePath(FOLDER_LEFT, i), 0);  // Get next image

		// Find position of feature in new image
		cv::calcOpticalFlowPyrLK(
			image_prev, image_next, // 2 consecutive images
			features_prev, // input point positions in first im
			features_next, // output point positions in the 2nd
			status,    // tracking success
			err      // tracking error
			);
		for (int i = 0; i < features_next.size(); i++)
		{
			cv::circle(copy, features_next[i], r, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255)), -1, 8, 0);
		}

		// Show what you got
		imshow(source_window, copy);
	}
}

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

void callBackTrackBarDisparity(int pos, void* userdata)
{
	cv::Mat left, right, disparity;

	left = cv::imread(getImagePath(FOLDER_LEFT, pos), 0);
	right = cv::imread(getImagePath(FOLDER_RIGHT, pos), 0);

	disparity = computeDisparity(left, right);
	cv::imshow("Window", disparity);
	cv::waitKey(1);
}

int main(void)
{
	cv::namedWindow("Window", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Window", NULL, NB_FRAME, callBackTrackBarDisparity);

	cv::namedWindow("Optical Flow", CV_WINDOW_AUTOSIZE);
	computeObjectMouvement();

	cv::waitKey();

	return 0;
}