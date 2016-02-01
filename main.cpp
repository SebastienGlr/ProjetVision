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

cv::Mat computeObjectMouvement(int frame)
{
	const int MAX_FEATURE = 500;
	cv::Mat image_next, image_prev, output;
	std::vector<cv::Point2f> features_next, features_prev;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
	
	output = cv::imread(getImagePath(FOLDER_LEFT, frame)); // Output frame for drawing
	image_prev = cv::imread(getImagePath(FOLDER_LEFT, frame - 1), 0); // Get previous image
	image_next = cv::imread(getImagePath(FOLDER_LEFT, frame), 0);  // Get next image

	cv::goodFeaturesToTrack(image_prev, features_prev, MAX_FEATURE, 0.01, 10);
	cv::cornerSubPix(image_prev, features_prev, cv::Size(10, 10), cv::Size(-1, -1), termcrit);

	features_next = features_prev;
	// Find position of feature in new image
	cv::calcOpticalFlowPyrLK(
		image_prev, image_next, // 2 consecutive images
		features_prev, // input point positions in first im
		features_next, // output point positions in the 2nd
		status,    // tracking success
		err,      // tracking error
		cv::Size(31,31),
		3,
		termcrit,
		0,
		0.001);
	for (int i = 0; i < features_next.size(); ++i)
	{
		if (status[i])
		{
			cv::arrowedLine(output, features_prev[i], features_next[i], cv::Scalar(255.0, 0.0, 0.0), 2);
		}
	}
	return output;
}

void callBackTrackBarDisparity(int pos, void*)
{
	cv::Mat left, right, disparity;

	left = cv::imread(getImagePath(FOLDER_LEFT, pos), 0);
	right = cv::imread(getImagePath(FOLDER_RIGHT, pos), 0);

	disparity = computeDisparity(left, right);
	cv::imshow("Window", disparity);
	cv::waitKey(1);
}

void callBackTrackBarOpticalFlow(int pos, void*)
{
	cv::Mat opticalFlow;
	if (pos > 0)
	{
		opticalFlow = computeObjectMouvement(pos);
		cv::imshow("Optical Flow", opticalFlow);
	}
}

int main(void)
{
	cv::namedWindow("Window", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Window", NULL, NB_FRAME, callBackTrackBarDisparity);

	cv::namedWindow("Optical Flow", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Optical Flow", NULL, NB_FRAME, callBackTrackBarOpticalFlow);

	cv::waitKey();

	return 0;
}