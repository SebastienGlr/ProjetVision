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

int match_method;
int max_Trackbar = 5;

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
	cv::imshow("Disparity", disparity);
	cv::waitKey(1);
}

void callBackTrackBarOpticalFlow(int pos, void*)
{
	if (pos > 0)
	{
		cv::Mat opticalFlow;
		opticalFlow = computeObjectMouvement(pos);
		cv::imshow("Optical Flow", opticalFlow);
	}
}

void callBackTrackBarMovingObject(int pos, void*)
{
	
}

void callBackTrackBarSigns(int , void*)
{
	cv::Mat img, templ, result;

	img = cv::imread(getImagePath(FOLDER_LEFT, 6), 0);
	templ = cv::imread("img/cedez.png", 0);

	/// Source image to display
	cv::Mat img_display;
	img.copyTo(img_display);

	/// Create the result matrix
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	cv::matchTemplate(img, templ, result, match_method);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	/// Localizing the best match with minMaxLoc
	double minVal, maxVal; 
	cv::Point minLoc, maxLoc, matchLoc;

	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	cv::rectangle(img_display, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
	cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);

	cv::imshow("Panneau", img_display);
	cv::imshow("Result", result);

}

int main(void)
{
	//cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	//cv::createTrackbar("Frame", "Disparity", NULL, NB_FRAME, callBackTrackBarDisparity);

	//cv::namedWindow("Optical Flow", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	//cv::createTrackbar("Frame", "Optical Flow", NULL, NB_FRAME, callBackTrackBarOpticalFlow);

	//cv::namedWindow("Moving Object", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	//cv::createTrackbar("Frame", "Moving Object", NULL, NB_FRAME, callBackTrackBarMovingObject);

	int matchMethod;
	cv::namedWindow("Panneau", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Methode", "Panneau", &match_method, max_Trackbar, callBackTrackBarSigns);

	cv::waitKey();

	return 0;
}