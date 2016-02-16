#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/opencv.hpp>


#include <iostream>
#include <iomanip>

#define FOLDER_LEFT "img/2011_09_26/2011_09_26_drive_0014_sync/image_02/data/" //Left Camera
#define FOLDER_RIGHT "img/2011_09_26/2011_09_26_drive_0014_sync/image_03/data/" //Right Camera
#define FOLDER2_LEFT "img/2011_09_28/2011_09_28_drive_0016_sync/image_03/data/" //Left Camera
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
	cv::Point2f average_prev(0, 0);
	cv::Point2f average_next(0, 0);
	for (int i = 0; i < features_next.size(); ++i)
	{
		if (status[i])
		{
			cv::arrowedLine(output, features_prev[i], features_next[i], cv::Scalar(255.0, 0.0, 0.0), 2);
			average_prev += features_prev[i];
			average_next += features_next[i];
		}
	}
	average_prev = average_prev / (int)features_next.size();
	average_next = average_next / (int)features_next.size();
	cv::Point2f center(output.cols / 2, output.rows / 2);
	//cv::arrowedLine(output, average_next, average_prev, cv::Scalar(0.0, 255.0, 0.0), 5);
	cv::arrowedLine(output, center, center + (average_prev - average_next)*3, cv::Scalar(0.0, 255.0, 0.0), 5);
	return output;
}

//void searchRoadSigns()
//{
//	cv::Mat img_object = cv::imread("img/cedez_le_passage.jpg", 0)
//		, img_scene = cv::imread(getImagePath(FOLDER_LEFT, 2), 0);
//	//Step 1
//	cv::Ptr< cv::xfeatures2d::SURF > detector = cv::xfeatures2d::SURF::create(400);
//	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
//	detector->detect(img_object, keypoints_object);
//	detector->detect(img_scene, keypoints_scene);
//
//	// Step 2
//	cv::Ptr< cv::xfeatures2d::SURF > extractor = cv::xfeatures2d::SURF::create();
//	cv::Mat descriptors_object, descriptors_scene;
//
//	extractor->compute(img_object, keypoints_object, descriptors_object);
//	extractor->compute(img_scene, keypoints_scene, descriptors_scene);
//
//	//Step 3
//	cv::FlannBasedMatcher matcher;
//	std::vector< cv::DMatch > matches;
//	matcher.match(descriptors_object, descriptors_scene, matches);
//
//	double max_dist(0.0), min_dist(100.0);
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		double dist = matches[i].distance;
//		if (dist < min_dist) min_dist = dist;
//		if (dist > max_dist) max_dist = dist;
//	}
//	std::cout << "Max dist : " << max_dist << std::endl;
//	std::cout << "Min dist : " << min_dist<< std::endl;
//
//	std::vector< cv::DMatch > good_matches;
//	for (int i = 0; i < descriptors_object.rows; i++)
//		if(matches[i].distance < 3*min_dist)
//			good_matches.push_back(matches[i]);
//
//	cv::Mat img_matches;
//	cv::drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
//		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
//		std::vector< char >(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	//Localize the object
//	std::vector<cv::Point2f> obj;
//	std::vector<cv::Point2f> scene;
//
//	for (int i = 0; i < good_matches.size(); i++)
//	{
//		//-- Get the keypoints from the good matches
//		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
//		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
//	}
//
//	cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
//
//	//-- Get the corners from the image_1 ( the object to be "detected" )
//	std::vector<cv::Point2f> obj_corners(4);
//	obj_corners[0] = cv::Point2f(0, 0); obj_corners[1] = cv::Point2f(img_object.cols, 0);
//	obj_corners[2] = cv::Point2f(img_object.cols, img_object.rows); obj_corners[3] = cv::Point2f(0, img_object.rows);
//	std::vector<cv::Point2f> scene_corners(4);
//
//	cv::perspectiveTransform(obj_corners, scene_corners, H);
//
//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
//	line(img_matches, scene_corners[0] + cv::Point2f(img_object.cols, 0), scene_corners[1] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[1] + cv::Point2f(img_object.cols, 0), scene_corners[2] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[2] + cv::Point2f(img_object.cols, 0), scene_corners[3] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[3] + cv::Point2f(img_object.cols, 0), scene_corners[0] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
//
//	//-- Show detected matches
//	cv::imshow("Good Matches & Object detection", img_matches);
//
//	cv::waitKey(0);
//}

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

void callBackTrackMovingObjects(int pos, void*)
{
	if (pos > 0)
	{
		cv::Mat img1, img2, fgMaskMOG2;
		cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2();
		img1 = cv::imread(getImagePath(FOLDER2_LEFT, pos-1), 0);
		img2 = cv::imread(getImagePath(FOLDER2_LEFT, pos), 0);
		pMOG2->apply(img1, fgMaskMOG2);
		pMOG2->apply(img2, fgMaskMOG2);
		cv::imshow("Moving Objects", fgMaskMOG2);
	}
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

void callBackTrackBarPedestrians(int pos, void*)
{
	cv::Mat frame = cv::imread(getImagePath(FOLDER2_LEFT, pos));
	cv::HOGDescriptor hog;
	std::vector<cv::Rect> found, found_filtered;
	
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	hog.detectMultiScale(frame, found);

	size_t i, j;
	for (int i = 0; i<found.size(); i++)
	{
		cv::Rect r = found[i];
		for (j = 0; j<found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	for (i = 0; i<found_filtered.size(); i++)
	{
		cv::Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.06);
		r.height = cvRound(r.height*0.9);
		rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("Pedestrians", frame);
}

void testMOG();
int threshold1 = 3, threshold2 = 150;

void callBackTrackBarMOG1(int pos, void*) {
	//cv::Mat left, right, disparity;

	//left = cv::imread(getImagePath(FOLDER_LEFT, pos), 0);
	//right = cv::imread(getImagePath(FOLDER_RIGHT, pos), 0);

	//disparity = computeDisparity(left, right);
	//cv::imshow("Disparity", disparity);
	//cv::waitKey(1);
	threshold1 = pos;
	testMOG();
}

void callBackTrackBarMOG2(int pos, void*) {
	//cv::Mat left, right, disparity;

	//left = cv::imread(getImagePath(FOLDER_LEFT, pos), 0);
	//right = cv::imread(getImagePath(FOLDER_RIGHT, pos), 0);

	//disparity = computeDisparity(left, right);
	//cv::imshow("Disparity", disparity);
	//cv::waitKey(1);
	threshold2 = pos;
	testMOG();
}

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255, 255, 255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}

void testMOG()
{
	cv::Mat src = cv::imread(getImagePath(FOLDER_LEFT, 0));

	// Convert to grayscale
	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);
	// Use Canny instead of threshold to catch squares with gradient shading
	cv::Mat bw;
	cv::Canny(gray, bw, threshold1, threshold2, 3, false);
	cv::imshow("bw", bw);
	cv::createTrackbar("threshold1", "bw", &threshold1, 1500, callBackTrackBarMOG1);
	cv::createTrackbar("threshold2", "bw", &threshold2, 1500, callBackTrackBarMOG2);
	// Find contours
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> approx;
	cv::Mat dst = src.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(cv::Mat(contours[i]), approx, 10, true);

		// Skip small or non-convex objects 
		if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
			continue;

		if (approx.size() == 3)
		{
			setLabel(dst, "TRI", contours[i]);    // Triangles
		}
		else if (approx.size() >= 4 && approx.size() <= 6)
		{
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc + 1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
				setLabel(dst, "RECT", contours[i]);
			else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
				setLabel(dst, "PENTA", contours[i]);
			else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
				setLabel(dst, "HEXA", contours[i]);
		}
		else
		{
			// Detect and label circles
			double area = cv::contourArea(contours[i]);
			cv::Rect r = cv::boundingRect(contours[i]);
			int radius = r.width / 2;

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
				std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
				setLabel(dst, "CIR", contours[i]);
		}
	}

	cv::imshow("src", src);
	cv::imshow("dst", dst);
}

int main(void)
{
	cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Disparity", NULL, NB_FRAME, callBackTrackBarDisparity);

	cv::namedWindow("Optical Flow", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Optical Flow", NULL, NB_FRAME, callBackTrackBarOpticalFlow);

	cv::namedWindow("Moving Objects", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Moving Objects", NULL, 185, callBackTrackMovingObjects);

	cv::namedWindow("Pedestrians", CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	cv::createTrackbar("Frame", "Pedestrians", NULL, 185, callBackTrackBarPedestrians);

	//int matchMethod;
	//cv::namedWindow("Panneau", cv::WINDOW_AUTOSIZE | CV_GUI_NORMAL);
	//cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
	//cv::createTrackbar("Methode", "Panneau", &match_method, max_Trackbar, callBackTrackBarSigns);

	//searchRoadSigns();

	testMOG();
	//cv::Mat frame, dframe;
	//frame = cv::imread(getImagePath(FOLDER_LEFT, 0), 0);
	//cv::fastNlMeansDenoising(frame, dframe);
	//cv::imshow("dframe", dframe);

	cv::waitKey();

	return 0;
}