

#include "stdafx.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <time.h>
// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"
#include <fstream>
#include <sstream>

// Include the rest of our code!


using namespace cv;
using namespace std;

static Mat processImage(Mat& image)
{
	cv::Mat resized;
	cv::Size originalSize = image.size();

	cv::Size goalSize = cv::Size(140, 200);
	bool keepAspectRatio = true;

	if (keepAspectRatio)
	{
		float ratio = static_cast<float>(goalSize.height) / originalSize.height;
		cv::Size newSize((int)(originalSize.width * ratio), (int)(originalSize.height * ratio));

		//fix possible rounding error by float
		if (newSize.height != goalSize.height) newSize.height = goalSize.height;

		cv::resize(image, resized, newSize);

		if (resized.size().width != goalSize.width)
		{
			if (keepAspectRatio)
			{
				int delta = goalSize.width - resized.size().width;

				if (delta < 0)
				{
					cv::Rect clipRect(std::abs(delta) / 2, 0, goalSize.width, resized.size().height);
					resized = resized(clipRect);
				}
				else if (delta > 0)
				{
					//width needs to be widened, create bigger mat, get region of 
					//interest at the center that matches the size of the resized   
					//image, and copy the resized image into that ROI

					cv::Mat widened(goalSize, resized.type());
					cv::Rect widenRect(delta / 2, 0, goalSize.width, goalSize.height);
					cv::Mat widenedCenter = widened(widenRect);
					resized.copyTo(widenedCenter);
					resized = widened; //we return resized, so set widened to resized
				}
			}
		}
	}
	else
		cv::resize(image, resized, goalSize);

	cv::Mat grayFrame;
	cv::cvtColor(resized, grayFrame, CV_BGR2GRAY);

	return grayFrame;
}



static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


int main(int argc, const char *argv[]) {
	int count = 0; //variable for timer
	VideoCapture cap0(0);

	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}
	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}
	//for training fisherfaces
	vector<Mat> images;
	vector<int> labels;





	// later: 
	while (cap0.isOpened()) {
		Mat frame0;    
		Mat img;
		Mat gray;
		Mat frame_gray;

		cap0.read(frame0);
		if (cv::waitKey(30) >= 0) break;
		// stereo processing here

		if (count % 30 == 0)
		{
		cap0 >> img;

		CascadeClassifier faceDetector;
		CascadeClassifier eyeDetector;


		try {
			faceDetector.load("C:\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml");
			eyeDetector.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");
		}
		catch (cv::Exception e){}
		if (faceDetector.empty() || eyeDetector.empty()) {
			cerr << "Error: Couldn't load Face/Eye Detector (";
			cerr << "lbpcascade_frontalface.xml" << ")!" << endl;
			exit(1);
		}


		if (img.channels() == 3) {
			cvtColor(img, gray, CV_BGR2GRAY);
		}
		else if (img.channels() == 4) {
			cvtColor(img, gray, CV_BGRA2GRAY);
		}
		else {
			gray = img;
		}

		const int DETECTION_WIDTH = 320;
		Mat smallImg;
		float scale = img.cols / (float)DETECTION_WIDTH;
		if (img.cols > DETECTION_WIDTH) {
			int scaledHeight = cvRound(img.rows / scale);
			resize(gray, smallImg, Size(DETECTION_WIDTH, scaledHeight));
		}
		else {
			smallImg = gray;
		}

		Mat equalizedImg;
		equalizedImg;
		equalizeHist(smallImg, equalizedImg);
		img = equalizedImg;

		int flags = CASCADE_SCALE_IMAGE;
		Size minFeatureSize(20, 20);
		float searchScaleFactor = 1.1f;
		int minNeighbors = 4;

		std::vector<Rect>faces;

		faceDetector.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);

		if (img.cols > DETECTION_WIDTH) {
			for (int i = 0; i < (int)faces.size(); i++) {
				faces[i].x = cvRound(faces[i].x * scale);
				faces[i].y = cvRound(faces[i].y * scale);
				faces[i].width = cvRound(faces[i].width * scale);
				faces[i].height = cvRound(faces[i].height * scale);
			}
		}


		for (int i = 0; i < (int)faces.size(); i++) {
			if (faces[i].x < 0)
				faces[i].x = 0;
			if (faces[i].y < 0)
				faces[i].y = 0;
			if (faces[i].x + faces[i].width > img.cols)
				faces[i].x = img.cols - faces[i].width;
			if (faces[i].y + faces[i].height > img.rows)
				faces[i].y = img.rows - faces[i].height;


			Mat faceROI = gray(faces[i]);
			std::vector<Rect> eyes;
			eyeDetector.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		}


		for (size_t i = 0; i < faces.size(); i++)
		{


			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(img, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}



		// add support to recognize eye detection to reduce false positives 
		// haaarcascade_mcs_lefteye.xml and haaarcascade_mcs_lefteye.xml have 80% reliability with open/closed eyes
		// cost is approximately 18ms




		imshow("gray", img);

		stringstream ss;
		//since we are sampling only every 30 frames we divide by 30 for the count
		int updatedCount = count / 30;
		ss << updatedCount;
		string str = ss.str();
		//we can adjust the subject name to save a new set of training images
		string subjectName = "first\\michael";
		//we can adjust the saveLoc to a new filepath on a different system
		string saveLoc = "C:\\Users\\michael\\Documents\\FacePhotos\\";
		string saveName = str + ".jpg";
		string testSave = saveLoc + subjectName + saveName;
		printf("%s\n", testSave.c_str());
		if (updatedCount < 10){
			imwrite(testSave, img);
		}

		}
		count++;
	}



	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	//model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
//	int predictedLabel = model->predict(testSample);





	waitKey(0);
	


}