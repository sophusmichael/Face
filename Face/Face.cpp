

#include "stdafx.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!


using namespace cv;
using namespace std;




int main(){

	string filePath = "C:\\Users\\michael\\Pictures\\lenacolor.jpg";
	Mat img = imread(filePath);

	if (img.empty()){

		cout << "cannot load image!" << endl;
		return -1;

	}




	CascadeClassifier faceDetector; 
	try {
		faceDetector.load("C:\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml");

	}
	catch (cv::Exception e){}
	if (faceDetector.empty()) {
		cerr << "Error: Couldn't load Face Detector (";
		cerr << "lbpcascade_frontalface.xml" << ")!" << endl;
		exit(1);
	}


	Mat gray;
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
	}

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(img, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}

	imshow("gray", img);

	cout << faces.size() << endl;




	waitKey(0);
	


}