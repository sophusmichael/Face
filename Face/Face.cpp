

#include "stdafx.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!


using namespace cv;
using namespace std;

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
	string fn_haar = string(argv[1]);
	string fn_csv = string(argv[2]);
	int deviceId = atoi(argv[3]);


	vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:
	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	// later: 
	while (cap0.isOpened()) {
		Mat frame0;
		Mat img;
		Mat gray;
		cap0.read(frame0);
		if (cv::waitKey(30) >= 0) break;
		// stereo processing here

		if (count % 30 == 0)
		{
			cap0 >> img;

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

				Rect face_i = faces[i];
				// Crop the face from the image. So simple with OpenCV C++:
				Mat face = gray(face_i);
				Mat face_resized;
				cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				// Now perform the prediction, see how easy that is:
				int prediction = model->predict(face_resized);
			}

			imshow("gray", img);

		}
		count++;
	}









	waitKey(0);



}