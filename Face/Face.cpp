
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
	string fn_haar = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";
	string fn_csv = "C:\\Users\\michael\\Documents\\att\\csvfile.csv";


	vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);

	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:


			Mat frame;
			for (;;) {
				cap0 >> frame;
				// Clone the current frame:
				Mat original = frame.clone();
				// Convert the current frame to grayscale:
				Mat gray;
				cvtColor(original, gray, CV_BGR2GRAY);
				// Find the faces in the frame:
				vector< Rect_<int> > faces;
				haar_cascade.detectMultiScale(gray, faces);
				for (int i = 0; i < faces.size(); i++) {
					// Process face by face:
					Rect face_i = faces[i];
					// Crop the face from the image. So simple with OpenCV C++:
					Mat face = gray(face_i);
					Mat face_resized;
					cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
					// Now perform the prediction, see how easy that is:
					int prediction = model->predict(face_resized);
					// And finally write all we've found out to the original image!
					// First of all draw a green rectangle around the detected face:
					rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
					// Create the text we will annotate the box with:
					string box_text = format("Prediction = %d", prediction);
					// Calculate the position for annotated text (make sure we don't
					// put illegal values in there):
					int pos_x = std::max(face_i.tl().x - 10, 0);
					int pos_y = std::max(face_i.tl().y - 10, 0);
					// And now put it into the image:
					putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
				}
				// Show the result:
				imshow("face_recognizer", original);
				// And display it:
				char key = (char)waitKey(20);
				// Exit this loop on escape:
				if (key == 27)
					break;
			}

	return 0;
}