
#include "stdafx.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <Windows.h>
// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!


using namespace cv;
using namespace std;


void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
	// If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
	Mat gray;
	if (img.channels() == 3) {
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else if (img.channels() == 4) {
		cvtColor(img, gray, CV_BGRA2GRAY);
	}
	else {
		// Access the input image directly, since it is already grayscale.
		gray = img;
	}

	// Possibly shrink the image, to run much faster.
	Mat inputImg;
	float scale = img.cols / (float)scaledWidth;
	if (img.cols > scaledWidth) {
		// Shrink the image while keeping the same aspect ratio.
		int scaledHeight = cvRound(img.rows / scale);
		resize(gray, inputImg, Size(scaledWidth, scaledHeight));
	}
	else {
		// Access the input image directly, since it is already small.
		inputImg = gray;
	}

	// Standardize the brightness and contrast to improve dark images.
	Mat equalizedImg;
	equalizeHist(inputImg, equalizedImg);

	// Detect objects in the small grayscale image.
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	// Enlarge the results if the image was temporarily shrunk before detection.
	if (img.cols > scaledWidth) {
		for (int i = 0; i < (int)objects.size(); i++) {
			objects[i].x = cvRound(objects[i].x * scale);
			objects[i].y = cvRound(objects[i].y * scale);
			objects[i].width = cvRound(objects[i].width * scale);
			objects[i].height = cvRound(objects[i].height * scale);
		}
	}

	// Make sure the object is completely within the image, in case it was on a border.
	for (int i = 0; i < (int)objects.size(); i++) {
		if (objects[i].x < 0)
			objects[i].x = 0;
		if (objects[i].y < 0)
			objects[i].y = 0;
		if (objects[i].x + objects[i].width > img.cols)
			objects[i].x = img.cols - objects[i].width;
		if (objects[i].y + objects[i].height > img.rows)
			objects[i].y = img.rows - objects[i].height;
	}

	// Return with the detected face rectangles stored in "objects".
}

void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
	// Only search for just 1 object (the biggest in the image).
	int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
	// Smallest object size.
	Size minFeatureSize = Size(20, 20);
	// How detailed should the search be. Must be larger than 1.0.
	float searchScaleFactor = 1.1f;
	// How much the detections should be filtered out. This should depend on how bad false detections are to your system.
	// minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
	int minNeighbors = 4;

	// Perform Object or Face Detection, looking for just 1 object (the biggest in the image).
	vector<Rect> objects;
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
	if (objects.size() > 0) {
		// Return the only detected object.
		largestObject = (Rect)objects.at(0);
	}
	else {
		// Return an invalid rect.
		largestObject = Rect(-1, -1, -1, -1);
	}
}
//this function is to convert strings so that it may be used to create a directory
std::wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, vector<string>& names, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);

	int temp_int = -1;
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel, namelabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);

		getline(liness, classlabel, separator);
		getline(liness, namelabel, separator);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
			names.push_back(namelabel);
			cout << namelabel;
		}
	}
}



int main(int argc, const char *argv[]) {
	int count = 0; //variable for timer
	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer(0, 1000);

	VideoCapture cap0(0);
	string fn_haar = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";
	string fn_left = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_lefteye.xml";
	string fn_right = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_righteye.xml";
	string fn_csv = "C:\\Users\\michael\\Documents\\att\\csvfile.csv";
	Point CenterPoint;

	vector<Mat> images;
	vector<int> labels;
	vector<string> names;
	std::map<int, std::string> integer_to_name;

	if (argv[1] != NULL){
		try {
			model->load("C:\\Users\\michael\\Documents\\att\\saved.xml");
		}
		catch (cv::Exception& e) {
			cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
			exit(1);
		}
	}

	try {
		read_csv(fn_csv, images, labels, names);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	//create a map to map integer labels to string names
	for (int i = 0; i < names.size(); i++){
		int current_label = labels[i];
		integer_to_name[current_label] = names[i];
	}


	int im_width = images[0].cols;
	int im_height = images[0].rows;


	if (argv[1] == NULL){
		model->train(images, labels);
	}
	//index for next face
	int new_index = labels.back() + 1;
	//model->load("C:\\Users\\michael\\Documents\\att\\fisherfaces_at.yml");

	CascadeClassifier haar_cascade;
	CascadeClassifier eyes_cascade_left;
	CascadeClassifier eyes_cascade_right;
	haar_cascade.load(fn_haar);
	eyes_cascade_left.load(fn_left);
	eyes_cascade_right.load(fn_right);
	// Get a handle to the Video device:

	int j = 0;

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


					const float EYE_SX = 0.16f;
					const float EYE_SY = 0.26f;
					const float EYE_SW = 0.30f;
					const float EYE_SH = 0.28f;
					Point leftEye;
					Point rightEye;

					int leftX = cvRound(face.cols * EYE_SX);
					int topY = cvRound(face.rows * EYE_SY);
					int widthX = cvRound(face.cols * EYE_SW);
					int heightY = cvRound(face.rows * EYE_SH);
					int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));  // Start of right-eye corner

					Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
					Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
					Rect leftEyeRect, rightEyeRect;

					detectLargestObject(topLeftOfFace, eyes_cascade_left, leftEyeRect, topLeftOfFace.cols);
					detectLargestObject(topRightOfFace, eyes_cascade_right, rightEyeRect, topRightOfFace.cols);

					if (leftEyeRect.width > 0) {   // Check if the eye was detected.
						leftEyeRect.x += leftX;    // Adjust the left-eye rectangle because the face border was removed.
						leftEyeRect.y += topY;
						leftEye = Point(leftEyeRect.x + leftEyeRect.width / 2, leftEyeRect.y + leftEyeRect.height / 2);
					}
					else {
						leftEye = Point(-1, -1);    // Return an invalid point
					}


					if (rightEyeRect.width > 0) { // Check if the eye was detected.
						rightEyeRect.x += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
						rightEyeRect.y += topY;  // Adjust the right-eye rectangle because the face border was removed.
						rightEye = Point(rightEyeRect.x + rightEyeRect.width / 2, rightEyeRect.y + rightEyeRect.height / 2);
					}
					else {
						rightEye = Point(-1, -1);    // Return an invalid point
					}

					cout << leftEye;

					//-- In each face, detect eyes

					cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
					// Now perform the prediction, see how easy that is:

					//only gives prediction if there is an eye detection too!
					
					//int prediction = model->predict(face_resized);
					int prediction = -1;
					double predicted_confidence = 0.0;
					// Get the prediction and associated confidence from the model
					model->predict(face_resized, prediction, predicted_confidence);
					cout << "prediction is" << prediction << "\n";
					cout << "predicted confidence is" << predicted_confidence << "\n";

					//here we will check to see if there is no good prediction than the face is unknown
					//also check to make sure we detected an eye!
					if (j < 20 && prediction == -1 && leftEye != Point(-1, -1) && rightEye != Point(-1, -1)){
							//take 10 photos? what if there is an error with training on the person speaking?
							stringstream ss;
							ss << j+1;
							string index = ss.str();
							String subject_name = "michael";
							string image = "\\";
							string jpg = ".jpg";
							string directory = "C:\\Users\\michael\\Documents\\att\\";
							string newdir = directory + subject_name;

							std::wstring stemp = s2ws(newdir);
							LPCWSTR dirname = stemp.c_str();
							CreateDirectory(dirname, NULL);
							string savepath = directory + subject_name + image + index + jpg;

							imwrite(savepath, face_resized);
							cout << j;
							j++;

							images.push_back(face_resized);
							labels.push_back(new_index);
							names.push_back(subject_name);

							if (j == 19){
								cout << "retraining \n";
								model->train(images, labels);
								model->save("C:\\Users\\michael\\Documents\\att\\saved.xml");
								//we must update the map as well
								for (int i = 0; i < labels.size(); i++){
									int current_label = labels[i];
									integer_to_name[current_label] = subject_name;
								}
							}

					}





					// And finally write all we've found out to the original image!
					// First of all draw a green rectangle around the detected face:
					rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
					// Create the text we will annotate the box with:

					string it; 
					it = integer_to_name[prediction];
					if (prediction == -1){
						it = "unknown";
					}
					cout << it <<"\n";
					string box_text = "Prediction = " + it;
					//string box_text = format("Prediction = ", boxname);
					// Calculate the position for annotated text (make sure we don't
					// put illegal values in there):
					int pos_x = std::max(face_i.tl().x - 10, 0);
					int pos_y = std::max(face_i.tl().y - 10, 0);
					// And now put it into the image:
					CenterPoint = Point(pos_x, pos_y);
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