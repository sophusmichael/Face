# Face
face detection via haarcascade for social robots - e-mail questions to msanfilippo@ucmerced.edu

I am using OpenCV 2.4.11 
https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.11/opencv-2.4.11.exe/download

Visual Studio 2013 and Windows 7, x86 version of OpenCV vc12
https://marcomuraresearch.wordpress.com/2015/04/16/install-opencv-visual-studio/

You need to run the python script in the att folder and update the path within the code to where your csv file is
You need to update in Face.cpp:

string fn_haar = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";

string fn_csv = "C:\\Users\\michael\\Documents\\att\\csvfile.csv";

String subject_name = "michael";

string directory = "C:\\Users\\michael\\Documents\\att\\";

Ptr<FaceRecognizer> model = createFisherFaceRecognizer(0,1000); //change 1000 to another constant to change the threshold 

Currently the code has 40 people with 10 images each from the AT&T DB as a base database

//the command to generate the CSV for the training database of images
C:\Python27> python create_csv.py c:/users/michael/documents/att > csvfile.csv

The program doesn't return anything currently but can be set to return the point which is the detected center of the frame

Next steps are to add aditional checks for false positive face detections

//next steps are to increase detection performance for non-frontal faces


//add a condition to only output detected faces after retraining happens
