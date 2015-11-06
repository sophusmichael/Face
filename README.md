# Face
face detection via LBP for social robots - e-mail questions to msanfilippo@ucmerced.edu

I am using OpenCV 2.4.11 
https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.11/opencv-2.4.11.exe/download

Visual Studio 2013 and Windows 7, x86 version of OpenCV vc12
https://marcomuraresearch.wordpress.com/2015/04/16/install-opencv-visual-studio/

This code takes a USB video stream and samples 1 in 30 frames, then performs face detection via LBPcascade_frontalface


You need to run the python script and update the path within the code to where your csv file is
You will also need to update where your haar frontal face file is 

Use this and the AT&T database to create your csv file
http://docs.opencv.org/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html#creating-the-csv-file

//the command to generate the CSV for the training database of images
C:\Python27> python create_csv.py c:/users/michael/documents/att > csvfile.csv


//next steps are to increase detection performance for non-frontal faces


//add a condition to only output detected faces after retraining happens
