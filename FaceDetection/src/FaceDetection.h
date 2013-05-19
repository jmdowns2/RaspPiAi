/*
 * FaceDetection.h
 *
 *  Created on: May 14, 2013
 *      Author: matt
 */

#ifndef FACEDETECTION_H_
#define FACEDETECTION_H_

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

class FaceDetection
{
public:
	FaceDetection();

	void initialize(string path, vector<string> files);
	int predictFace(Mat& img);

	bool findAndNormalizeFace(Mat& img);

protected:
	CascadeClassifier faceClassifier;
	CascadeClassifier leftEyeClassifier;
	CascadeClassifier rightEyeClassifier;

	int width;
	int height;

	vector<string> files;

	Ptr<FaceRecognizer> model;

	bool normalizeImage(Mat& img);
	bool findEye(vector< Rect_<int> >& eyes, Rect_<int>& found, bool left);
	Point findCenter(Rect_<int> rect);
	Mat rotateImage(const Mat& source, Point2f pivot, double angle);
};


#endif /* FACEDETECTION_H_ */
