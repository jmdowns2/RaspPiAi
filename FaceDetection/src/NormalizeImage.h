/*
 * NormalizeImage.h
 *
 *  Created on: May 26, 2013
 *      Author: matt
 */

#ifndef NORMALIZEIMAGE_H_
#define NORMALIZEIMAGE_H_

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class NormalizeImage {
public:
	NormalizeImage(int bEqHisto, int iSize, float fPercentCrop, int iOutSize);
	virtual ~NormalizeImage();

	int init();
	Mat normalize(Mat input);


	int detectAndDisplay( Mat frame );
	float Distance(CvPoint p1, CvPoint p2);
	Mat rotate(Mat& image, double angle, CvPoint centre);

	int CropFace(Mat &MyImage,
		CvPoint eye_left,
		CvPoint eye_right,
		CvPoint offset_pct,
		CvPoint dest_sz);

	void resizePicture(Mat& src, int coeff);


	CvPoint Myoffset_pct;
	CvPoint Mydest_sz;
	CvPoint Myeye_left;
	CvPoint Myeye_right;

protected:

	int detectEyes(Mat& face, int iOffsetX, int iOffsetY);
	int detectEyesWithGlasses(Mat& face, int iOffsetX, int iOffsetY);

	int detectEyesIndependently(Mat& face, int iOffsetX, int iOffsetY);
	bool _findEye(vector<Rect_<int> >& eyes, Rect_<int>& found,bool left);


	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;
	CascadeClassifier glasses_cascade;
	CascadeClassifier eyes_left_cascade;
	CascadeClassifier eyes_right_cascade;

	int bEqHisto;

	int iSize;
	int bEqHistoWhenFindingEyes;
};

#endif /* NORMALIZEIMAGE_H_ */
