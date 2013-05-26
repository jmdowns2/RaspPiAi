/////////////////////////////////////////////////////////
//
// preparePhoto.cpp
// this file prepare photo : crop, rotate, align, gray
// for facial reco training
//
// This file is strongly inspired
// from Philipp Wagner Phyton works
// http://docs.opencv.org/trunk/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html
// Thank you Philipp, you're good.
//
// This file is put as exemple - considered it as a draft source code
// debug lines are still here. No memory optimization etc.
// but it does the job
//
// Pierre Raufast - 2013
// (for a better world, read Giono's books)
//
/////////////////////////////////////////////////////////


#include "NormalizeImage.h"

#include <iostream>
#include <stdio.h>
#include <dirent.h>

#define TRACE 0						// for trace fonction
#define DEBUG_MODE 0				//for debug trace
#define DEBUG if (DEBUG_MODE==1)	// for debug trace
#define MIN_EYE_DIST 15
using namespace std;
using namespace cv;


// change opencv path here <---------------
string face_cascade_name = "/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_frontalface_default.xml";
string glasses_cascade_name = "/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string eyes_cascade_name = "/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_eye.xml";
string eyes_left_cascade_name = "/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_lefteye_2splits.xml";
string eyes_right_cascade_name = "/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_righteye_2splits.xml";



///////////////////////////////////////////////////
// trace fonction, output only if #define TRACE=1
///////////////////////////////////////////////////
void trace(string s)
{
	if (TRACE==1)
	{
		cout<<s<<"\n";
	}
}


NormalizeImage::NormalizeImage(int bEqHisto, int iSize, float fPercentCrop, int iOutSize)
{
	this->bEqHisto = bEqHisto;
	this->iSize = iSize;

	bEqHistoWhenFindingEyes = 0;

	Myoffset_pct.x =100.0*fPercentCrop;
	Myoffset_pct.y = Myoffset_pct.x;

	// size of new picture
	Mydest_sz.x = iOutSize;
	Mydest_sz.y = Mydest_sz.x;
}

NormalizeImage::~NormalizeImage() {
}

int NormalizeImage::init()
{
	//-- 1. Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !glasses_cascade.load( glasses_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_left_cascade.load( eyes_left_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_right_cascade.load( eyes_right_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	return 0;
}


//////////////////////////////////////////////
// detectAndDisplay
//////////////////////////////////////////////

int NormalizeImage::detectEyes(Mat& face, int iOffsetX, int iOffsetY)
{
	std::vector<Rect> eyes;

	//-- In each face, detect eyes
	eyes_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	// if no glasses
	if (eyes.size()==2)
	{
		trace("-- face without glasses");
		// detect eyes
		for( size_t j = 0; j < 2; j++ )
		{
			Point eye_center( iOffsetX + eyes[1-j].x + eyes[1-j].width/2, iOffsetY + eyes[1-j].y + eyes[1-j].height/2 );

			if (j==0) // left eye
			{
				Myeye_left.x =eye_center.x;
				Myeye_left.y =eye_center.y;
			}
			if (j==1) // right eye
			{
				Myeye_right.x =eye_center.x;
				Myeye_right.y =eye_center.y;
			}
		}

		return 1;
	}

	return 0;
}

int NormalizeImage::detectEyesWithGlasses(Mat& face, int iOffsetX, int iOffsetY)
{
	std::vector<Rect> eyes;

	// tests with glasses
	glasses_cascade.detectMultiScale( face, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
	if (eyes.size()!=2) return 0;
	else
	{

		trace("-- face with glasses");

		for( size_t j = 0; j < 2; j++ )
		{
			Point eye_center( iOffsetX + eyes[1-j].x + eyes[1-j].width/2, iOffsetY + eyes[1-j].y + eyes[1-j].height/2 );
			if (j==0) // left eye
			{
				Myeye_left.x =eye_center.x;
				Myeye_left.y =eye_center.y;
			}
			if (j==1) // right eye
			{
				Myeye_right.x =eye_center.x;
				Myeye_right.y =eye_center.y;
			}
		}

		return 1;
	}

}


bool NormalizeImage::_findEye(vector<Rect_<int> >& eyes, Rect_<int>& found,bool left)
{
	if(eyes.size() == 0)
	{
		cout << "ERROR No eyes found" << endl;
		return false;
	}
	else if (eyes.size() == 1)
	{
		found = eyes.at(0);
		return true;
	}
	else if (eyes.size() == 2)
	{
		Rect_<int> eye1 = eyes.at(0);
		Rect_<int> eye2 = eyes.at(1);

		int eye1Center = eye1.x + eye1.width/2;
		int eye2Center = eye2.x + eye2.width/2;

		if(left)
		{
			if(eye1Center > eye2Center)
				found = eye1;
			else
				found = eye2;
		}
		else
		{
			if(eye1Center > eye2Center)
				found = eye2;
			else
				found = eye1;
		}
		return true;
	}
	else
	{
		cout << "ERROR detecting eyes";
		return false;
	}
}

int NormalizeImage::detectEyesIndependently(Mat& face, int iOffsetX, int iOffsetY)
{
	vector< Rect_<int> > leftEyes;
	vector< Rect_<int> > rightEyes;
	eyes_left_cascade.detectMultiScale(face, leftEyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(10, 10));
	eyes_right_cascade.detectMultiScale(face, rightEyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(10, 10));
	//eyes_left_cascade.detectMultiScale(face, leftEyes);
	//eyes_right_cascade.detectMultiScale(face, rightEyes);

	cout << "W=" << face.rows << endl;
	cout << "Left=" << leftEyes.size() << endl;
	cout << "Right=" << rightEyes.size() << endl;


	Rect_<int> leftEye;
	bool found = true;
	found &= _findEye(leftEyes, leftEye, true);

	Rect_<int> rightEye;
	found &= _findEye(rightEyes, rightEye, false);

	if(!found)
		return 0;

	trace("-- face found independently");

	Myeye_left.x = iOffsetX + leftEye.x + leftEye.width/2;
	Myeye_left.y = iOffsetY + leftEye.y + leftEye.height/2;

	Myeye_right.x = iOffsetX + rightEye.x + rightEye.width/2;;
	Myeye_right.y = iOffsetY + rightEye.y + rightEye.height/2;

	return 1;
}


int NormalizeImage::detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	//convert to gray scale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	if (bEqHisto==1 && bEqHistoWhenFindingEyes)
	{
		equalizeHist( frame_gray, frame_gray );
	}
	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );


	// simplify : we only take picture with one face !
	DEBUG printf("(D) detectAndDisplay : nb face=%d\n",faces.size());
	if (faces.size()==0) return 0;
	else
		for( size_t i = 0; i < 1; i++ ) // only first face !
		{
			Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );

			Mat faceROI = frame_gray( faces[i] );
//			std::vector<Rect> eyes;

			//-- In each face, detect eyes
//			eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

			// if no glasses
			if(detectEyes(faceROI, faces[i].x, faces[i].y))
			{
				trace("FOUND EYES");
			}
			else if(detectEyesWithGlasses(faceROI, faces[i].x, faces[i].y))
			{
				trace("FOUND EYES");
			}
			else if(detectEyesIndependently(faceROI, faces[i].x, faces[i].y))
			{
				trace("FOUND EYES");
			}
			else
			{
				return 0;
			}

/*

			if (eyes.size()==2)
			{
				trace("-- face without glasses");
				// detect eyes
				for( size_t j = 0; j < 2; j++ )
				{
					Point eye_center( faces[i].x + eyes[1-j].x + eyes[1-j].width/2, faces[i].y + eyes[1-j].y + eyes[1-j].height/2 );

					if (j==0) // left eye
					{
						Myeye_left.x =eye_center.x;
						Myeye_left.y =eye_center.y;
					}
					if (j==1) // right eye
					{
						Myeye_right.x =eye_center.x;
						Myeye_right.y =eye_center.y;
					}
				}
			}
			else
			{
				// tests with glasses
				glasses_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
				if (eyes.size()!=2) return 0;
				else
				{

					trace("-- face with glasses");

					for( size_t j = 0; j < 2; j++ )
					{
						Point eye_center( faces[i].x + eyes[1-j].x + eyes[1-j].width/2, faces[i].y + eyes[1-j].y + eyes[1-j].height/2 );
						if (j==0) // left eye
						{
							Myeye_left.x =eye_center.x;
							Myeye_left.y =eye_center.y;
						}
						if (j==1) // right eye
						{
							Myeye_right.x =eye_center.x;
							Myeye_right.y =eye_center.y;
						}
					}
				}
			}
			*/

		}
	// sometimes eyes are inversed ! we switch them
	if (Myeye_right.x<Myeye_left.x)
	{
		int tmpX = Myeye_right.x;
		int tmpY = Myeye_right.y;
		Myeye_right.x=Myeye_left.x;
		Myeye_right.y=Myeye_left.y;
		Myeye_left.x=tmpX;
		Myeye_left.y=tmpY;
		trace("-- oups, switch eyes");

	}

	return 1;
}


//////////////////////////////////////////////
// compute distance btw 2 points
//////////////////////////////////////////////
float NormalizeImage::Distance(CvPoint p1, CvPoint p2)
{
	int dx = p2.x - p1.x;
	int dy = p2.y - p1.y;
	return sqrt(dx*dx+dy*dy);
}


//////////////////////////////////////////////
// rotate picture (to align eyes-y)
//////////////////////////////////////////////
Mat NormalizeImage::rotate(Mat& image, double angle, CvPoint centre)
{
	Point2f src_center(centre.x, centre.y);
	// conversion en degre
	angle = angle*180.0/3.14157;
	DEBUG printf("(D) rotate : rotating : %fÂ° %d %d\n",angle, centre.x, centre.y);
	Mat rot_matrix = getRotationMatrix2D(src_center, angle, 1.0);

	Mat rotated_img(Size(image.size().height, image.size().width), image.type());

	warpAffine(image, rotated_img, rot_matrix, image.size());
	return (rotated_img);
}



//////////////////////////////////////////////
// crop picture
//////////////////////////////////////////////
int NormalizeImage::CropFace(Mat &MyImage,
		CvPoint eye_left,
		CvPoint eye_right,
		CvPoint offset_pct,
		CvPoint dest_sz)
{

	// calculate offsets in original image
	int offset_h = (offset_pct.x*dest_sz.x/100);
	int offset_v = (offset_pct.y*dest_sz.y/100);
	DEBUG printf("(D) CropFace : offeth=%d, offsetv=%d\n",offset_h,offset_v);

	// get the direction
	CvPoint eye_direction;
	eye_direction.x = eye_right.x - eye_left.x;
	eye_direction.y = eye_right.y - eye_left.y;


	// calc rotation angle in radians
	float rotation = atan2((float)(eye_direction.y),(float)(eye_direction.x));

	// distance between them
	float dist = Distance(eye_left, eye_right);
	DEBUG printf("(D) CropFace : dist=%f\n",dist);

	if(dist < MIN_EYE_DIST)
	{
		cout << "DISTANCE too small" << endl;
		return 0;
	}


	// calculate the reference eye-width
	int reference = dest_sz.x - 2*offset_h;

	// scale factor
	float scale = dist/(float)reference;
	DEBUG printf("(D) CropFace : scale=%f\n",scale);

	// rotate original around the left eye
	char sTmp[16];
	sprintf(sTmp,"%f",rotation);
	trace("-- rotate image "+string(sTmp));
	MyImage = rotate(MyImage, (double)rotation, eye_left);

	// crop the rotated image
	CvPoint crop_xy;
	crop_xy.x = eye_left.x - scale*offset_h;
	crop_xy.y = eye_left.y - scale*offset_v;

	CvPoint crop_size;
	crop_size.x = dest_sz.x*scale;
	crop_size.y = dest_sz.y*scale;

	// Crop the full image to that image contained by the rectangle myROI
	trace("-- crop image");
	DEBUG printf("(D) CropFace : crop_xy.x=%d, crop_xy.y=%d, crop_size.x=%d, crop_size.y=%d",crop_xy.x, crop_xy.y, crop_size.x, crop_size.y);

	cv::Rect myROI(crop_xy.x, crop_xy.y, crop_size.x, crop_size.y);
	if ((crop_xy.x+crop_size.x<MyImage.size().width)&&(crop_xy.y+crop_size.y<MyImage.size().height))
	{MyImage = MyImage(myROI);}
	else
	{
		trace("-- error cropping");
		return 0;
	}

	//resize it
	trace("-- resize image");
	cv::resize(MyImage, MyImage, Size(dest_sz));

	return 1;
}


//////////////////////////////////////////////
// resize picture
//////////////////////////////////////////////
void NormalizeImage::resizePicture(Mat& src, int coeff)
{
	// Resize src to img size
	Size oldTaille = src.size();
	Size newTaille(coeff,oldTaille.height*coeff/oldTaille.width);
	cv::resize(src, src, newTaille);
}


Mat NormalizeImage::normalize(Mat frame)
{
	try
	{
		// resize picture
		if (frame.size().width > iSize)
		{
			trace("- image need to be resized");
			resizePicture(frame,iSize);
			//		imwrite(imageName,frame,qualityType);
		}


		// Apply the classifier to the frame
		if( !frame.empty() )
		{
			trace("- start detect");

			int result = detectAndDisplay( frame );
			if (result==0)
			{
				trace("- no face detected");
			}
			else
			{
				// crop face
				trace ("- start cropFace");
				if (CropFace(frame, Myeye_left, Myeye_right, Myoffset_pct,Mydest_sz)==1)
				{
					char newName[16];
					//			sprintf(newName,"%s%d.jpg",argv[3],index);
					//			string newNameS(newName);

					// convert to grayscale
					Mat grayframe;
					trace("- transforme : gray");
					cvtColor(frame, grayframe, CV_BGR2GRAY);

					// equalize histo color
					if (bEqHisto==1)
					{
						trace("- transforme : equalize histo");
						equalizeHist( grayframe, grayframe);
					}
					// save face

					return grayframe;

					//			trace("- save image "+newNameS);
					//			imwrite(newNameS,grayframe,qualityType);
				}
				else
				{
					trace("- crop face failed");
				}
			}
		}
	}
	catch(Exception e)
	{
		trace("Exception");
	}
	Mat dummy;
	return dummy;
}


//////////////////////////////////////////////
// Main program
//////////////////////////////////////////////
/*
int main( int argc, char** argv )
{

	Mat frame;

	if (argc!=6)
	{
		printf("prepare %of_picture_cropped size_of_train_picture image_name_prefix original_picture_new_size equalize_color \nprepare 0.3 100 p 800 1\n");
		return 0;
	}


	int iPercent = atoi(argv[1]);
	float iTrainSize = atoi(argv[2]);

	// equalize histo color ?
	int bEqHisto = atoi(argv[5]);

	// new size of picture
	int newSize = atoi(argv[4]);

	NormalizeImage imgNorm(bEqHisto, newSize, iPercent, iTrainSize);
	if(imgNorm.init() < 0)
		return -1;

	// quality type JPG to save image
	std::vector<int> qualityType;
	qualityType.push_back(CV_IMWRITE_JPEG_QUALITY);
	qualityType.push_back(90);

	// read the current directory
	DIR * rep =opendir(".");
	if (rep==NULL) return 0;

	struct dirent *ent;
	int index=1;

	while ((ent=readdir(rep)) != NULL)
	{
		int nLen = strlen(ent->d_name);
		char * imageName = ent->d_name;

		// read extention, only keep jpg file
		if (nLen>4)
		{
			if ((imageName[nLen-1]=='g')&&(imageName[nLen-2]=='p')&&(imageName[nLen-3]=='j'))
			{
				// Read the video stream
				trace("lecture : "+string(imageName));
				frame = imread(imageName,1);

				Mat normalized = imgNorm.normalize(frame);

				if(normalized.empty())
					continue;

				char newName[16];
				sprintf(newName,"%s%d.jpg",argv[3],index);
				string newNameS(newName);

				// save face
				trace("- save image "+newNameS);
				imwrite(newNameS, normalized, qualityType);
			}
		}
		index ++;
	}
	closedir(rep);
	return 0;
}




*/
