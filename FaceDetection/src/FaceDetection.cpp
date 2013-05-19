#include "FaceDetection.h"
#include "math.h"
#include <sys/types.h>
#include <dirent.h>
#include <iostream>

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
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}



FaceDetection::FaceDetection()
{
    cout << "Loading Classifiers" << endl;

    faceClassifier.load("/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_frontalface_default.xml");
    leftEyeClassifier.load("/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_lefteye_2splits.xml");
    rightEyeClassifier.load("/home/matt/workspace/opencv-2.4.5/data/haarcascades/haarcascade_righteye_2splits.xml");
}

void FaceDetection::initialize(string path, vector<string> files)
{
	cout << "Initializing..." << endl;

	this->files = files;

    vector<Mat> images;
    vector<int> labels;

    for(int iFile=0;iFile<files.size(); ++iFile)
    {
    	string file = files.at(iFile);

        if(file.find(".jpg") == string::npos )
        {
        	continue;
        }


    	Mat temp = imread(path+"/"+file, 0);
    	cout << temp.rows << "  " << temp.cols << endl;
        images.push_back(temp);
        labels.push_back(iFile);
    }

    /*
    try {
        read_csv(csvFile, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << csvFile << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }*/


    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    width = images[0].cols;
    height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
   // model = createFisherFaceRecognizer();
    //model = createEigenFaceRecognizer(6, 0.0);
    model = createLBPHFaceRecognizer();
    model->train(images, labels);

    cout << "Initialized" << endl;
}

int FaceDetection::predictFace(Mat& img)
{
	cout << "Predicting..." << endl;

	int prediction = -1;

	// Clone the current frame:
	Mat original = img.clone();
	// Convert the current frame to grayscale:
	Mat gray;
	cvtColor(original, gray, CV_BGR2GRAY);
	// Find the faces in the frame:
	vector< Rect_<int> > faces;
	faceClassifier.detectMultiScale(gray, faces);
	// At this point you have the position of the faces in
	// faces. Now we'll get the faces, make a prediction and
	// annotate it in the video. Cool or what?

	cout << "Found " << faces.size() << " faces" << endl;
	for(uint i = 0; i < faces.size(); i++) {
		// Process face by face:
		Rect face_i = faces[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = gray(face_i);

		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:

		Mat face_resized;
		cv::resize(face, face_resized, Size(width, height), 1.0, 1.0, INTER_CUBIC);

		bool normalized = normalizeImage(face_resized);
		if(!normalized)
		{
			continue;
		}

		// Now perform the prediction, see how easy that is:
		prediction = model->predict(face_resized);
		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(original, face_i, CV_RGB(0, 255,0), 1);
		// Create the text we will annotate the box with:

		string box_text = "Prediction = " + this->files.at(prediction);
		// Calculate the position for annotated text (make sure we don't
		// put illegal values in there):
		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);
		// And now put it into the image:
		putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
	}
	// Show the result:
	imshow("face_recognizer", original);
	// And display it:
	(char) waitKey(2000);

	cout << "Prediction finished" << endl;

	return prediction;
}

bool FaceDetection::findAndNormalizeFace(Mat& img)
{
	// Find the faces in the frame:
	vector< Rect_<int> > faces;
	faceClassifier.detectMultiScale(img, faces);

	if(faces.size() != 1)
	{
		cout << "Unable to find faces" << endl;
		return false;
	}

	Rect face = faces.at(0);
	img = img(face);
	return normalizeImage(img);
}

bool FaceDetection::normalizeImage(Mat& img)
{
	cvtColor( img, img, CV_BGR2GRAY );
	equalizeHist(img, img);
	imshow("Intro", img);
	waitKey(4000);


	int horPercent = 20;
	int vertPercent = 10;
	int width = 200;
	int height = 200;

	vector< Rect_<int> > leftEyes;
	vector< Rect_<int> > rightEyes;
	//leftEyeClassifier.detectMultiScale(img, leftEyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(5, 5));
	//rightEyeClassifier.detectMultiScale(img, rightEyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(5, 5));
	leftEyeClassifier.detectMultiScale(img, leftEyes);
	rightEyeClassifier.detectMultiScale(img, rightEyes);

	Rect_<int> leftEye;
	bool found = true;
	found &= findEye(leftEyes, leftEye, true);
	Rect_<int> rightEye;
	found &= findEye(rightEyes, rightEye, false);

	if(!found)
	{
		cout << "Failed to find eyes aborting normailization" << endl;
		return false;
	}

	Point leftCenter = findCenter(leftEye);
	Point rightCenter = findCenter(rightEye);

//	cv::rectangle(img, rightEye, Scalar( 100, 0, 0 ), -1, 8 );
//	cv::rectangle(img, leftEye, Scalar( 255, 0, 0 ), -1, 8 );
//	imshow("face_recognizer", img);
//	waitKey(2000);


	// Handle rotation
	double angle = atan( 1.0 * leftCenter.y - rightCenter.y ) / (leftCenter.x - rightCenter.x );
	Mat rotatedImg = rotateImage(img, rightCenter, angle);

	// Center eyes
	int iDistance = leftCenter.x - rightCenter.x;

	int iCenterPercent = 100 - horPercent * 2;
	double perWidth = 1.0 * iDistance / iCenterPercent;
	int iLeft = rightCenter.x - perWidth * horPercent;
	int iRight = leftCenter.x + perWidth * horPercent;
	int iTop = rightCenter.y - perWidth * vertPercent;

	cout << iLeft << " " << iRight << " " << iTop << endl;

	rotatedImg = rotatedImg(Rect_<int>(iLeft, iTop, iRight-iLeft, iRight-iLeft));

	// Resize image
	cv::resize(rotatedImg, img, Size(width, height), 1.0, 1.0, INTER_CUBIC);

	imshow("test1", img);
	waitKey(4000);


	return true;
	/*
	Point2f srcTri[3];
	Point2f dstTri[3];

	Mat warpMat( 2, 3, CV_32FC1 );
	Mat warp_dst = Mat::zeros( height, width, rotatedImg.type() );

	srcTri[0] = Point2f(rightEye.x , rightEye.y);
	srcTri[1] = Point2f(leftEye.x, leftEye.y);
	int distance = leftEye.x - rightEye.x;
	srcTri[2] = Point2f(rightEye.x, rightEye.y + distance);

	dstTri[0] = desiredRightPos;
	dstTri[1] = desiredLeftPos;
	dstTri[2] = Point2f( desiredRightPos.x, desiredRightPos.y + (desiredLeftPos.x - desiredRightPos.y));


	cout << "DESIRED left" << desiredLeftPos.x << desiredLeftPos.y << endl;
	warpMat = getAffineTransform( srcTri, dstTri );
	warpAffine( rotatedImg, warp_dst, warpMat, warp_dst.size() );

	cout << "Rows:" << warp_dst.rows << "  " << warp_dst.cols << endl;

	imshow("face_recognizer", warp_dst);

	waitKey(5000);
	*/
}

bool FaceDetection::findEye(vector<Rect_<int> >& eyes, Rect_<int>& found,bool left)
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

Point FaceDetection::findCenter(Rect_<int> rect)
{
	int x = rect.x + rect.width/2;
	int y = rect.y + rect.height/2;
	return Point(x, y);
}


Mat FaceDetection::rotateImage(const Mat& source, Point2f pivot, double angle)
{
//    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    //Point2f src_center(source.cols, source.rows);
    Mat rot_mat = getRotationMatrix2D(pivot, angle *57.2957795, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size());
    return dst;
}


void findAllFilesInDir(string dir, vector<string>& files)
{
	DIR *dp;
    struct dirent *dirp;

	if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error opening " << dir << endl;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        string file = string(dirp->d_name);
        files.push_back(file);
    }
    closedir(dp);

}

int main(int argc, const char *argv[]) {


    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 3) {
        cout << "usage: " << argv[0] << " <action predict|normalize>" << endl;
        cout << "\t " << argv[0] << " predict </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
        cout << "\t " << argv[0] << " normalize </path/to/inputDir> </path/to/outDir> -- Normalize all the images in the inputDir and add them to the output dir" << endl;
        exit(1);
    }

    cout << argv[1] << endl;
    if(string(argv[1]).compare("normalize") == 0)
    {
    	cout << "Normalizing..." << endl;

    	string inputDir = string(argv[2]);
    	string outDir = string(argv[3]);

    	vector<string> files;
    	findAllFilesInDir(inputDir, files);

        FaceDetection detect;

        for(uint i=0; i<files.size(); ++i)
        {
            string file = files.at(i);
            if(file.find(".jpg") == string::npos )
            {
            	continue;
            }

            cout << "Normalizing " << file << endl;

            Mat img = imread(inputDir + "/" + file);
            bool normalized = detect.findAndNormalizeFace(img);
            if(normalized)
            {
            	imwrite(outDir+"/"+file, img);
            }
        }

    	return 0;
    }
    else if(string(argv[1]).compare("predict") == 0)
    {
		FaceDetection detect;
		string imageDir = string(argv[2]);
		vector<string> files;
    	findAllFilesInDir(imageDir, files);

		detect.initialize(imageDir, files);
		vector<string> testPaths;
		testPaths.push_back("/home/matt/workspace/FaceDetection/data/test/bush.jpg");
		testPaths.push_back("/home/matt/workspace/FaceDetection/data/test/mccain.jpg");
		testPaths.push_back("/home/matt/workspace/FaceDetection/data/test/paul.jpg");
		testPaths.push_back("/home/matt/workspace/FaceDetection/data/test/cal.jpg");

		Mat test;
		for(int i=0;i<testPaths.size(); ++i) {
			test = imread(testPaths.at(i));
			detect.predictFace(test);
		}
    }
    else
    {
    	cout << "Bad params" << endl;
    }
    return 0;
}
