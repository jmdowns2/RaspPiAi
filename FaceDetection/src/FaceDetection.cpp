#include "FaceDetection.h"
#include "NormalizeImage.h"
#include "math.h"
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include "FileUtils.h"

NormalizeImage createNormalizeImage()
{
	return NormalizeImage(true, 800, 0.3, 100);
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

        if(file.find(".jpg") == string::npos && file.find(".pgm") == string::npos )
        {
        	continue;
        }


    	Mat temp = imread(path+"/"+file, 0);

        images.push_back(temp);
        labels.push_back(iFile);
    }


    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    width = images[0].cols;
    height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    model = createFisherFaceRecognizer();
    //model = createEigenFaceRecognizer(6, 0.0);
    //model = createLBPHFaceRecognizer();
    model->train(images, labels);

    cout << "Initialized" << endl;
}

int FaceDetection::predictFace(Mat& img)
{
	cout << "Predicting..." << endl;

	NormalizeImage imgNorm = createNormalizeImage();
	if(imgNorm.init() < 0)
		return -1;


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

		Mat face_resized;
		face_resized = imgNorm.normalize(img);

		if(face_resized.empty())
		{
			cout << " No face " << endl;
			continue;
		}
		else
		{
			cout << " A face was found " << endl;
		}


		// Now perform the prediction, see how easy that is:
		prediction = model->predict(face_resized);
		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(original, face_i, CV_RGB(0, 255,0), 1);
		// Create the text we will annotate the box with:

		string box_text = "Prediction = " + this->files.at(prediction);

		cout << "Predicting " << box_text << endl;

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

    	vector<FileInfo*> fileVec;
    	FileUtils fileUtils;
    	fileUtils.findFilesInDir(inputDir, fileVec);

        FaceDetection detect;

    	NormalizeImage imgNorm = createNormalizeImage();
    	if(imgNorm.init() < 0)
    		return -1;


        for(uint i=0; i<fileVec.size(); ++i)
        {
            FileInfo* file = fileVec.at(i);
            if(file->fileName.find(".jpg") == string::npos && file->fileName.find(".pgm") == string::npos)
            {
            	continue;
            }

            cout << "Normalizing " << file->path << "/" << file->fileName << endl;


            Mat img = imread(file->path + "/" + file->fileName);

            Mat normalized = imgNorm.normalize(img);
            if(!normalized.empty())
            {
            	string outName = file->path;
            	fileUtils.replace(outName, ".", "");
            	fileUtils.replace(outName, "/", "_");

            	imwrite(outDir+outName+"_"+file->fileName, normalized);
            }

            /*
            bool normalized = detect.findAndNormalizeFace(img);
            if(normalized)
            {
            	string outName = file->path;
            	fileUtils.replace(outName, ".", "");
            	fileUtils.replace(outName, "/", "_");

            	imwrite(outDir+outName+"_"+file->fileName, img);
            }
            */
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
		testPaths.push_back("/home/matt/workspace/RaspPiAi/FaceDetection/data/test/bush.jpg");
		testPaths.push_back("/home/matt/workspace/RaspPiAi/FaceDetection/data/test/mccain.jpg");
		testPaths.push_back("/home/matt/workspace/RaspPiAi/FaceDetection/data/test/paul.jpg");
		testPaths.push_back("/home/matt/workspace/RaspPiAi/FaceDetection/data/test/cal.jpg");

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
