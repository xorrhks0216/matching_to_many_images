/**
  @FileName : testMatching.cpp
  @Date : 9/26/2014
  @Author : Taekgwan Kim
  @Description :
    using input parameters, changing keypoint, descriptor, matching algorithms.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <stdio.h>
#include <iostream>

using namespace cv;

const string defaultDetectorType = "SIFT";
const string defaultDescriptorType = "SIFT";
const string defaultMatcherType = "BruteForce";
const string defaultQueryFileName = "../Debug/query.jpg";
const string defaultTrainFileName = "../Debug/train.png";

/** @function main */
static void readme(const string& app)
{
	std::cout << "Usage: \ttestMatching.exe\n";
	std::cout << "\ttestMatching.exe <Detector> <Descriptor> <Matcher>" << std::endl;
	std::cout << "\ttestMatching.exe <Detector> <Descriptor> <Matcher> <QueryPath> <TrainPath>" << std::endl << std::endl;
	
	std::cout << "Available Type of Detectors : \t\"FAST\", \"STAR\", \"SIFT\", \"SURF\",\n";
	std::cout << "\t\t\t\t\"ORB\", \"BRISK\", \"MSER\",\n";
	std::cout << "\t\t\t\t\"GFTT\", \"HARRIS\", \"Dense\", \"SimpleBlob\"" << std::endl;
	
	std::cout << "Available Type of Descriptors : \"SIFT\", \"SURF\",\n";
	std::cout << "\t\t\t\t\"BRIEF\", \"BRISK\", \"ORB\", \"FREAK\"" << std::endl;
	
	std::cout << "Available Type of Matcher : \t\"BruteForce\", \"BruteForce-L1\",\n";
	std::cout << "\t\t\t\t\"BruteForce-Hamming\", \"BruteForce-Hamming(2)\",\n";
	std::cout << "\t\t\t\t\"FlannBased\"\n";
}

/** @function main */
static void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask )
{
    mask.resize( matches.size() );
    fill( mask.begin(), mask.end(), 0 );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        if( matches[i].imgIdx == trainImgIdx )
            mask[i] = 1;
    }
}

/** @function main */
bool createDetectorDescriptorMatcher(const string& detectorType, const string& descriptorType, const string& matcherType,
	Ptr<FeatureDetector>& featureDetector,
	Ptr<DescriptorExtractor>& descriptorExtractor,
	Ptr<DescriptorMatcher>& descriptorMatcher)
{
	std::cout << "< Creating feature detector, descriptor extractor and descriptor matcher ...";
	featureDetector = FeatureDetector::create(detectorType);
	descriptorExtractor = DescriptorExtractor::create(descriptorType);
	descriptorMatcher = DescriptorMatcher::create(matcherType);
	std::cout << ">" << std::endl;

	bool isCreated = !(featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty());
	if(!isCreated)
		std::cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types.>" << std::endl;

	return isCreated;
}

/** @function main */
bool readOneImage(const string& fileName, Mat& image)
{
	image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
	if(image.empty())
	{
		std::cout << "image " << fileName << " can not be read." << std::endl;
		return false;
	}
	std::cout << "image " << fileName << " was read." << std::endl;
	return true;
}

/** @function main */
void detectKeypoints(const Mat& image, vector<KeyPoint>& keyPoints, Ptr<FeatureDetector>& featureDetector)
{
	std::cout << "< Extracting keypoints from image... ";
	featureDetector->detect(image, keyPoints);
	std::cout << ">" << std::endl;
}

void detectKeypoints(const vector<Mat>& image, vector<vector<KeyPoint>>& keyPoints, Ptr<FeatureDetector>& featureDetector)
{
	std::cout << "< Extracting keypoints from image... ";
	featureDetector->detect(image, keyPoints);
	std::cout << ">" << std::endl;
}

/** @function main */
void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
                         const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
                         Ptr<DescriptorExtractor>& descriptorExtractor )
{
    std::cout << "< Computing descriptors for keypoints...>" << std::endl;
    descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
    descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

    int totalTrainDesc = 0;
    for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
        totalTrainDesc += tdIter->rows;

    std::cout << "Query descriptors count: " << queryDescriptors.rows << "; Total train descriptors count: " << totalTrainDesc << std::endl;
}

void computeDescriptors(const Mat& image, vector<KeyPoint>& keyPoints, Mat& descriptors, Ptr<DescriptorExtractor>& descriptorExctractor)
{
	std::cout << "< Computing descriptors for image keypoints..." << std::endl;
	descriptorExctractor->compute(image, keyPoints, descriptors);
	std::cout << "image descriptor count : " << descriptors.rows;
	std::cout << ">" << std::endl;
}

/** @function main */
void matchDescriptors(const Mat& queryDescriptors, const vector<Mat>& trainDescriptors,
					vector<DMatch>& matches, Ptr<DescriptorMatcher> descriptorMatcher)
{
	std::cout << "< Set object descriptors collection in the matcher and match query descriptors to them..." << std::endl;
	descriptorMatcher->add(trainDescriptors);
	descriptorMatcher->train();

	descriptorMatcher->match(queryDescriptors, matches);
	CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

	std::cout << "Number of matches: " << matches.size();
	std::cout << ">" << std::endl;
}

void matchDescriptors(const Mat& queryDescriptors, const Mat& trainDescriptors,
					vector<DMatch>& matches, Ptr<DescriptorMatcher> descriptorMatcher)
{
	std::cout << "< Match query descriptors to object descriptor in matcher..." << std::endl;
	descriptorMatcher->match(queryDescriptors, trainDescriptors, matches);
	CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

	std::cout << "Number of matches: " << matches.size();
	std::cout << ">" << std::endl;
}

Mat showResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
						const vector<Mat>& trainImages, const vector<vector<KeyPoint>>& trainKeypoints,
						const vector<DMatch>& matches )
{
	std::cout << "< Show results...";
	Mat drawImg;

	vector<char> mask;
	const int x = 0;
	vector<int> hist((int)trainImages.size(), x);
	for(int i = 0; i < (int)matches.size(); i++)
	{
		hist.at(matches.at(i).imgIdx)++;
	}
	int max = 0;
	int maxIdx = 0;
	for(int i = 0; i < (int)hist.size(); i++)
	{
		if(hist.at(i) > max)
		{
			max = hist.at(i);
			maxIdx = i;
		}
		else continue;
	}

	maskMatchesByTrainImgIdx( matches, max, mask );

	drawMatches( queryImage, queryKeypoints, trainImages.at(maxIdx), trainKeypoints.at(maxIdx),
		matches, drawImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	std::cout << ">" << std::endl;
	return drawImg;
}

Mat showResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
						const Mat& trainImages, const vector<KeyPoint>& trainKeypoints,
						const vector<DMatch>& matches )
{
	std::cout << "< Show results...";
	Mat drawImg;

	drawMatches( queryImage, queryKeypoints, trainImages, trainKeypoints,
		matches, drawImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	std::cout << ">" << std::endl;
	return drawImg;
}

/** @function main */
int main(int argc, char** argv)
{
	TickMeter tm;

	string detectorType = defaultDetectorType;
	string descriptorType = defaultDescriptorType;
	string matcherType = defaultMatcherType;
	string queryFileName = defaultQueryFileName;
	string trainFileName = defaultTrainFileName;
	
	if(argc != 1 && argc != 4 && argc != 6)
	{
		readme(argv[0]);
		return -1;
	}
	std::cout << argc << std::endl;
	if(argc != 1)
	{

		detectorType = argv[1];
		descriptorType = argv[2];
		matcherType = argv[3];
		if(argc != 4)
		{
			queryFileName = argv[4];
			trainFileName = argv[5];
		}
	}

	Mat trainImage = imread(trainFileName, CV_LOAD_IMAGE_GRAYSCALE);
	Mat queryImage = imread(queryFileName, CV_LOAD_IMAGE_GRAYSCALE);

	if(!trainImage.data || !queryImage.data)
	{
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}

	//Create Detector Phase
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;
	initModule_nonfree();
	if(!createDetectorDescriptorMatcher(detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher))
	{
		readme(argv[0]);
		return -1;
	}
	
	//get keypoints phase
	vector<KeyPoint> queryKeypoints;
	vector<KeyPoint> trainKeypoints;
	tm.start();
	detectKeypoints(queryImage, queryKeypoints, featureDetector);
	detectKeypoints(trainImage, trainKeypoints, featureDetector);
	tm.stop();
	double keypointTime = tm.getTimeMilli();

	//get descriptor phase
	Mat queryDescriptors;
	Mat trainDescriptors;
	tm.start();
	computeDescriptors(queryImage, queryKeypoints, queryDescriptors, descriptorExtractor);
	//computeDescriptors(trainImage, trainKeypoints, trainDescriptors, descriptorExtractor);
	cv::FileStorage fs2("data.xml", cv::FileStorage::READ);
	fs2["trainDescriptors"] >> trainDescriptors;
	fs2.release();
	tm.stop();
	double descriptorTime = tm.getTimeMilli();

	//matching Phase
	vector<DMatch> matches;
	tm.start();
	matchDescriptors(trainDescriptors, queryDescriptors, matches, descriptorMatcher);
	tm.stop();
	double matcherTime = tm.getTimeMilli();
	
	//show result Phase
	double max_dist = 0; double min_dist = 100;

	for( int i = 0; i < trainDescriptors.rows; i++ )
	{
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	vector<DMatch> good_matches;

	for( int i = 0; i < trainDescriptors.rows; i++ )
	{
		if( matches[i].distance < 3*min_dist )
		{
			good_matches.push_back( matches[i]);
		}
	}

	Mat matchesImage = showResultImages(trainImage, trainKeypoints, queryImage, queryKeypoints, matches);
	Mat goodmatchesImage = showResultImages(trainImage, trainKeypoints, queryImage, queryKeypoints, good_matches);
	namedWindow("matches", 1);
	namedWindow("good_matches", 1);
	imshow("matches", matchesImage);
	imshow("good_matches", goodmatchesImage);

	std::cout << "\n" << detectorType << " + " << descriptorType << " + " << matcherType << std::endl;
	std::cout << "detecting time : " << keypointTime << " ms" << std::endl;
	std::cout << "computing descriptor time : " << descriptorTime << " ms" << std::endl;
	std::cout << "matching time : " << matcherTime << " ms" << std::endl;
	std::cout << "total time : " << keypointTime + descriptorTime + matcherTime << " ms" << std::endl;

	cv::FileStorage fs1("data.xml", cv::FileStorage::WRITE);
	fs1 << "trainDescriptors" << trainDescriptors;
	fs1.release();

	cvWaitKey(0);

	return 0;
}