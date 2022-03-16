// Non-optimized functions for Hough transform

#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

// Scharr edge detection
void ScharrEdge(const Mat& image, Mat& output);

// Otsu thresholding algorithm
void OtsuThreshold(const Mat& image, Mat& output);

void HoughTransform(const Mat& image, vector<Vec3f>& circles, int min_r, int max_r, int threshold);