// Optimized functions for Hough transform

#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <math.h>
#include <thread>
#include <mutex>

using namespace std;
using namespace cv;

// Optimized Scharr edge detection
void ScharrEdgeOptim(const Mat& image, Mat& output);

// Optimized Otsu thresholding algorithm
void OtsuThresholdOptim(const Mat& image, Mat& output);

// Optimized Hough Transform algorithm
void HoughTransformOptim(const Mat& image, vector<Vec3f>& circles, int min_r, int max_r, int threshold);