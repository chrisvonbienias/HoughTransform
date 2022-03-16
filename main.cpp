#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hough.h"
#include "hough_optim.h"

using namespace cv;
using namespace std;

int main() {

	// Timer
	clock_t start, end;

	// input image
	Mat img = imread("2021_image.jpg", 1);
	// circles container
	vector<Vec3f> circles;

	Mat imgCircles(img.size(), CV_8U);
	cvtColor(img, imgCircles, COLOR_BGR2GRAY);

	// Gaussian blur
	GaussianBlur(imgCircles, imgCircles, Size(3, 3), 1);

	start = clock();
	// Non-optimized functions
	// OtsuThreshold(imgCircles, imgCircles);
	// ScharrEdge(imgCircles, imgCircles);
	// HoughTransform(imgCircles, circles, 25, 27, 140);

	// Optimized functions
	OtsuThresholdOptim(imgCircles, imgCircles);
	ScharrEdgeOptim(imgCircles, imgCircles);
	HoughTransformOptim(imgCircles, circles, 25, 27, 140);

	// Benchmark function
	// HoughCircles(imgCircles, circles, HOUGH_GRADIENT, 1, imgCircles.rows / 16, 100, 20, 25, 27);

	end = clock();

	// drawing cicles
	for (size_t i = 0; i < circles.size(); i++) {

		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		int radius = c[2];
		circle(img, center, radius, Scalar(255, 000, 0), 2, LINE_AA);

	}
	
	printf("Time: %0.8f sec\n", ((float)end - start) / CLOCKS_PER_SEC);
	imshow("Original", img);
	waitKey(0);

	return 0;
}