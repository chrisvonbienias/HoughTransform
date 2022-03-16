#include "hough_optim.h"

// Lock for multithreading
mutex m;

// Dummy function for multithreading
// for some reason I can't pass filter2D directly to thread
void Filter(const Mat& image, Mat& output, int param, Mat& kernel)
{
	filter2D(image, output, param, kernel);
}

void ScharrEdgeOptim(const Mat& image, Mat& output)
{
	Mat kernelX, kernelY, scharrX, scharrY;

	float scharrx_data[9] = { -3, 0, 3, -10, 0, 10, -3, 0, 3 };
	float scharry_data[9] = { -3, -10, -3, 0, 0, 0, 3, 10, 3 };

	kernelX = Mat(3, 3, CV_32F, scharrx_data);
	kernelY = Mat(3, 3, CV_32F, scharry_data);

	thread t1(Filter, ref(image), ref(scharrX), -1, ref(kernelX));
	thread t2(Filter, ref(image), ref(scharrY), -1, ref(kernelY));
	t1.join();
	t2.join();

	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			auto x = scharrX.at<uchar>(i, j);
			auto y = scharrY.at<uchar>(i, j);

			float g = hypot(x, y);
			output.at<uchar>(i, j) = (uchar)g;
		}
	}
}

void OtsuThresholdOptim(const Mat& image, Mat& output)
{
	const int N = image.cols * image.rows;

	float threshold, var_max, sum, sumB, q1, q2, u1, u2, variance;
	threshold = var_max = sum = sumB = q1 = q2 = u1 = u2 = 0;
	const int max_intensity = 255;

	// initialize histogram
	int histogram[255];
	for (size_t i = 0; i < max_intensity; ++i)
	{
		histogram[i] = 0;
	}

	// calculate histogram
	for (size_t i = 0; i < max_intensity; ++i)
	{
		auto value = image.at<uchar>(i);
		histogram[value] += 1;
	}

	// aux value for u2 calculation
	for (size_t i = 0; i < max_intensity; ++i)
	{
		sum += i * histogram[i];
	}

	// otsu threshold calculation
	for (size_t i = 0; i < max_intensity; ++i)
	{
		q1 += histogram[i];

		if (q1 == 0)
		{
			continue;
		}
		else {

			q2 = N - q1;

			sumB += i * histogram[i];
			u1 = sumB / q1;
			u2 = (sum - sumB) / q2;

			variance = q1 * q2 * powf(u1 - u2, 2);

			if (variance > var_max)
			{
				threshold = i;
				var_max = variance;
			}
		}
	}

	for (size_t i = 0; i < image.rows; ++i)
	{
		for (size_t j = 0; j < image.cols; ++j)
		{
			if (image.at<uchar>(i, j) > threshold)
			{
				output.at<uchar>(i, j) = 255;
			}
			else {
				output.at<uchar>(i, j) = 0;
			}
		}
	}
}

// Dummy function for multithreading
void FindCircle(const Mat& image, vector<Vec3f>& circles, int radius, int threshold, float* sin_table, float* cos_table)
{
	int rows = image.rows;
	int cols = image.cols;

	Mat acc = Mat::zeros(image.size(), CV_8U);

	// iterating through image and accumulating
	for (size_t x = 0; x < rows; x++)
	{
		for (size_t y = 0; y < cols; y++)
		{
			if (image.at<uchar>(x, y) == 255) // edge
			{
				for (size_t angle = 0; angle < 360; angle++)
				{
					int a = x - radius * sin_table[angle];
					int b = y - radius * cos_table[angle];

					if (a >= 0 && a < rows && b >= 0 && b < cols)
					{
						acc.at<uchar>(a, b) += 1;
					}
				}
			}
		}
	}

	double min, max;
	minMaxLoc(acc, &min, &max);

	if (max > threshold)
	{
		// initial threshold
		for (size_t x = 0; x < rows; x++)
		{
			for (size_t y = 0; y < cols; y++)
			{
				if (acc.at<uchar>(x, y) < threshold)
				{
					acc.at<uchar>(x, y) = 0;
				}
			}
		}

		// finding circles
		for (size_t i = 1; i < rows - 1; i++)
		{
			for (size_t j = 1; j < cols - 1; j++)
			{
				if (acc.at<uchar>(i, j) != 0)
				{
					m.lock();
					circles.push_back(Vec3f(j, i, radius));
					m.unlock();

					// purge neighborhood
					acc.at<uchar>(i - 1, j - 1) = 0;
					acc.at<uchar>(i - 1, j) = 0;
					acc.at<uchar>(i - 1, j + 1) = 0;
					acc.at<uchar>(i, j - 1) = 0;
					acc.at<uchar>(i, j + 1) = 0;
					acc.at<uchar>(i + 1, j - 1) = 0;
					acc.at<uchar>(i + 1, j) = 0;
					acc.at<uchar>(i + 1, j + 1) = 0;
				}
			}
		}
	}
	// resetting the accumulator
	acc = Mat::zeros(image.size(), CV_8U);
}

void HoughTransformOptim(const Mat& image, vector<Vec3f>& circles, int min_r, int max_r, int threshold)
{
	// angle sin and cos look-up tables
	float sin_table[360];
	float cos_table[360];
	for (size_t angle = 0; angle < 360; angle++)
	{
		float rad = (angle + M_PI) / 180;
		sin_table[angle] = sinf(rad);
		cos_table[angle] = cosf(rad);
	}
	
	// Searching for all unkonown radius'
	for (size_t r = min_r; r < max_r; r += 2)
	{
		thread t1(FindCircle, ref(image), ref(circles), r, threshold, sin_table, cos_table);
		thread t2(FindCircle, ref(image), ref(circles), r + 1, threshold, sin_table, cos_table);
		t1.join();
		t2.join();
	}
}
