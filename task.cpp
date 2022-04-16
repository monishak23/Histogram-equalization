#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\imgproc.hpp>

using namespace cv;

class Histogram {
public:
	Mat calc_histogram(Mat scr) {
		Mat hist;
		hist = Mat::zeros(256, 1, CV_32F);
		scr.convertTo(scr, CV_32F);
		double value = 0;
		for (int i = 0; i < scr.rows; i++)
		{
			for (int j = 0; j < scr.cols; j++)
			{
				value = scr.at<float>(i, j);
				hist.at<float>(value) = hist.at<float>(value) + 1;
			}
		}
		return hist;
	}

	void plot_histogram(Mat histogram,const char *name) {
		Mat histogram_image(400, 512, CV_8UC3, Scalar(0, 0, 0));
		Mat normalized_histogram;
		normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());

		for (int i = 0; i < 256; i++)
		{
			rectangle(histogram_image, Point(2 * i, histogram_image.rows - normalized_histogram.at<float>(i)),
				Point(2 * (i + 1), histogram_image.rows), Scalar(255, 0, 0));
		}

		namedWindow(name, WINDOW_AUTOSIZE);
		imshow(name, histogram_image);
	}
};

void main() {

	Mat img,weighted_hist;
	img = imread("C:\\Users\\Monisha\\Desktop\\cernue\\task\\images.jfif", 0);
	imshow("original image", img);


	Histogram H1,H2;
	Mat hist = H1.calc_histogram(img);
	H1.plot_histogram(hist,"ORIGINAL_HISTOGRAM");

	weighted_hist = hist / sum(hist);

	// calculate cumulative histogram
	Mat acc_hist = Mat::zeros(weighted_hist.size(), weighted_hist.type());
	acc_hist.at<float>(0) = weighted_hist.at<float>(0);
	for (int i = 1; i < 256; i++)
	{
		acc_hist.at<float>(i) = weighted_hist.at<float>(i) + acc_hist.at<float>(i - 1);
	}
	acc_hist = acc_hist * 255;

	Mat imgClone = Mat::zeros(img.size(), CV_32FC1);
	img.convertTo(imgClone, CV_32FC1);
	Mat output = Mat::zeros(img.size(), CV_32FC1);
	for (int m = 0; m < img.rows; m++)
	{
		for (int n = 0; n < img.cols; n++)
		{
			output.at<float>(m, n) = acc_hist.at<float>(imgClone.at<float>(m, n));
		}
	}
	output.convertTo(output, CV_8UC1);
	imshow("equalized image", output);

	Mat new_hist = H2.calc_histogram(output);
	H2.plot_histogram(new_hist,"NEW_HISTOGRAM");


	waitKey(0);

}