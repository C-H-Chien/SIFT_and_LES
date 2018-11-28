// lowpasstest.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <atltime.h>
#include <stdio.h>
#include "math.h"

#define M_PI 3.14159265358979323846
#define round(x) ((int)(x + 0.5))
#define DESCRIPTOR_SIZE 128
#define OCTAVE 4	//組數
#define LAYER 6		//層數
#define nOctaveLayers 3
#define DOG_LAYER 5
#define SIFT_SIGMA 1.6

/** default number of sampled intervals per octave */
#define SIFT_INTVLS 3

/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA 1.6

/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR 0.04

/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR 10

/** double image size before pyramid construction? */
#define SIFT_IMG_DBL 1

/** default width of descriptor histogram array */
#define SIFT_DESCR_WIDTH 4

/** default number of bins per histogram in descriptor array */
#define SIFT_DESCR_HIST_BINS 8

/* assumed gaussian blur for input image */
#define SIFT_INIT_SIGMA 0.5

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 0.8

/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0

#define SIFT_FEATURE_VECTOR 128

#define FEATURE_LOWE_COLOR CV_RGB(255,0,0)

#define OCTAVE 1	//組數

#define LAYER 4		//層數

#define nOctaveLayers 1

#define DOG_LAYER 3

#define ABS(x) ( ( (x) < 0 )? -(x) : (x) )

#define MAX_(a,b)  ((a) < (b) ? (b) : (a))

typedef struct
{
	///////////////////////////////////
	//特徵點實際資料
	double x;
	double y;
	double scl;
	double angle;

	///////////////////////////////////
	//偵測到資料
	int r;
	int c;
	int octv;
	int intvl;
	double subintvl;
	double scl_octv;
	double sum;
	///////////////////////////////////
	//特徵點描述
	double descr[SIFT_FEATURE_VECTOR];

}key_point;


using namespace cv;

typedef struct
{
	int data;
	int magnitude;
	int orientation;
	int descriptor[DESCRIPTOR_SIZE];
	int n_descriptor[DESCRIPTOR_SIZE];
}gradient;

vector <key_point> sift_algorithm(Mat image);

Mat GaussFilter(Mat Image, double sigma, int mask_size, int image_heigh, int image_width, int image_pixel);
int* GaussMask(double sigma, int mask_size);
uchar get_pixel_32f(Mat& img, int r, int c);
Mat SIFTDetection(Mat image_pre, Mat image_mid, Mat image_next, int image_heigh, int image_width, int* keypoint_num);
bool ExtremaDetection(Mat image_pre, Mat image_mid, Mat image_next, int image_width, int i, int j);
bool UnstableKeypointDetect(Mat image_pre, Mat image_mid, Mat image_next, int rows, int cols, int size);
bool Inverse_3x3_Matrix (int** m, int** adj, int* det);
int LowContastMatrixOperation(int ** adj, int *diff);
void GaussianPyramid_parallel(Mat base, Mat* gpyr);
void buildGaussianPyramid(Mat base, Mat* gpyr);
void buildDoGPyramid(Mat* gpyr, Mat* dogpyr);

gradient* image_descriptor(Mat Gaussian_image, int image_heigh, int image_width, int image_pixel);
void gradient_calculation(gradient* image, int image_heigh, int image_width);
double image_gradient_magnitude(double dx, double dy);
double image_gradient_orientation(double dx, double dy);
void histogram_descriptor(gradient* image, int image_heigh, int image_width);
void normalization(gradient* image, int image_pixel);

Mat combination_image(Mat& img1, Mat& img2);
vector <key_point> get_match_data(Mat image_result, gradient* image, int image_heigh, int image_width);
void euclidean_distance(vector<key_point>& kpt1, vector<key_point>& kpt2, Mat trans, Mat& match_img, int width);
double* min_data_detector(double* distance_data, int size);
int* bubble_sort(double* original_data, int size);
void sumvalue_detec(vector<key_point>& kpt1, vector<key_point>& kpt2, Mat& match_img);

void save_picture_to_txt(Mat image, char* file);
vector <key_point> get_hardware_feature_descriptor(char* detection_file, int image_heigh, int image_width, int size);

int main()
{
	Mat image1, image2, input_image, match_image, image1_rgb, image2_rgb;

	char image_name1[] = "7.bmp";
	char image_name2[] = "a7.bmp";

	image1_rgb = imread(image_name1, 1);
	image2_rgb = imread(image_name2, 1);

	//Mat a;
	//resize(image1_rgb, a, Size(800, 480));
	//imwrite("11.bmp", a);

	Point2f AffinePoints0[3] = { Point2f(0, 0), Point2f(800, 0), Point2f(0, 480)};
	Point2f AffinePoints1[3] = { Point2f(50, 90), Point2f(750, 90), Point2f(60, 410) };
	Mat Trans = getAffineTransform(AffinePoints0, AffinePoints1);
	warpAffine(image1_rgb, image2_rgb, Trans, Size(image1_rgb.cols, image1_rgb.rows), CV_INTER_CUBIC);

	cvtColor(image1_rgb, image1, CV_BGR2GRAY);
	cvtColor(image2_rgb, image2, CV_BGR2GRAY);

	char save_file_name1[] = "image_gray1.txt";
	char save_file_name2[] = "image_gray2.txt";
	save_picture_to_txt(image1, save_file_name1);
	save_picture_to_txt(image2, save_file_name2);

	vector <key_point> kpt1, kpt2;

	input_image = image1.clone();
	kpt1 = sift_algorithm(input_image);
	input_image = image2.clone();
	kpt2 = sift_algorithm(input_image);

	FILE *pfile1 = fopen("image1_descriptors.txt", "wb");
	FILE *pfile2 = fopen("image2_descriptors.txt", "wb");

	double p;
	for (int i = 0; i < kpt1.size(); i++)
	{
		p = i;
		fprintf(pfile1, "%f\t", p);
		for (int j = 0; j < 128; j++)
		{
			p = kpt1[i].descr[j];
			fprintf(pfile1, "%f\t", p);
		}
		fprintf(pfile1, "\n", p);
	}
	fclose(pfile1);

	for (int i = 0; i < kpt2.size(); i++)
	{
		p = i;
		fprintf(pfile2, "%f\t", p);
		for (int j = 0; j < 128; j++)
		{
			p = kpt2[i].descr[j];
			fprintf(pfile2, "%f\t", p);
		}
		fprintf(pfile2, "\n", p);
	}
	fclose(pfile2);

	//char file_name1[] = "image1-1.txt";
	//char file_name2[] = "image1-2.txt";
	//kpt1 = get_hardware_feature_descriptor(file_name1, 480, 800, 1846);
	//kpt2 = get_hardware_feature_descriptor(file_name2, 480, 800, 1199);

	printf("size = %d\n", kpt1.size());
	printf("size = %d\n", kpt2.size());

	int x, y;

	for (int i = 0; i < kpt1.size(); i++)
	{
		x = int (kpt1[i].x);
		y = int (kpt1[i].y);
		image1_rgb.data[(y * image1_rgb.cols * 3) + (x * 3)] = 0;
		image1_rgb.data[(y * image1_rgb.cols * 3) + (x * 3 + 1)] = 0;
		image1_rgb.data[(y * image1_rgb.cols * 3) + (x * 3 + 2)] = 255;
	}

	for (int i = 0; i < kpt2.size(); i++)
	{
		x = int(kpt2[i].x);
		y = int(kpt2[i].y);
		image2_rgb.data[(y * image2_rgb.cols * 3) + (x * 3)] = 0;
		image2_rgb.data[(y * image2_rgb.cols * 3) + (x * 3 + 1)] = 0;
		image2_rgb.data[(y * image2_rgb.cols * 3) + (x * 3 + 2)] = 255;
	}

	imshow("image1", image1_rgb);
	imshow("image2", image2_rgb);

	waitKey(20);

	match_image = combination_image(image1_rgb, image2_rgb);
	euclidean_distance(kpt1, kpt2, Trans, match_image, image1.cols);
	namedWindow("match", WINDOW_NORMAL);
	imwrite("matching.bmp", match_image);
	imshow("match", match_image);

	waitKey(0);

	return 0;
}

vector <key_point> sift_algorithm(Mat image)
{
	int image_width, image_heigh, image_pixel;
	image_width = image.cols;
	image_heigh = image.rows;
	image_pixel = image_heigh * image_width;

	Mat *gpyr, *dogpyr;
	gpyr = new Mat[OCTAVE * LAYER];
	dogpyr = new Mat[OCTAVE * DOG_LAYER];

	gpyr[0] = GaussFilter(image, 1.2, 7, 480, 800, 800 * 480);
	gpyr[1] = GaussFilter(image, 1.15, 7, 480, 800, 800 * 480);
	gpyr[2] = GaussFilter(image, 1.295, 7, 480, 800, 800 * 480);
	gpyr[3] = GaussFilter(image, 1.62, 7, 480, 800, 800 * 480);

	//GaussianPyramid_parallel(image, gpyr);
	//buildGaussianPyramid(image, gpyr);
	buildDoGPyramid(gpyr, dogpyr);
	
	Mat image_pre, image_mid, image_next;
	image_pre = dogpyr[0];
	image_mid = dogpyr[1];
	image_next = dogpyr[2];

	int SW_keypoint_num = 0;
	int *get_keypoint_num;
	Mat image_detection;

	get_keypoint_num = &SW_keypoint_num;
	image_detection = SIFTDetection(image_pre, image_mid, image_next, image_heigh, image_width, get_keypoint_num);

	printf("keypoint_num = %d\n", *get_keypoint_num);

	gradient* img_description;
	img_description = image_descriptor(gpyr[1], image_heigh, image_width, image_pixel);

	delete [] gpyr;
	delete [] dogpyr;

	vector <key_point> kpt;
	key_point kpt_tmp;

	kpt = get_match_data(image_detection, img_description, image_heigh, image_width);

	delete [] img_description;
	return kpt;
}

Mat GaussFilter(Mat Image, double sigma, int mask_size, int image_heigh, int image_width, int image_pixel)
{
	int*D_GaussMask;
	D_GaussMask = GaussMask(sigma, mask_size);

	int tmp_value;
	int i, j, g, h, k;
	Mat GaussImage = Mat(image_heigh, image_width, CV_8UC1);

	for (i = 0; i < image_pixel; i++)
	{
		GaussImage.data[i] = 0;
	}

	for (i = 3; i < image_heigh - 3; i++)
	{
		for (j = 3; j < image_width - 3; j++)
		{
			k = 0;
			tmp_value = 0;
			for (g = -3; g <= 3; g++)
			{
				for (h = -3; h <= 3; h++)
				{
					tmp_value += ((int)Image.data[(j + h) + (i + g)*image_width] * D_GaussMask[k]);
					k++;
				}
			}
			GaussImage.data[j + i*image_width] = tmp_value / 1024;
		}
	}
	return GaussImage;
}

int* GaussMask(double sigma, int mask_size)
{
	int i, j, k;
	int kernel_mid;
	double GaussMaskSum = 0;

	double *GaussMask;
	GaussMask = (double*)malloc(sizeof(double) * mask_size * mask_size);

	int*D_GaussMask;
	D_GaussMask = (int*)malloc(sizeof(int) * mask_size * mask_size);

	kernel_mid = (mask_size / 2) + 1;

	k = 0;
	for (i = 1; i < (mask_size + 1); i++)
	{
		for (j = 1; j < (mask_size + 1); j++)
		{
			GaussMask[k] = exp(-((i - kernel_mid) * (i - kernel_mid) + (j - kernel_mid) * (j - kernel_mid)) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			GaussMaskSum += GaussMask[k++];
		}
	}

	for (i = 0; i < mask_size * mask_size; i++)
	{
		D_GaussMask[i] = round(GaussMask[i] * 1024 / GaussMaskSum);
	}

	return D_GaussMask;
}

Mat SIFTDetection(Mat image_pre, Mat image_mid, Mat image_next, int image_heigh, int image_width, int* keypoint_num)
{
	Mat image_result = Mat(image_heigh, image_width, CV_8UC1);
	for (int i = 1; i < (image_heigh - 1); i++)
	{
		for (int j = 1; j < (image_width - 1); j++)
		{
			if (ExtremaDetection(image_pre, image_mid, image_next, image_width, i, j))
			{
				if (UnstableKeypointDetect(image_pre, image_mid, image_next, i, j, image_width))
				{
					image_result.data[j + i*image_width] = 255;
					(*keypoint_num)++;
				}
				else
				{
					image_result.data[j + i*image_width] = 0;
				}
			}
			else
				image_result.data[j + i*image_width] = 0;
		}
	}
	return image_result;
}

bool ExtremaDetection(Mat image_pre, Mat image_mid, Mat image_next, int image_width, int i, int j)
{
	if (image_mid.data[j + i*image_width] >= image_pre.data[(j - 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[j + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[(j + 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[(j - 1) + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[j + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[(j + 1) + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[(j - 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[j + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_pre.data[(j + 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[(j - 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[j + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[(j + 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[(j - 1) + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[(j + 1) + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[(j - 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[j + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_mid.data[(j + 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[(j - 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[j + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[(j + 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[(j - 1) + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[j + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[(j + 1) + i*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[(j - 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[j + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] >= image_next.data[(j + 1) + (i + 1)*image_width])
		return 1;
	else if (image_mid.data[j + i*image_width] <= image_pre.data[(j - 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[j + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[(j + 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[(j - 1) + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[j + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[(j + 1) + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[(j - 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[j + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_pre.data[(j + 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[(j - 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[j + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[(j + 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[(j - 1) + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[(j + 1) + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[(j - 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[j + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_mid.data[(j + 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[(j - 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[j + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[(j + 1) + (i - 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[(j - 1) + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[j + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[(j + 1) + i*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[(j - 1) + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[j + (i + 1)*image_width] &&
		image_mid.data[j + i*image_width] <= image_next.data[(j + 1) + (i + 1)*image_width])
		return 1;
	else
		return 0;
}

bool UnstableKeypointDetect(Mat image_pre, Mat image_mid, Mat image_next, int i, int j, int size)
{
	int dx = 0;
	int dy = 0;
	int ds = 0;
	int dxx = 0;
	int dyy = 0;
	int dss = 0;
	int dxy = 0;
	int dxs = 0;
	int dys = 0;

	bool inverse_exist;
	int *d_diff;
	int **H, **H_adj;
	int H_det = 0;

	d_diff = new int[3];
	H = new int*[3];
	H_adj = new int*[3];
	for (int i = 0; i < 3; i++)
	{
		H[i] = new int[3];
		H_adj[i] = new int[3];
	}

	CvScalar H_trace;
	double D1;
	double D2;
	uchar keypoint_en;
	keypoint_en = 1;

	dx = floorf(double(image_mid.data[(j + 1) + i*size] - image_mid.data[(j - 1) + i*size]) / 2);

	dy = floorf(double(image_mid.data[j + (i + 1)*size] - image_mid.data[j + (i - 1)*size]) / 2);
	ds = floorf(double(image_next.data[j + i*size] - image_pre.data[j + i*size]) / 2);
	d_diff[0] = dx;
	d_diff[1] = dy;
	d_diff[2] = ds;

	dxx = floorf(double(image_mid.data[(j + 1) + i*size] + image_mid.data[(j - 1) + i*size]) - 2 * image_mid.data[j + i*size]);
	dyy = floorf(double(image_mid.data[j + (i + 1)*size] + image_mid.data[j + (i - 1)*size]) - 2 * image_mid.data[j + i*size]);
	dss = floorf(double(image_next.data[j + i*size] + image_pre.data[j + i*size]) - 2 * image_mid.data[j + i*size]);
	dxy = floorf((double(image_mid.data[(j + 1) + (i + 1)*size] - image_mid.data[(j - 1) + (i + 1)*size]) + double(image_mid.data[(j - 1) + (i - 1)*size] - image_mid.data[(j + 1) + (i - 1)*size])) / 4);
	dxs = floorf((double(image_next.data[(j + 1) + i*size] - image_next.data[(j - 1) + i*size]) + double(image_pre.data[(j - 1) + i*size] - image_pre.data[(j + 1) + i*size])) / 4);
	dys = floorf((double(image_next.data[j + (i + 1)*size] - image_next.data[j + (i - 1)*size]) + double(image_pre.data[j + (i - 1)*size] - image_pre.data[j + (i + 1)*size])) / 4);
	H[0][0] = dxx;
	H[0][1] = dxy;
	H[0][2] = dxs;
	H[1][0] = dxy;
	H[1][1] = dyy;
	H[1][2] = dys;
	H[2][0] = dxs;
	H[2][1] = dys;
	H[2][2] = dss;

	int matrix_value = 0;
	int counter = 0;

	inverse_exist = Inverse_3x3_Matrix(H, H_adj, &H_det);
	matrix_value = LowContastMatrixOperation(H_adj, d_diff);

	int a, b, c, left_value, right_value;

	a = image_mid.data[j + i*size] * image_mid.data[j + i*size] * H_det * H_det;
	b = image_mid.data[j + i*size] * matrix_value * H_det;
	c = matrix_value * matrix_value / 4;

	left_value = (a - b + c);
	right_value = H_det * H_det * 1;

	if (left_value > 10000)
	{
		left_value = (a - b + c);
		right_value = H_det * H_det * 1;
	}


	if (left_value <= right_value)
	{
		return 0;
	}

	int tr, det, edge_left, edge_right;
	int r = 10;

	tr = dxx + dyy;
	det = (dxx * dyy) - dxy * dxy;
	edge_left = tr * tr * r;
	edge_right = (r + 1) * (r + 1) * det;
	if ((det <= 0) || (edge_left >= edge_right))
	{
		return 0;
	}

	return 1;
}

bool Inverse_3x3_Matrix(int** m, int** adj, int* det)
{
	adj[0][0] = (m[1][1] * m[2][2]) - (m[1][2] * m[2][1]);
	adj[0][1] = (m[0][2] * m[2][1]) - (m[0][1] * m[2][2]);
	adj[0][2] = (m[0][1] * m[1][2]) - (m[0][2] * m[1][1]);
	adj[1][0] = (m[1][2] * m[2][0]) - (m[1][0] * m[2][2]);
	adj[1][1] = (m[0][0] * m[2][2]) - (m[0][2] * m[2][0]);
	adj[1][2] = (m[0][2] * m[1][0]) - (m[0][0] * m[1][2]);
	adj[2][0] = (m[1][0] * m[2][1]) - (m[1][1] * m[2][0]);
	adj[2][1] = (m[0][1] * m[2][0]) - (m[0][0] * m[2][1]);
	adj[2][2] = (m[0][0] * m[1][1]) - (m[0][1] * m[1][0]);

	*det = (m[0][0] * m[1][1] * m[2][2]) + (m[0][2] * m[1][0] * m[2][1]) + (m[0][1] * m[1][2] * m[2][0]) - (m[0][2] * m[1][1] * m[2][0]) - (m[0][0] * m[1][2] * m[2][1]) - (m[0][1] * m[1][0] * m[2][2]);

	if (*det == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

int LowContastMatrixOperation(int ** adj, int *diff)
{
	int* product_value;
	product_value = new int[3];
	int last_value;

	product_value[0] = diff[0] * adj[0][0] + diff[1] * adj[1][0] + diff[2] * adj[2][0];
	product_value[1] = diff[0] * adj[0][1] + diff[1] * adj[1][1] + diff[2] * adj[2][1];
	product_value[2] = diff[0] * adj[0][2] + diff[1] * adj[1][2] + diff[2] * adj[2][2];

	last_value = product_value[0] * diff[0] + product_value[1] * diff[1] + product_value[2] * diff[2];

	return last_value;
}

uchar get_pixel_32f(Mat& img, int r, int c)
{
	uchar * pData = img.data;

	return pData[r * img.cols + c];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void GaussianPyramid_parallel(Mat base, Mat* gpyr)
{
	double* sig;
	sig = new double[LAYER];	//記錄每組每層尺度

	double sig_prev, sig_total;
	double k = 1.28;

	sig[0] = SIFT_SIGMA;

	int index = 0;

	for (int i = 1; i < LAYER; i++)
	{
		sig_prev = pow(k, (double)(i - 1)) * SIFT_SIGMA;
		sig_total = sig_prev * k;
		sig[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);	//尺度座標
		printf("sig[%d] = %f\n", i, sig[i]);
	}

	for (int i = 0; i < OCTAVE * LAYER; i++)
	{
		index = i % 6;
		if (i < 6 || index != 0)
			GaussianBlur(base, gpyr[i], Size(7,7), sig[index]);
		else
			pyrDown(gpyr[i-3], gpyr[i], Size(gpyr[i - 3].rows, gpyr[i - 3].cols));
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void buildGaussianPyramid(Mat base, Mat* gpyr)
{
	double* sig;
	sig = new double[LAYER];	//記錄每組每層尺度

	double sig_prev, sig_total;
	int rows, cols;

	sig[0] = SIFT_SIGMA;
	double k = pow(2., 1. / nOctaveLayers);

	for (int i = 1; i < LAYER; i++)
	{
		sig_prev = pow(k, (double)(i - 1)) * SIFT_SIGMA;
		sig_total = sig_prev * k;
		sig[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);	//尺度座標
	}

	for (int i = 0; i < OCTAVE; i++)
	{
		for (int j = 0; j < LAYER; j++)
		{
			if (i == 0 && j == 0)
			{
				gpyr[i * LAYER + j] = base.clone();
				rows = gpyr[0].rows;
				cols = gpyr[0].cols;
			}
			else if (j == 0)
			{
				pyrDown(gpyr[(i - 1) * LAYER + 3], base, Size(cols /= 2, rows /= 2));
				gpyr[i * LAYER + j] = base.clone();
			}
			else
			{
				GaussianBlur(base, base, Size(0, 0), sig[j], sig[j]);
				gpyr[i * LAYER + j] = base.clone();

			}
		}
	}
	delete[] sig;
}

void buildDoGPyramid(Mat* gpyr, Mat* dogpyr)
{
	for (int i = 0; i < OCTAVE; i++)
	{
		for (int j = 0; j < DOG_LAYER; j++)
		{
			dogpyr[i * DOG_LAYER + j] = gpyr[i * LAYER + (j + 1)] - gpyr[i * LAYER + j];
		}
	}
}

gradient* image_descriptor(Mat Gaussian_image, int image_heigh, int image_width, int image_pixel)
{
	gradient* image;
	image = new gradient[image_pixel];

	for (int i = 0; i < image_pixel; i++)
	{
		image[i].data = Gaussian_image.data[i];
		image[i].magnitude = 0;
		image[i].orientation = 0;
		for (int j = 0; j < 128; j++)
		{
			image[i].descriptor[j] = 0;
		}
	}

	gradient_calculation(image, image_heigh, image_width);

	//FILE *pfile_r1 = fopen("Orientation_HW.txt", "r");
	//FILE *pfile_r2 = fopen("Magnitude_HW.txt", "r");

	//int *Orientation_HW, *Magnitude_HW;
	//Orientation_HW = new int[image_pixel];
	//Magnitude_HW = new int[image_pixel];

	//int p, a;

	//for (int i = 0; i < image_pixel; i++)
	//{
	//	fscanf(pfile_r1, "%d\n", &p);
	//	Orientation_HW[i] = p;
	//	fscanf(pfile_r2, "%d\n", &p);
	//	Magnitude_HW[i] = p;
	//}

	//for (int i = 0; i < image_pixel; i++)
	//{
	//	image[i].magnitude = Magnitude_HW[i];
	//	image[i].orientation = Orientation_HW[i];
	//}

	histogram_descriptor(image, image_heigh, image_width);
	normalization(image, image_pixel);

	return image;
}

void gradient_calculation(gradient* image, int image_heigh, int image_width)
{
	double dx, dy;

	for (int i = 1; i < image_heigh - 1; i++)
	{
		for (int j = 1; j < image_width - 1; j++)
		{
			dx = image[i*image_width + (j + 1)].data - image[i*image_width + (j - 1)].data;
			dy = image[(i + 1)*image_width + j].data - image[(i - 1)*image_width + j].data;
			image[i*image_width + j].magnitude = image_gradient_magnitude(dx, dy);
			image[i*image_width + j].orientation = image_gradient_orientation(dx, dy);
		}
	}
}

double image_gradient_orientation(double dx, double dy)
{
	double orientation;

	if (dx >= 0 && dy >= 0)
	{
		orientation = (atan2(dy, dx) * 180 / M_PI);
	}
	else if (dx < 0 && dy > 0)
	{
		orientation = 180 - (atan2(dy, -dx) * 180 / M_PI);
	}
	else if (dx < 0 && dy < 0)
	{
		orientation = (atan2(-dy, -dx) * 180 / M_PI) + 180;
	}
	else if (dx >= 0 && dy < 0)
	{
		orientation = 360 - (atan2(-dy, dx) * 180 / M_PI);
	}
	else
	{
		orientation = (atan2(dy, dx) * 180 / M_PI);
	}

	return orientation;
}

double image_gradient_magnitude(double dx, double dy)
{
	double magnitude = 0;

	magnitude = sqrt(dx * dx + dy * dy);

	return magnitude;
}

void histogram_descriptor(gradient* image, int image_heigh, int image_width)
{
	int counter_i = 0, counter_j = 0;
	int bin_i, bin_j;
	int index_i, index_j;
	int bin;
	int orientation;

	for (int i = 8; i < image_heigh - 8; i++)
	{
		counter_j = 0;
		for (int j = 8; j < image_width - 8; j++)
		{
			for (int k = -8; k < 8; k++)
			{
				for (int h = -8; h < 8; h++)
				{
					index_i = (i + k);
					index_j = (j + h);

					/////////////////////////////////
					//判斷區域
					bin_i = (i + k - counter_i) / 4;
					bin_j = (j + h - counter_j) / 4;
					bin = bin_i * 4 + bin_j;
					/////////////////////////////////
					orientation = image[index_i * image_width + index_j].orientation / 45;
					image[i * image_width + j].descriptor[bin * 8 + orientation] += image[index_i * image_width + index_j].magnitude;
				}
			}
			counter_j++;
		}
		counter_i++;
	}
}

void normalization(gradient* image, int image_pixel)
{
	double sum = 0;

	for (int i = 0; i < image_pixel; i++)
	{
		sum = 0;
		for (int j = 0; j < DESCRIPTOR_SIZE; j++)
		{
			sum += image[i].descriptor[j];
		}

		for (int j = 0; j < DESCRIPTOR_SIZE; j++)
		{
			if (sum == 0 || sum < 16)
				image[i].n_descriptor[j] = 0;
			else
				image[i].n_descriptor[j] = image[i].descriptor[j] * 1023 / sum;
		}

		//for (int j = 0; j < DESCRIPTOR_SIZE; j++)
		//{
		//	image[i].n_descriptor[j] = image[i].descriptor[j] / sum;
		//}
	}
}

Mat combination_image(Mat& img1, Mat& img2)
{

	IplImage* img_1;
	img_1 = &IplImage(img1);
	IplImage* img_2;
	img_2 = &IplImage(img2);

	IplImage* stacked = cvCreateImage(cvSize(img_1->width + img_2->width, MAX(img_1->height , img_2->height)), IPL_DEPTH_8U, 3);

	cvZero(stacked);
	cvSetImageROI(stacked, cvRect(0, 0, img_1->width, img_1->height));
	cvAdd(img_1, stacked, stacked, NULL);
	cvSetImageROI(stacked, cvRect(img_1->width, 0, img_2->width, img_2->height));
	cvAdd(img_2, stacked, stacked, NULL);
	cvResetImageROI(stacked);

	Mat out_img(stacked, 0);

	//Mat newimg = Mat((img1.w + img2height), img_1->width * 2, CV_8UC3);

	//Mat tmpimg;

	//tmpimg = newimg(Rect(0, 0, img.rows, img.cols));
	//img.copyTo(tmpimg);

	//tmpimg = newimg(Rect(img.cols, 0, img.rows, img.cols));
	//img.copyTo(tmpimg);



	return out_img;
}

vector <key_point> get_match_data (Mat image_result, gradient* image, int image_heigh, int image_width)
{
	vector <key_point> kpt;
	key_point kpt_tmp;

	for (int i = 0; i < image_heigh; i++)
	{
		for (int j = 0; j < image_width; j++)
		{
			if (image_result.data[i*image_width + j] == 255)
			{
				kpt_tmp.x = j;
				kpt_tmp.y = i;
				for (int k = 0; k < DESCRIPTOR_SIZE; k++)
					kpt_tmp.descr[k] = image[i*image_width + j].n_descriptor[k];
				kpt.push_back(kpt_tmp);
			}
		}
	}

	return kpt;
}

void euclidean_distance(vector<key_point>& kpt1, vector<key_point>& kpt2, Mat trans, Mat& match_img, int width)
{
	int s_size;
	int b_size;
	double* distance_data;
	double* min_distance_data;
	int size = kpt2.size();
	int q = 0;
	int tmp, tmp1;
	double tmp2;
	int* min_index;
	bool matching_flag = 0;

	int matching_x, matching_y, matching_counter = 0;

	distance_data = new double[size];

	FILE *pfile3 = fopen("match.txt", "wb");
	double p;

	for (int i = 0; i < kpt1.size(); i++)
	{
		for (int j = 0; j < kpt2.size(); j++)
		{
			distance_data[j] = 0;
			for (int k = 0; k < SIFT_FEATURE_VECTOR; k++)
			{
				distance_data[j] += sqrt (ABS(kpt1[i].descr[k] - kpt2[j].descr[k]) * ABS(kpt1[i].descr[k] - kpt2[j].descr[k]));
			}
		}

		min_index = bubble_sort(distance_data, size);

		tmp = min_index[0];
		tmp1 = min_index[1];

		delete[] min_index;

		if (distance_data[tmp1] != 0)
		{
			tmp2 = distance_data[tmp] / distance_data[tmp1];
			if (tmp2 < 0.6)
			{
				q++;
				Point center1(kpt1[i].x, kpt1[i].y);
				Point center2(kpt2[tmp].x + width, kpt2[tmp].y);
				//line(match_img, center1, center2, Scalar(rand() % 256, rand() % 256, rand() % 256), 1, 8, 0);

				//------------------------------------
				matching_x = trans.at<double>(0, 0) * kpt1[i].x + trans.at<double>(0, 1) * kpt1[i].y + trans.at<double>(0, 2);
				matching_y = trans.at<double>(1, 0) * kpt1[i].x + trans.at<double>(1, 1) * kpt1[i].y + trans.at<double>(1, 2);

				if (abs(matching_x - kpt2[tmp].x) < 3 && abs(matching_y - kpt2[tmp].y) < 3)
					matching_flag = 1;
				else
					matching_flag = 0;

				if (matching_flag == 1)
				{
					p = i;
					fprintf(pfile3, "%f\t", p);
					p = tmp;
					fprintf(pfile3, "%f\t", p);
					fprintf(pfile3, "%f\t", 1.0);
					fprintf(pfile3, "\n", p);

					line(match_img, center1, center2, Scalar(255, 0, 0), 1, 8, 0);
					matching_counter++;
				}
				else
				{
					p = i;
					fprintf(pfile3, "%f\t", p);
					p = tmp;
					fprintf(pfile3, "%f\t", p);
					fprintf(pfile3, "%f\t", 0.0);
					fprintf(pfile3, "\n", p);
					line(match_img, center1, center2, Scalar(0, 0, 255), 1, 8, 0);
				}
			}
		}
	}
	printf("matching sucessful counter = %d\n", matching_counter);

	fclose(pfile3);

	delete[] distance_data;

	printf("%d\n", q);
}

double* min_data_detector(double* distance_data, int size)

{
	double tmp1, tmp2;
	double* min_data;	//儲存最小兩筆
	min_data = new double[2];



	if (distance_data[0] < distance_data[1])	//
	{
		tmp1 = distance_data[0];
		min_data[0] = 0;
		tmp2 = distance_data[1];
		min_data[1] = 1;
	}
	else
	{
		tmp1 = distance_data[1];
		min_data[0] = 1;
		tmp2 = distance_data[0];
		min_data[1] = 0;
	}

	for (int i = 2; i < size; i++)
	{
		if (distance_data[i] < tmp1 && distance_data[i] < tmp2)
		{
			tmp2 = tmp1;
			min_data[1] = min_data[0];
			tmp1 = distance_data[i];
			min_data[0] = i;
		}
		else if (distance_data[i] > tmp1 && distance_data[i] < tmp2)
		{
			tmp2 = distance_data[i];
			min_data[1] = i;
		}
	}
	return min_data;
}

int* bubble_sort(double* original_data, int size)
{
	double tmp, index_tmp;

	double* data;
	data = new double[size];

	int *data_index;
	data_index = new int[size];

	for (int i = 0; i < size; i++)
	{
		data_index[i] = i;
		data[i] = original_data[i];
	}

	for (int i = 0; i < size - 1; i++)
	{
		for (int j = (i + 1); j < size; j++)
		{
			if (data[i] > data[j])
			{
				tmp = data[i];
				data[i] = data[j];
				data[j] = tmp;

				index_tmp = data_index[i];
				data_index[i] = data_index[j];
				data_index[j] = index_tmp;
			}
		}
	}
	return data_index;
}

void sumvalue_detec(vector<key_point>& kpt1, vector<key_point>& kpt2, Mat& match_img)
{
	for (int i = 0; i < kpt1.size(); i++)
	{
		kpt1[i].sum = 0;
		for (int j = 0; j < SIFT_FEATURE_VECTOR; j++)
		{
			kpt1[i].sum += kpt1[i].descr[j];
		}
	}

	for (int i = 0; i < kpt2.size(); i++)
	{
		kpt2[i].sum = 0;
		for (int j = 0; j < SIFT_FEATURE_VECTOR; j++)
		{
			kpt2[i].sum += kpt2[i].descr[j];
		}
	}
}

void save_picture_to_txt(Mat image, char* file)
{
	FILE *pfile = fopen(file, "w+");

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			fprintf(pfile, "%d\n", image.at<uchar>(i, j));
		}
	}

	fclose (pfile);
}

vector <key_point> get_hardware_feature_descriptor(char* detection_file, int image_heigh, int image_width, int size)
{
	vector <key_point> kpt;
	key_point kpt_tmp;
	int image_pixel = image_heigh * image_width;
	
	FILE *pfile1 = fopen(detection_file, "r");
	int p, a;

	for (int i = 0; i < size; i++)
	{
		fscanf(pfile1, "%d\t", &p);
		kpt_tmp.x = p % 800;
		kpt_tmp.y = p / 800;
		for (int i = 0; i < 128; i++)
		{
			fscanf(pfile1, "%d\t", &p);
			kpt_tmp.descr[i] = p;
		}
		fscanf(pfile1, "\n", &a);
		kpt.push_back(kpt_tmp);
	}

	fclose(pfile1);
	return kpt;
}
