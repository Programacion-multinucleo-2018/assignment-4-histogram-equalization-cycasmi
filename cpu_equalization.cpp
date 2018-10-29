/*
 * File:   cpu_image_blurring.cpp
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 * Created on September 9th, 2018, 01:33 PM
 */

//g++ cpu_image_blurring.cpp `pkg-config --cflags --libs opencv`

#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

void CPU_img_equalization(const cv::Mat& M_input, cv::Mat& M_output)
{	
	int histogram[256] = {0}; //of pixels
	int histogram_s[256] = {0}; //Cumulative Sum of pixels values
	int lookUp[256] = {0}; //normalized values 
	double normalValue = 255;
	normalValue /= (M_input.cols * M_input.rows);

	int colorWidthStep = static_cast<int>(M_input.step); //image width
	size_t inputBytes = M_input.step*M_input.rows;
	unsigned char *input, *output;
	input = output = (unsigned char *) malloc(inputBytes*sizeof(unsigned char));
	memcpy(input, M_input.ptr(), inputBytes*sizeof(unsigned char));

	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < M_input.cols; i++)
	{
		for (int j = 0; j < M_input.rows; j++)
		{
			#pragma omp atomic
			histogram[(int)M_input.at<uchar>(j, i)] += 1;
		}
	}

	//for (int i = 0; i < 256; i++)
	//	cout << "Histograma CPU en " << i << " : " << histogram[i] << endl;

	//Cumulative Sum
	histogram_s[0] = histogram[0];
	lookUp[0] = histogram_s[0] * normalValue;
	for (int i = 1; i < 256; i++)
	{
		histogram_s[i] = histogram_s[i-1] + histogram[i];
		lookUp[i] = histogram_s[i] * normalValue;
	}
	//cout << "Normal value: " << normalValue << endl;
	//cout << "suma acumulada: " << histogram_s[255] << endl;
	//cout << "suma acumulada normalizada inicial: " << lookUp[0] << endl;
	//cout << "suma acumulada normalizada final: " << lookUp[255] << endl;
	
	int index = 0;
	for (int i = 0; i < M_input.cols; i++)
	{
		for (int j = 0; j < M_input.rows; j++)
		{
			index = j * colorWidthStep + i;
			output[index] = lookUp[input[index]];
		}
	}

	memcpy(M_output.ptr(), output, inputBytes*sizeof(unsigned char));

	//Save resultant image
	cv::imwrite("Modified_Img/CPU_Image.jpg", M_output);
}

int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		imagePath = "Images/scenery.jpg";
		//imagePath = "image.jpg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat original = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (original.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	cv::Mat input;
	cv::cvtColor( original, input, CV_BGR2GRAY );
	cv::imwrite("Modified_Img/CPU_Original.jpg", input);

	cv::Mat output(input.rows, input.cols, input.type());

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// NO THREADS CPU TEST
	chrono::duration<float, std::milli> duration_ms = chrono::high_resolution_clock::duration::zero();
	auto start =  chrono::high_resolution_clock::now();
	
	CPU_img_equalization(input, output);

	auto end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("CPU image equalization elapsed %f ms\n\n", duration_ms.count());

	/* ********* DISPLAY IMAGES **********/
	//Allow the windows to resize
	namedWindow("CPU INPUT", cv::WINDOW_NORMAL);
	namedWindow("CPU OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("CPU INPUT", input);
	imshow("CPU OUTPUT", output);
	
	//Wait for key press
	cv::waitKey();
	

	return 0;	
}