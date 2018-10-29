/*
 * File:   image_blurring.cu
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include <cuda_runtime.h>

using namespace std;

// KERNEL 1: Pixel's color Histogram (atomic operations)
// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images (cols, rows)
__global__ void histogram_kernel(unsigned int *histogram, unsigned char* input, int width, int height)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    //Creating shared memory histogram
    __shared__ unsigned int hist[256];

    //Global Index (for image accessing)
    int i = yIndex * width + xIndex;

    //Indeces inside the current Block
    int hist_index = threadIdx.x + blockDim.x * threadIdx.y;

    //Clear shared memory 
    if (hist_index < 256)
		hist[hist_index] = 0;
	__syncthreads();

	// Only valid threads in the IMAGE
	if ((xIndex < width) && (yIndex < height))
		atomicAdd(&hist[(int)input[i]], 1);

	__syncthreads();

	//Get all the values in the partial histogram (one histogram per block)
	if (hist_index < 256)
		//Add the result of the histograms obtained per block
		//in order to obtain the complete histogram calculation
		atomicAdd(&histogram[hist_index], hist[hist_index]);
}

// KERNEL 2: Calculate cummulative Sum & normalize
__global__ void normalize_hist_kernel(unsigned int *histogram, unsigned  int *histogram_s, unsigned  int *lookUp, double normalValue)
{
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	if (ind < 256)
	{
		histogram_s[ind] = 0;
		for (int i = ind; i >= 0; i--)
			histogram_s[ind] += histogram[i];
		lookUp[ind] = histogram_s[ind] * normalValue;
	}

}

// KERNEL 3: Create new image
__global__ void newImg_kernel(unsigned int *lookUp, unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		int i = yIndex * width + xIndex;
		output[i] = static_cast<unsigned char>(lookUp[input[i]]);
	}
}

void equalizate_img(const cv::Mat& input, cv::Mat& output)
{
	//cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	
	// Calculate total number of bytes of input and output image	
	size_t imgBytes = input.step * input.rows;
	double normalValue = 255;
	normalValue /= (input.cols * input.rows);

	unsigned char *d_input, *d_output;
	unsigned int *histogram, *histogram_s, *lookUp;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, imgBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, imgBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned int>(&histogram, 256 * sizeof(unsigned int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned int>(&histogram_s, 256 * sizeof(unsigned int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned int>(&lookUp, 256 * sizeof(unsigned int)), "CUDA Malloc Failed");

	//Initialize arrays
	cudaMemset(histogram, 0, 256*sizeof(int));
	cudaMemset(histogram_s, 0, 256*sizeof(int));
	cudaMemset(lookUp, 0, 256*sizeof(int));

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), imgBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Block size with enough threads for the whole image
	const dim3 block_Img(64, 4);
	// Calculate grid size to cover the whole image
	const dim3 grid_Img((int)ceil((float)input.cols / block_Img.x), (int)ceil((float)input.rows/ block_Img.y));

	// Block size for the histogram values only
	const dim3 block_Hist(16, 16);
	// Calculate grid size to cover the whole image
	//const dim3 grid_Hist((256 + block.x - 1) / block.x, 1);

	printf("Histogram_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid_Img.x, grid_Img.y, block_Img.x, block_Img.y);
	printf("Normalized_hist_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid_Img.x, grid_Img.y, block_Hist.x, block_Hist.y);
	printf("newImg_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid_Img.x, grid_Img.y, block_Img.x, block_Img.y);

	chrono::duration<float, std::milli> duration_ms = chrono::high_resolution_clock::duration::zero();
	auto start_gpu =  chrono::high_resolution_clock::now();

	//Kernels implementation
	histogram_kernel <<<grid_Img, block_Img >>> (histogram, d_input, input.cols, input.rows);
	normalize_hist_kernel <<<grid_Img, block_Hist >>> (histogram, histogram_s, lookUp, normalValue);
	newImg_kernel <<<grid_Img, block_Img >>> (lookUp, d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));

	auto end_gpu =  chrono::high_resolution_clock::now();
	duration_ms = end_gpu - start_gpu;
	printf("GPU Image Equalization elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, imgBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Save resultant image
	cv::imwrite("Modified_Img/GPU_Image.jpg", output);

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
	SAFE_CALL(cudaFree(histogram), "CUDA Free Failed");
	SAFE_CALL(cudaFree(histogram_s), "CUDA Free Failed");
	SAFE_CALL(cudaFree(lookUp), "CUDA Free Failed");
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
	cv::imwrite("Modified_Img/GPU_Original.jpg", input);

	cv::Mat output(input.rows, input.cols, input.type());

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	//Call the wrapper function
	equalizate_img(input, output);

	/* ********* DISPLAY IMAGES **********/
	//Allow the windows to resize
	namedWindow("GPU INPUT", cv::WINDOW_NORMAL);
	namedWindow("GPU OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("GPU INPUT", input);
	imshow("GPU OUTPUT", output);
	
	//Wait for key press
	cv::waitKey();
	

	return 0;	
}


/* For printing Something in CUDA

//unsigned int *h;
//h = (unsigned int *)malloc(256 * sizeof(unsigned int));

cudaMemcpy(h, histogram, 256*sizeof(int), cudaMemcpyDeviceToHost);
for (int i = 0; i < 256; i++)
	cout << "Histograma en " << i << " : " << h[i] << endl;
*/