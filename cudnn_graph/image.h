#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include <sstream>
#include <FreeImage.h>
#include "paramter.h"

//#include <helper_cuda.h>       // helper for CUDA Error handling and initialization
//#include <helper_string.h>  // helper for string parsing

using namespace std;
using namespace cv;


#ifndef _IMAGE_H
#define _IMAGE_H

template<class value_type>
class image
{
public:
	value_type* imgData_h;
	int size;
	int width;
	int height;
	int channl;

	void readImage(const char* fname);
	void printDeviceVector(int size, value_type* vec_d);

	image();
	~image();
};
#endif
