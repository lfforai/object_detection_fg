#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include "math.h"
#include <device_launch_parameters.h>
#include <sstream>
#include <curand.h>
#include "cufft.h"
#include <typeinfo>
#include <time.h>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <helper_cuda.h>       // helper for CUDA Error handling and initialization
#include <helper_string.h>  // helper for string parsing
#include <assert.h>
#include <mutex>
#include <ctime> 
#include "fp16_dev.h"
//#include "fp16_dev.h"

using namespace std;

#ifndef _GPU_MATH_TOOL_CUH
#define _GPU_MATH_TOOL_CUH
template <class value_type>
void add_vector_gpu(int size, value_type *buffA, value_type *buffB);
#endif // !_GPU_MATH_TOOL_CUH
