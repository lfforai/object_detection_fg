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
typedef enum {
	//cudnnReduceTensorOp_t
	CONS_LOG = 0,
	CONS_APOWX = 1,
	CONS_XPOWA = 2,
	CONS_SQRT = 3,
	CONS_COS = 4,
	CONS_SIN = 5,
	CONS_EXP = 6,
    CONS_REC = 7,//reciprocal=1/x
} gpu_math_op;

template <class value_type>
//A only used by apowx,xpowa
void math_vector_gpu(gpu_math_op math_op, int size, value_type *buffA, value_type  aphal);

template <class value_type>
//A only used by apowx,xpowa
void dmath_vector_gpu(gpu_math_op math_op, int size, value_type *buffA, value_type  aphal);

//static int checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
//	if (code) {
//		printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
//		return 1;
//	}
//	return 0;
//}
//
//static int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
//	if (code) {
//		printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
//		return 1;
//	}
//	return 0;
//}

//
//#define checkCudaErr(...)       do { int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); if (err) goto clean; } while (0)
//
//#define checkCudnnErr(...)      do { int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  if (err) goto clean; } while (0)

//function
template <class value_type>
void add_vector_gpu(int size, value_type *buffA, value_type *buffB);
//
#endif // !_GPU_MATH_TOOL_CUH
