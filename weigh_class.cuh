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
#include "hash_global.cuh"
#include <assert.h>
#include <mutex>
#include <ctime> 
#include "test_tool.h"

#include <windows.h>
#include <wincrypt.h>
using namespace std;
#ifndef _WEIGH_CLASS_CUH
#define _WEIGH_CLASS_CUH 
template<class T>
struct variable{
	
	T* w = NULL;//w
	int* w_stride = NULL;
	int  w_dim_num;
	int* w_dim = NULL;

	int device = 0;
	bool trainable = true;

	string var_name;

	static variable<T>* getObject(string var_name_o,int device_o, int w_dim_num_o, int *w_dim_o, T* w_src)
	  {
		variable<T>* r = new variable<T>;
		r->var_name = var_name_o;
		r->device = device_o;
		r->w_dim_num = w_dim_num_o;

		r->w_dim = (int *)malloc(r->w_dim_num * sizeof(int));
		r->w_stride = (int *)malloc(r->w_dim_num * sizeof(int));
		memcpy(r->w_dim, w_dim_o, r->w_dim_num * sizeof(int));

		if (r->w_dim_num > 1)
		{
			r->w_stride[r->w_dim_num - 1] = 1;
			for (int d = r->w_dim_num - 2; d >= 0; d--) {
				r->w_stride[d] = r->w_stride[d + 1] * r->w_dim[d + 1];
			}
		}
		else {
			r->w_stride[r->w_dim_num - 1] = 1;
		}

		int length = r->w_stride[0] * r->w_dim[0];

		if (r->device == 0){
			r->w = (T*)malloc(length * sizeof(T));
			memcpy(r->w, w_src, length * sizeof(T));
		}
		else {
			checkCudaErrors(cudaMalloc((void**)&r->w, length * sizeof(T)));
			checkCudaErrors(cudaMemcpy(r->w, w_src, length * sizeof(T), cudaMemcpyHostToDevice));
		}
		return r;
	  }
};

#endif // !_WEIGH_CLASS_CUH
