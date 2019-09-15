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
#include "constant_class.cuh"
#include <windows.h>
#include <wincrypt.h>
using namespace std;
#ifndef _WEIGH_CLASS_CUH
#define _WEIGH_CLASS_CUH 
template<class T>
class variable:public constant<T>
{
private:
public:
	bool trainable=true;
	string var_name;
	//init in cpu=0,init in gpu=1
	variable(){};
	variable(bool trainble_o,string con_name_o, int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {
		device = device_o;
		x_dim_num = x_dim_num_o;
		var_name = con_name_o;
		r->trainable = trainble_o;

		x_dim = (int *)malloc(x_dim_num * sizeof(int));
		x_stride = (int *)malloc(x_dim_num * sizeof(int));
		memcpy(x_dim, x_dim_o, x_dim_num * sizeof(int));

		if (x_dim_num > 1)
		{
			x_stride[x_dim_num - 1] = 1;
			for (int d = x_dim_num - 2; d >= 0; d--) {
				x_stride[d] = x_stride[d + 1] * x_dim[d + 1];
			}
		}
		else {
			x_stride[x_dim_num - 1] = 1;
		}

		int length = x_stride[0] * x_dim[0];

		if (device == 0) {
			x = (T*)malloc(length * sizeof(T));
			memcpy(x, x_src, length * sizeof(T));
		}
		else {
			checkCudaErrors(cudaMemcpy(x, x_src, length * sizeof(T), cudaMemcpyHostToDevice));
		}
	}

	static variable<T>* getObject(bool trainble_o,string con_name_o, int device_o, int x_dim_num_o, int *x_dim_o, T* x_src)
	{
		variable<T>* r = new variable<T>;
		r->trainable= trainble_o;
		r->var_name = con_name_o;
		r->device = device_o;
		r->x_dim_num = x_dim_num_o;

		r->x_dim = (int *)malloc(r->x_dim_num * sizeof(int));
		r->x_stride = (int *)malloc(r->x_dim_num * sizeof(int));
		memcpy(r->x_dim, x_dim_o, r->x_dim_num * sizeof(int));

		if (r->x_dim_num > 1)
		{
			r->x_stride[r->x_dim_num - 1] = 1;
			for (int d = r->x_dim_num - 2; d >= 0; d--) {
				r->x_stride[d] = r->x_stride[d + 1] * r->x_dim[d + 1];
			}
		}
		else {
			r->x_stride[r->x_dim_num - 1] = 1;
		}

		int length = r->x_stride[0] * r->x_dim[0];

		if (r->device == 0) {
			r->x = (T*)malloc(length * sizeof(T));
			memcpy(r->x, x_src, length * sizeof(T));
		}
		else {
			checkCudaErrors(cudaMallocManaged((void**)&r->x, length * sizeof(T)));
			checkCudaErrors(cudaMemcpy(r->x, x_src, length * sizeof(T), cudaMemcpyDefault));
		}
		return r;
	}
};

#endif // !_WEIGH_CLASS_CUH
