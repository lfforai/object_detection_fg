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
#include "constant_class.cuh"
#include <windows.h>
#include <wincrypt.h>
#include "paramter.h"

using namespace std;
#ifndef _WEIGH_CLASS_CUH
#define _WEIGH_CLASS_CUH 

template<class T>
class variable:public constant<T>
{
private:
public:
	Layer_load_t<T>* paramterload;
	Layer_load_t<T>* paramtersave;

	bool trainable=true;
	string var_name;
	variable<T>* copy()
	{
		this->copynum += 1;
		variable<T>* result = variable<T>::getObject(this->trainable,this->var_name + "_" + to_string(this->copynum), this->device, this->x_dim_num, this->x_dim, this->x);
		return result;
	}

	variable<T>* copy_zero() //only copy dim_num ,dim,set all x=0
	{
		int length = this->x_stride[0] * this->x_dim[0];
		this->copynum += 1;
		T* temp;
		if (this->device == 1)//on gpu
		{    
			checkCudaErrors(cudaMallocManaged((void**)&temp, length * sizeof(T)));
			checkCudaErrors(cudaMemset(temp, 0, sizeof(T)*length));
		}else{
			temp = (T*)malloc(length * sizeof(T));
			memset(temp,0, length * sizeof(T));
		}
		variable<T>* result = variable<T>::getObject(this->trainable,this->var_name + "_" + to_string(this->copynum), this->device, this->x_dim_num, this->x_dim, temp);
		return result;
	}

	//init in cpu=0,init in gpu=1
	variable(){};
	variable(bool trainble_o,string var_name_o, int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {

		this->paramterload = new Layer_load_t<T>;
		this->paramtersave = new Layer_load_t<T>;

		this->trainable = trainble_o;
		this->var_name = var_name_o;
		this->device = device_o;
		this->x_dim_num = x_dim_num_o;

		this->x_dim = (int *)malloc(this->x_dim_num * sizeof(int));
		this->x_stride = (int *)malloc(this->x_dim_num * sizeof(int));
		memcpy(this->x_dim, x_dim_o, this->x_dim_num * sizeof(int));

		if (this->x_dim_num > 1)
		{
			this->x_stride[this->x_dim_num - 1] = 1;
			for (int d = this->x_dim_num - 2; d >= 0; d--) {
				this->x_stride[d] = this->x_stride[d + 1] * this->x_dim[d + 1];
			}
		}
		else {
			this->x_stride[this->x_dim_num - 1] = 1;
		}

		int length = this->x_stride[0] * this->x_dim[0];
		this->length = length;

		if (this->device == 0) {
			this->x = (T*)malloc(length * sizeof(T));
			memcpy(this->x, x_src, length * sizeof(T));
		}
		else {
			checkCudaErrors(cudaMallocManaged((void**)&this->x, length * sizeof(T)));
			checkCudaErrors(cudaMemcpy(this->x, x_src, length * sizeof(T), cudaMemcpyDefault));
		}
	}

	static variable<T>* getObject(bool trainble_o,string var_name_o, int device_o, int x_dim_num_o, int *x_dim_o, T* x_src)
	{  
		variable<T>* r = new variable<T>;

		r->paramterload =new Layer_load_t<T>;
		r->paramtersave = new Layer_load_t<T>;

		r->trainable= trainble_o;
		r->var_name = var_name_o;
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
		r->length = length;

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
