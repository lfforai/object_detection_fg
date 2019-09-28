#pragma once
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include "math.h"
#include <vector>
#include "convert.h"


// load paramter
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
	std::ifstream dataFile(fname, std::ios::in | std::ios::binary);//输入2进制 用于读取float型的参数
	std::stringstream error_s;
	if (!dataFile)
	{
		error_s << "Error opening file " << fname;
		FatalError(error_s.str());
	}
	// we assume the data stored is always in float precision
	float* data_tmp = new float[size];
	int size_b = size * sizeof(float);
	if (!dataFile.read((char*)data_tmp, size_b))
	{
		error_s << "Error reading file " << fname;
		FatalError(error_s.str());
	}
	// conversion
	Convert<value_type> fromReal; //将参数转换为 需要的float  double  或者half1 类型
	for (int i = 0; i < size; i++)
	{
		data_h[i] = fromReal(data_tmp[i]);
	}
	delete[] data_tmp;
}



template <class value_type> //参数读取到device上去
void readAllocMemcpy(const char* fname, int size, value_type* &data_h, value_type* &data_d)
{
	data_h = new value_type[size];

	readBinaryFile<value_type>(fname, size, data_h);

	int size_b = size * sizeof(value_type);
	checkCudaErrors(cudaMallocManaged((void**)&data_d, size_b));
	checkCudaErrors(cudaMemcpy(data_d,data_h,size_b, cudaMemcpyDefault));
}

//paramter
typedef enum {
	FP16_HOST = 0,
	FP16_CUDA = 1,
	FP16_CUDNN = 2
} fp16Import_t;

template <class value_type>
struct Layer_load_t
{
	fp16Import_t fp16Import;
	value_type *data_h, *data_d; //weight 参数
	int size;
	Layer_load_t() : data_h(NULL), data_d(NULL),size(0),fp16Import(FP16_HOST){};
	Layer_load_t(const char* fname_weights,int size_o, fp16Import_t _fp16Import = FP16_HOST):size(size_o)
	{
		fp16Import = _fp16Import;
		string weights_path = string(fname_weights);
		if(!weights_path.empty())
		readAllocInit(weights_path.c_str(),this->size,data_h, data_d);
	}

	~Layer_load_t()
	{
		if (data_h != NULL) delete[] data_h;
		if (data_d != NULL) checkCudaErrors(cudaFree(data_d));
	}
private:
	void readAllocInit(const char* fname, int size, value_type* &data_h, value_type* &data_d)
	{
		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
	}
};

template <>
void Layer_load_t<half1>::readAllocInit(const char* fname, int size, half1* &data_h, half1* &data_d)
{
	data_h = new half1[size];
	int size_b = size * sizeof(half1);
	checkCudaErrors(cudaMallocManaged((void **)&data_d, size_b));

	switch (fp16Import)
	{
		case FP16_HOST: 
		{
			readBinaryFile<half1>(fname, size, data_h);
			checkCudaErrors(cudaMemcpy((void **)data_d, data_h, size_b, cudaMemcpyDefault));
			break;
		}
		case FP16_CUDA:
		{   float *data_float_h, *data_float_d;
			readAllocMemcpy<float>(fname, size, data_float_h, data_float_d);
			gpu_float2half_rn<float>(size, data_float_d, data_d);
			checkCudaErrors(cudaMemcpy((void **)data_h, data_d, size_b,cudaMemcpyDefault));
			checkCudaErrors(cudaFree(data_float_d));
			delete[] data_float_h;
			break;
		}
		case FP16_CUDNN:
		{   //用cudnn 的类型转换函数 将float转half //效率最高
			float *data_float_h, *data_float_d;
			readAllocMemcpy<float>(fname, size, data_float_h, data_float_d);
			delete[] data_float_h;

			cudnnHandle_t cudnnHandle;
			cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
			checkCUDNN(cudnnCreate(&cudnnHandle));
			checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
			checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
			checkCUDNN(cudnnSetTensor4dDescriptorEx(srcTensorDesc,
				CUDNN_DATA_FLOAT,
				1, size,
				1, 1,
				size, 1, 1, 1));
			checkCUDNN(cudnnSetTensor4dDescriptorEx(dstTensorDesc,
				CUDNN_DATA_HALF,
				1, size,
				1, 1,
				size, 1, 1, 1));
			float alpha = 1.0f;
			float beta = 0.0f;
			checkCUDNN(cudnnTransformTensor(cudnnHandle, &alpha,
				srcTensorDesc,
				data_float_d, &beta,
				dstTensorDesc,
				data_d));
			checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
			checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
			checkCUDNN(cudnnDestroy(cudnnHandle));
			checkCudaErrors(cudaFree(data_float_d));
			checkCublasErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy((void **)data_h, data_d, size_b,
				cudaMemcpyDefault));
			break;
		}
	}
}

// save parameter-----------------------------------------------------------------------------------
template <class value_type>  //save的所有东西都在host上
void saveBinaryFile(const char* fname, int size, value_type* data_h)
{
	std::ofstream dataFile(fname, ios::trunc|ios::out|ios::binary);//输入2进制 用于读取float型的参数
	std::stringstream error_s;
	if (!dataFile)
	{
		error_s << "Error opening file " << fname;
		FatalError(error_s.str());
	}

	float* data_tmp = new float[size];

	Convert<float> fromReal;

	for (int i = 0; i < size; i++)
	{
		data_tmp[i] = fromReal(data_h[i]);
	}

	int size_b = size * sizeof(float);
	// we assume the data stored is always in float precision
	if (!dataFile.write((char*)data_tmp, size_b))
	{
		error_s << "Error reading file " << fname;
		FatalError(error_s.str());
	}

	// conversion
	delete[] data_tmp;
}

template <class value_type> //把参数从device提取到data_h上
void saveAllocMemcpy(const char* fname, int size, value_type* data_d)
{
	value_type* data_h = new value_type[size];

	int size_b = size * sizeof(value_type);
	checkCudaErrors(cudaMemcpy(data_h, data_d,
		size_b,
		cudaMemcpyDeviceToHost));

	saveBinaryFile<value_type>(fname, size, data_h);
	delete[]  data_h;
}

template <class value_type>
struct Layer_save_t
{   
	fp16Import_t fp16Import;
	int size;
	value_type *data_d; 

	Layer_save_t() : data_d(NULL),size(0),fp16Import(FP16_HOST) {};
	
	Layer_save_t(const char* fname_weights,value_type* data_d_o,int size_o, fp16Import_t _fp16Import = FP16_HOST):
		size(size_o)
	{
		fp16Import = _fp16Import;
		string weights_path = string(fname_weights);

		this->data_d = data_d_o;

		if(!weights_path.empty())
		saveAllocInit(weights_path.c_str(),size,data_d);
	}

	~Layer_save_t()
	{
		if (data_d != NULL) checkCudaErrors(cudaFree(data_d));
	}

private:
	void saveAllocInit(const char* fname, int size, value_type* data_d)
	{
		saveAllocMemcpy<value_type>(fname, size, data_d);
	}
};

template <>
void Layer_save_t<half1>::saveAllocInit(const char* fname, int size, half1* data_d)
{   
	float *data_tmp_d;
	int size_b = size * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&data_tmp_d, size_b));

	switch (fp16Import)
	 {
		case FP16_HOST:
		{   half1* data_h = (half1* )malloc(size*sizeof(half1));
		    checkCudaErrors(cudaMemcpy(data_h, data_d,
			size_b,
			cudaMemcpyDeviceToHost));
			saveBinaryFile<half1>(fname, size, data_h);
			delete[] data_h;
			break;
		}
		case FP16_CUDA:
		{   
			gpu_half2float_rn<float>(size,data_d,data_tmp_d);
			saveAllocMemcpy<float>(fname, size, data_tmp_d);
			checkCudaErrors(cudaFree(data_tmp_d));
			break;
		}
		case FP16_CUDNN:
		{   //用cudnn 的类型转换函数 将float转half //效率最高
			cudnnHandle_t cudnnHandle;
			cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
			checkCUDNN(cudnnCreate(&cudnnHandle));
			checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
			checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
			
			checkCUDNN(cudnnSetTensor4dDescriptorEx(dstTensorDesc,
				CUDNN_DATA_FLOAT,
				1, size,
				1, 1,
				size, 1, 1, 1));
			
			checkCUDNN(cudnnSetTensor4dDescriptorEx(srcTensorDesc,
				CUDNN_DATA_HALF,
				1, size,
				1, 1,
				size, 1, 1, 1));

			float alpha = 1.0f;
			float beta = 0.0f;
			checkCUDNN(cudnnTransformTensor(cudnnHandle, &alpha,
				srcTensorDesc,
				data_d, &beta,
				dstTensorDesc,
				data_tmp_d));

			saveAllocMemcpy<float>(fname, size, data_tmp_d);
			checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
			checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
			checkCUDNN(cudnnDestroy(cudnnHandle));
			checkCudaErrors(cudaFree(data_tmp_d));
			break;
		}
	  }
}
template class Layer_load_t<float>;
