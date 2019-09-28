#pragma once
#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include <FreeImage.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "error_util.h"

#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
	cudnnTensorFormat_t& tensorFormat,
	cudnnDataType_t& dataType,
	int n,
	int c,
	int h,
	int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w));

#elif defined(ND_TENSOR_DESCRIPTOR) //这个是最常用的
	int* dimA_o;
	int* strideA_o;
	const int nDims = 4;

	if (tensorFormat == CUDNN_TENSOR_NCHW || CUDNN_TENSOR_NCHW_VECT_C)
	{
		int dimA[nDims] = { n,c,h,w };
		int strideA[nDims] = { c*h*w, h*w, w, 1 };
		dimA_o = dimA;
		strideA_o = strideA;
	}

	if (tensorFormat == CUDNN_TENSOR_NHWC)
	{
		int dimA[nDims] = { n,h,w,c };
		int strideA[nDims] = { c*h*w, w*c, c, 1 };
		dimA_o = dimA;
		strideA_o = strideA;
	}

	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc,
		dataType,
		4,
		dimA_o,
		strideA_o));
#else
	checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc,
		dataType,
		n, c,
		h, w,
		c*h*w, h*w, w, 1));
#endif
}