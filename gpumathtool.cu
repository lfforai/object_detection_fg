#pragma once
#include "gpumathtool.cuh"

#define BLOCK_SIZE 128
template <class value_type>
__global__ void add_vector_kernel(int size, value_type *buffA, value_type *buffB)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] =buffA[idx] + buffB[idx];
};

template <>
__global__ void add_vector_kernel<half1>(int size, half1 *buffA, half1 *buffB)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = __hadd(buffA[idx], buffB[idx]);
}

template <class value_type>
void add_vector_gpu(int size, value_type *buffA, value_type *buffB)
{
	int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	add_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA, buffB);
	checkCudaErrors(cudaDeviceSynchronize());
};

template void add_vector_gpu<float>(int, float*, float*);
template void add_vector_gpu<double>(int,double*, double*);
template void add_vector_gpu<int>(int, int*, int*);
template void add_vector_gpu<half1>(int, half1*, half1*);

//math function
template <class value_type>
__global__ void log_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = logf(buffA[idx]);
};


typedef enum {
	//cudnnReduceTensorOp_t
	CONS_LOG = 0,
	CONS_APOWX = 1,
	CONS_XPOWA = 1,
	CONS_SQRT = 2,
	CONS_X2 = 3,
	CONS_COS = 4,
	CONS_SIN = 5,
	CONS_EXP = 6,
} gpu_math_op;

template <class value_type>
//A only used by apowx,xpowa
void math_vector_gpu(gpu_math_op math_op,int size, value_type *buffA,value_type A)
{
	gpu_math_op cudnn_op_math;
	int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	switch (math_op)
	{
	case 0:
		cudnn_op_math = CONS_LOG;
		log_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA, buffB);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 1:
		cudnn_op_math = CONS_APOWX;
		break;
	case 2:
		cudnn_op_math = CONS_XPOWA;
		break;
	case 3:
		cudnn_op_math = CONS_X2;
		break;
	case 4:
		cudnn_op_math = CONS_COS;//only A useed 2=2^0.5=1.14...
		break;
	case 5:
		cudnn_op_math = CONS_SIN;//only A useed ,eg::5=> -5+1= -4
		break;
	case 6:
		cudnn_op_math = CONS_EXP;//only A useed ,eg::5=> -5+1= -4
		break;
	default:
		cout << "wrong constant<T> math op type " << endl;
	}
};