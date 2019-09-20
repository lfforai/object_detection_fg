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

//math function---------------------------------------------------------------
template <class value_type>
__global__ void log_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = logf(buffA[idx]);
};

template <class value_type>  //   aphal^buffA=a^x
__global__ void apowx_vector_kernel(int size, value_type *buffA, value_type aphal)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = powf(aphal,buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void xpowa_vector_kernel(int size, value_type *buffA, value_type aphal)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = powf(buffA[idx], aphal);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void sqrt_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = sqrtf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void cos_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] =cosf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void sin_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = sinf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void exp_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = expf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void rec_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = fdividef((value_type)1.0,buffA[idx]);
};


template <class value_type>
//A only used by apowx,xpowa
void math_vector_gpu(gpu_math_op math_op,int size, value_type *buffA,value_type  aphal)
{
	gpu_math_op cudnn_op_math;
	int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	switch(math_op){
	case 0:
		cudnn_op_math = CONS_LOG;
		log_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 1:
		cudnn_op_math = CONS_APOWX;
		apowx_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA, aphal);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 2:
		cudnn_op_math = CONS_XPOWA;
		xpowa_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA, aphal);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 3:
		cudnn_op_math = CONS_SQRT;
		sqrt_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 4:
		cudnn_op_math = CONS_COS;
		cos_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 5:
		cudnn_op_math = CONS_SIN;//
		sin_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 6:
		cudnn_op_math = CONS_EXP;//
		exp_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 7:
		cudnn_op_math = CONS_REC;//1/x,x!=0
		rec_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	default:
		cout << "wrong constant<T> math op type " << endl;
	}
};
template void math_vector_gpu<float>(gpu_math_op, int, float*, float);

//dy/dx ,math function-----------------------------------------
template <class value_type>
__global__ void dlog_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = fdividef(value_type(1.0),(buffA[idx]));
};

template <class value_type>  //   aphal^buffA=a^x
__global__ void dapowx_vector_kernel(int size, value_type *buffA, value_type aphal)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = powf(aphal, buffA[idx])*log(aphal);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void dxpowa_vector_kernel(int size, value_type *buffA, value_type aphal)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = powf(buffA[idx], aphal-1)*aphal;
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void dsqrt_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = powf(buffA[idx],-0.5)*0.5;
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void dcos_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = -sinf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void dsin_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = cosf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void dexp_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = expf(buffA[idx]);
};

template <class value_type>  //   buffA^aphal=x^a
__global__ void drec_vector_kernel(int size, value_type *buffA)
{
	const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	buffA[idx] = -fdividef(value_type(1.0), buffA[idx] * buffA[idx]);
};

//typedef enum {
//	//cudnnReduceTensorOp_t
//	CONS_LOG = 0,
//	CONS_APOWX = 1,
//	CONS_XPOWA = 2,
//	CONS_SQRT = 3,
//	CONS_X2 = 4,
//	CONS_COS = 5,
//	CONS_SIN = 6,
//	CONS_EXP = 7,
//} gpu_math_op;

template <class value_type>
//A only used by apowx,xpowa
void dmath_vector_gpu(gpu_math_op math_op, int size, value_type *buffA, value_type  aphal)
{
	gpu_math_op cudnn_op_math;
	int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	switch (math_op) {
	case 0:
		cudnn_op_math = CONS_LOG;
		dlog_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 1:
		cudnn_op_math = CONS_APOWX;
		dapowx_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA, aphal);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 2:
		cudnn_op_math = CONS_XPOWA;
		dxpowa_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA, aphal);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 3:
		cudnn_op_math = CONS_SQRT;
		dsqrt_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 4:
		cudnn_op_math = CONS_COS;
		dcos_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 5:
		cudnn_op_math = CONS_SIN;//
		dsin_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 6:
		cudnn_op_math = CONS_EXP;//only A useed ,eg::5=> -5+1= -4
		dexp_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	case 7:
		cudnn_op_math = CONS_REC;//1/x,x!=0
		drec_vector_kernel<value_type> << <grid_size, BLOCK_SIZE >> > (size, buffA);
		checkCudaErrors(cudaDeviceSynchronize());
		break;
	default:
		cout << "aaaa wrong constant<T> math op type " << endl;
	}
};

template void dmath_vector_gpu<float>(gpu_math_op, int, float*, float);