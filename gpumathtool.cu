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
