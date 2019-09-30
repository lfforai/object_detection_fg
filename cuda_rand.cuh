#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>       // helper for CUDA Error handling and initialization
#include <helper_string.h>  // helper for string parsing

#define CURAND_CALL(x) do{if((x)!=CURAND_STATUS_SUCCESS){printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}}while(0)

float* cuda_rand(int n) {
	size_t i;
	curandGenerator_t gen;
	float *devData,*hostData;

	/* Allocate n floats on host */
	hostData = (float *)calloc(n, sizeof(float));

	/* Allocate n floats on device */
	checkCudaErrors(cudaMalloc((void **)&devData, n * sizeof(float)));

	/* Create pseudo-random number generator */
	curandCreateGenerator(&gen,
		CURAND_RNG_PSEUDO_DEFAULT);

	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);

	/* Generate n floats on device */
	curandGenerateUniform(gen, devData, n);

	/* Copy device memory to host */
	checkCudaErrors(cudaMemcpy(hostData, devData, n * sizeof(float),
		cudaMemcpyDeviceToHost));

	/* Cleanup */
    curandDestroyGenerator(gen);
	checkCudaErrors(cudaFree(devData));
	return hostData;
}