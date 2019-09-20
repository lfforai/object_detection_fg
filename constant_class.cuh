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
#include <cudnn.h>
#include "gpumathtool.cuh"
#include <windows.h>
#include <wincrypt.h>
#include "fp16_dev.h"
using namespace std;

#ifndef _CONSTANT_CLASS_CUH
#define _CONSTANT_CLASS_CUH


typedef enum{
	CONSTANT_OP_ADD = 0,
	CONSTANT_OP_MUL = 1,
	CONSTANT_OP_TENSOR_MIN = 2,
	CONSTANT_OP_TENSOR_MAX = 3,
	CONSTANT_OP_TENSOR_SQRT = 4,
	CONSTANT_OP_TENSOR_NOT = 5,
} constant_math_op;

typedef enum {
	//cudnnReduceTensorOp_t
	CONS_REDUCE_TENSOR_ADD = 0,
	CONS_REDUCE_TENSOR_MUL = 1,
	CONS_REDUCE_TENSOR_MIN = 2,
	CONS_REDUCE_TENSOR_MAX = 3,
	CONS_REDUCE_TENSOR_AMAX = 4,
	CONS_REDUCE_TENSOR_AVG = 5,
	CONS_REDUCE_TENSOR_NORM1 = 6,
	CONS_REDUCE_TENSOR_NORM2 = 7,
	CONS_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} constant_math_op_reduce;

template<class T>
class constant{
private:

public:
	//input from father 
	T*  x;//forward input
	int* x_stride;
	int  x_dim_num;//length of x_dim
	int* x_dim;
	int  device;
	int  placeholder = 0; //not placeholder,placeholder::[None,244,244,5] list tensorflow
	int copynum = 0;
	string con_name;
	
	void clear() {
		if (device == 1)
			cudaFree(this->x);
		else
			free(this->x);
		free(x_stride);
		free(x_dim);
	}
	//log ,exp.....
	//typedef enum {
	//	//cudnnReduceTensorOp_t
	//	CONS_LOG = 0,
	//	CONS_APOWX = 1,
	//	CONS_XPOWA = 2,
	//	CONS_SQRT = 3,
	//	CONS_COS = 4,
	//	CONS_SIN = 5,
	//	CONS_EXP = 6,
	//} gpu_math_op;
	//partial derivative;
	constant<T>* function_tensor(gpu_math_op math_op,T aphal,int pderi)
	  {
		constant<T>* result =this->copy();
		int len= result->x_stride[0] * result->x_dim[0];
		if (pderi == 0)  //ong-
			math_vector_gpu(math_op, len, result->x, aphal);
		if(pderi == 1) //partial derivative
		   dmath_vector_gpu(math_op, len, result->x, aphal);
		return  result;
	  }
	
	//---------------------
	constant<T>* copy()
	{
		this->copynum+=1;
		constant<T>* result=constant<T>::getObject(this->con_name+"_"+to_string(this->copynum), this->device, this->x_dim_num, this->x_dim, this->x);
		return result;
	}

	constant<T>* scala_mul(T beta)
	{   
		constant<T>* result =this->copy();
		T alpha1 = 0;
		T alpha2 = 0;
		constant<T>::op_math(CONSTANT_OP_MUL, this, this, result, &alpha1, &alpha2, &beta);
		return result;
	}

	//cudnnReduceTensor
	static void op_math_reduce(constant_math_op_reduce math_op, constant<T>* A,constant<T>* C, T*alpha,T* beta,int* result)
	{
		if (A->device == 0 || C->device == 0)
		{
			printf("constant->x must be localed on gpu!");
			exit(0);
		}

		cudnnReduceTensorOp_t cudnn_op_math_reduce;
		cudnnReduceTensorIndices_t  ReduceTensorIndices;//only  max, min used
		cudnnNanPropagation_t  cudnnNanPropagation;

		//CUDNN_NOT_PROPAGATE_NAN = 0,
	    //CUDNN_PROPAGATE_NAN = 1,

		switch (math_op)
		{
		case 0:
			cudnnNanPropagation = CUDNN_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_ADD;
			break;
		case 1:
			cudnnNanPropagation = CUDNN_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_MUL;
			break;
		case 2:
			cudnnNanPropagation = CUDNN_NOT_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_MIN;
			break;
		case 3:
			cudnnNanPropagation = CUDNN_NOT_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_MAX;
			break;
		case 4:
			cudnnNanPropagation = CUDNN_NOT_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_AMAX;//abs max
			break;
		case 5:
			cudnnNanPropagation = CUDNN_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_AVG;//only A useed 2=2^0.5=1.14...
			break;
		case 6:
			cudnnNanPropagation = CUDNN_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_NORM1;//only A useed ,eg::5=> -5+1= -4
			break;
		case 7:
			cudnnNanPropagation = CUDNN_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_NORM2;//only A useed ,eg::5=> -5+1= -4
			break;
		case 8:
			cudnnNanPropagation = CUDNN_PROPAGATE_NAN;
			ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
			cudnn_op_math_reduce = CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;//only A useed ,eg::5=> -5+1= -4
			break;
		default:
			cout << "wrong constant<T> math reduce op type " << endl;
		}

		cudnnHandle_t handle_;
		checkCudnnErr(cudnnCreate(&handle_));

		cudnnTensorDescriptor_t cudnnDescA;
		checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnDescA));

		cudnnTensorDescriptor_t cudnnDescC;
		checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnDescC));

		//typedef enum {
		//	CUDNN_DATA_FLOAT = 0,
		//	CUDNN_DATA_DOUBLE = 1,
		//	CUDNN_DATA_HALF = 2,
		//	CUDNN_DATA_INT8 = 3,
		//	CUDNN_DATA_INT32 = 4,
		//	CUDNN_DATA_INT8x4 = 5,
		//	CUDNN_DATA_UINT8 = 6,
		//	CUDNN_DATA_UINT8x4 = 7,
		//	CUDNN_DATA_INT8x32 = 8,
		//} cudnnDataType_t;

		cudnnDataType_t dataType;
		if (typeid(T) == typeid(float))
		{
			dataType = CUDNN_DATA_FLOAT;
		}

		if (typeid(T) == typeid(double))
		{
			dataType = CUDNN_DATA_DOUBLE;
		}

		if (typeid(T) == typeid(half1))
		{
			dataType = CUDNN_DATA_HALF;
		}

		if (typeid(T) == typeid(char))
		{
			dataType = CUDNN_DATA_INT8;
		};

		checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnDescA, dataType, A->x_dim_num, A->x_dim, A->x_stride));
		checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnDescC, dataType, C->x_dim_num, C->x_dim, C->x_stride));

		//createReduce descriptor
		cudnnReduceTensorDescriptor_t reduceTensorDesc;
		checkCudnnErr(cudnnCreateReduceTensorDescriptor(&reduceTensorDesc));

		//setReduce descriptor
		//cudnnGetReduceTensorDescriptor()
		checkCudnnErr(cudnnSetReduceTensorDescriptor(
			reduceTensorDesc,
			cudnn_op_math_reduce,
			dataType,
			cudnnNanPropagation,
			ReduceTensorIndices,
			CUDNN_32BIT_INDICES));

		//computer IndicesSize
		size_t sizeInBytes_Indices;
		checkCudnnErr(cudnnGetReductionIndicesSize(
			handle_,
			reduceTensorDesc,
			cudnnDescA,
			cudnnDescC,
			&sizeInBytes_Indices));
		void* indices;
		if(cudnn_op_math_reduce== CUDNN_REDUCE_TENSOR_MIN || cudnn_op_math_reduce == CUDNN_REDUCE_TENSOR_MAX ||
		   cudnn_op_math_reduce == CUDNN_REDUCE_TENSOR_AMAX)
		{
			//indices_num = sizeInBytes_Indices * 8 / 32;
			printf("indices:%d \n", sizeInBytes_Indices * 8 / 32);
			checkCudaErr(cudaMallocManaged((void**)&indices, sizeInBytes_Indices));
		}
		else {
			checkCudaErr(cudaMallocManaged((void**)&indices, 1));
		}
		
		//computer worspace size
		size_t sizeInBytes_Workspace;
		checkCudnnErr(cudnnGetReductionWorkspaceSize(
			handle_,
			reduceTensorDesc,
			cudnnDescA,
			cudnnDescC,
			&sizeInBytes_Workspace));
		void* workspace;
		checkCudaErr(cudaMallocManaged((void**)&workspace,sizeInBytes_Workspace));
		//printf("workspace:%d \n", sizeInBytes_Workspace);
		
		////C = alpha * reduce op(A)+ beta * C
		checkCudnnErr(cudnnReduceTensor(
			handle_,
			reduceTensorDesc,
			indices,
			sizeInBytes_Indices,
			workspace,
			sizeInBytes_Workspace,
			alpha,
			cudnnDescA,
			A->x,
			beta,
			cudnnDescC,
			C->x));
		checkCudaErr(cudaDeviceSynchronize());	
		if (cudnn_op_math_reduce == CUDNN_REDUCE_TENSOR_MIN || cudnn_op_math_reduce == CUDNN_REDUCE_TENSOR_MAX ||
			cudnn_op_math_reduce == CUDNN_REDUCE_TENSOR_AMAX)
		{
			memcpy(result, indices, sizeInBytes_Indices);
			//result = (int*)indices;
			printf("open! \n");
		}
	clean:
	}

	static void op_math(constant_math_op math_op,constant<T>* A, constant<T>* B, constant<T>* C, T*alpha1, T*alpha2, T* beta)
	{
		if (A->device == 0 || B->device == 0 || C->device == 0)
		{   printf("constant->x must be localed on gpu!");
			//exit(0);
		}
		cudnnOpTensorOp_t cudnn_op_math;

		switch (math_op)
		{
		case 0:
			cudnn_op_math = CUDNN_OP_TENSOR_ADD;
			break;
		case 1:
			cudnn_op_math = CUDNN_OP_TENSOR_MUL;
			break;
		case 2:
			cudnn_op_math = CUDNN_OP_TENSOR_MIN;
			break;
		case 3:
			cudnn_op_math = CUDNN_OP_TENSOR_MAX;
			break;
		case 4:
			cudnn_op_math = CUDNN_OP_TENSOR_SQRT;//only A useed 2=2^0.5=1.14...
			break;
		case 5:
			cudnn_op_math = CUDNN_OP_TENSOR_NOT;//only A useed ,eg::5=> -5+1= -4
			break;
		default:
			cout << "wrong constant<T> math op type " << endl;
			exit(0);
		}

		cudnnHandle_t handle_;
		checkCudnnErr(cudnnCreate(&handle_));

		cudnnTensorDescriptor_t cudnnDescA;
		checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnDescA));

		cudnnTensorDescriptor_t cudnnDescB;
		checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnDescB));

		cudnnTensorDescriptor_t cudnnDescC;
		checkCudnnErr(cudnnCreateTensorDescriptor(&cudnnDescC));

		//typedef enum {
		//	CUDNN_DATA_FLOAT = 0,
		//	CUDNN_DATA_DOUBLE = 1,
		//	CUDNN_DATA_HALF = 2,
		//	CUDNN_DATA_INT8 = 3,
		//	CUDNN_DATA_INT32 = 4,
		//	CUDNN_DATA_INT8x4 = 5,
		//	CUDNN_DATA_UINT8 = 6,
		//	CUDNN_DATA_UINT8x4 = 7,
		//	CUDNN_DATA_INT8x32 = 8,
		//} cudnnDataType_t;

		cudnnDataType_t dataType;
		if (typeid(T) == typeid(float))
		{
			dataType = CUDNN_DATA_FLOAT;
		}

		if (typeid(T) == typeid(double))
		{
			dataType = CUDNN_DATA_DOUBLE;
		}

		if (typeid(T) == typeid(half1))
		{
			dataType = CUDNN_DATA_HALF;
		}

		if (typeid(T) == typeid(char))
		{
			dataType = CUDNN_DATA_INT8;
		};

		checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnDescA, dataType, A->x_dim_num, A->x_dim, A->x_stride));
		checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnDescB, dataType, B->x_dim_num, B->x_dim, B->x_stride));
		checkCudnnErr(cudnnSetTensorNdDescriptor(cudnnDescC, dataType, C->x_dim_num, C->x_dim, C->x_stride));
		
		cudnnOpTensorDescriptor_t optype;
		checkCudnnErr(cudnnCreateOpTensorDescriptor(&optype));
		cudnnSetOpTensorDescriptor(optype, cudnn_op_math, dataType, CUDNN_NOT_PROPAGATE_NAN);
		//C = op(alpha1[0] * A, alpha2[0] * B) + beta[0]*C
		checkCudnnErr(cudnnOpTensor(
			handle_,
			optype,
			alpha1,
			cudnnDescA,
			A->x,
			alpha2,
			cudnnDescB,
			B->x,
			beta,
			cudnnDescC,
			C->x));
		
		checkCudaErr(cudaDeviceSynchronize());
	 clean:
	}

	//init in cpu=0,init in gpu=1
	constant(){};
	constant(string con_name_o,int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {
		device = device_o;
		x_dim_num = x_dim_num_o;
		con_name = con_name_o;

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

	static constant<T>* getObject(string con_name_o,int device_o, int x_dim_num_o, int *x_dim_o, T* x_src)
	{
		constant<T>* r=new constant<T>;
		r->con_name = con_name_o;
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
			checkCudaErrors(cudaMemcpy(r->x, x_src, length * sizeof(T),cudaMemcpyDefault));
		}
		return r;
	}
	
	// look like tf.placeholder()
	static constant<T>* getPlaceholder(string con_name_o, int device_o, int x_dim_num_o, int *x_dim_o)
	{
		constant<T>* r = new constant<T>;
		r->con_name = con_name_o;
		r->device = device_o;
		r->x_dim_num = x_dim_num_o;
		r->placeholder = 1;// mark  r is a plaeholder constant
		return r;
	}

	void initPlaceholder(int x_dim_num_o, int *x_dim_o, T* x_src) {
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

		if (this->device == 0) {
			this->x = (T*)malloc(length * sizeof(T));
			memcpy(this->x, x_src, length * sizeof(T));
		}
		else {
			checkCudaErrors(cudaMallocManaged((void**)&this->x, length * sizeof(T)));
			checkCudaErrors(cudaMemcpy(this->x, x_src, length * sizeof(T), cudaMemcpyDefault));
		}
	}
};

#endif // !_CONSTANT_CLASS_CUH
