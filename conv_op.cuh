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
#include "weigh_class.cuh"
#include "base_op.cuh"

#include "paramter.h"
#include <cudnn.h>
#include "net_tool.cuh"
#include "convert.h"

using namespace std;

#ifndef _CONV_OP_CUH
#define _CONV_OP_CUH



template<class T>
class conv_op :public base_op<T>
{

public:
	int n=0;          //sample num
	int c=0;          //inpput dim
	int k=0;          //outut dim
	int h_out_f=0;    //output height 
	int w_out_f=0;    //output wight  
	int h_filter=0;   //h_filter
	int w_filter=0;   //w_filter
	int h_in_f = 0;   //input height
	int w_in_f = 0;   //input width

	int  convAlgorithm=-1;
	int  convDims=2;
	int  filterStrideA[2]= { 1,1 }; //stride
	int  padA[2]= { 0,0 };          //pad
	int  upscaleA[2]= { 1,1 };      //dilat
	int  filterDimA[4] = {k,c,h_filter,w_filter};
	int  tensorOuputDimA[4] = {n,k,h_out_f,w_out_f};

	typedef typename ScaleFactorTypeMap<T>::Type scaling_type;

	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
	cudnnTensorDescriptor_t cudnndydesc, cudnndwdesc;

	cudnnHandle_t cudnnHandle;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnFilterDescriptor_t filterDesc;

	cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
	cudnnConvolutionMode_t ConvolutionMode = CUDNN_CROSS_CORRELATION;
	cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

	void createHandles()
	{
		checkCUDNN(cudnnCreate(&cudnnHandle));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	}
	
	void destroyHandles()
	{
		checkCUDNN(cudnnDestroyLRNDescriptor(normDesc));
		checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(activDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
		checkCUDNN(cudnnDestroy(cudnnHandle));
		checkCublasErrors(cublasDestroy(cublasHandle));
	}
	//Layer_save_t*  save_para;
	//Layer_load_t*  load_para;

	void resize(int size, T* data)
	{
		if (*data != NULL)
		{
			checkCudaErrors(cudaFree(data));
		}
		checkCudaErrors(cudaMallocManaged((void **)&data, size * sizeof(value_type)));
	}

	void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
	{
		convAlgorithm = (int)algo;
	}
	
	//k=output feature num
	static conv_op<T>* getObejct(base_op<T>* op1,string name_o,int k_o,int* filterDimA_o)
	{
		//assume size must be same
		conv_op<T>* result = new conv_op<T>;
		result->name_of_op = name_o;
		result->x  = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
	   
		result->k = k_o;
		result->h_filter = filterDimA_o[2];
		result->w_filter = filterDimA_o[3];

		//result->w =newvariale<T>(true, result->name_of_op, 1, convDims + 2, filterDimA, x_src);
		int len = filterDimA_o[0] * filterDimA_o[1] * filterDimA_o[2] * filterDimA_o[3];
		float* x_src_f32 = cuda_rand(len);
		Convert<T> vert;
		T* temp =(T*) malloc(len * sizeof(T));
		for (int i = 0; i < len; i++) {
			temp[i] = vert(x_src_f32[i]);
		}
		result->w=new variable<T>(true,name_o,1,4,filterDimA_o, temp);
		free(x_src_f32);
		free(temp);

		if (typeid(T) == typeid(float))
		   result->dataType = CUDNN_DATA_FLOAT;
		if (typeid(T) == typeid(double))
			result->dataType = CUDNN_DATA_DOUBLE;
		if (typeid(T) == typeid(half1))
			result->dataType = CUDNN_DATA_HALF;
	
		result->fathers.push_back(op1);
		result->fathers_name.push_back(op1->name_of_op);
		result->fathers_num = 1;
		result->xdx_num += 1;
		op1->sons.push_back(result);
		op1->sons_name.push_back(result->name_of_op);
		op1->sons_num += 1;
		op1->ydy_num += 1;

		result->neededBackwark_dx = true;

		if (result->w->trainable == false)
		   result->neededBackwark_dw = false;

		if (result->w->trainable == true && !base_op<T>::global_w_trainable->if_find(((variable<T>*)result->w)->var_name))
		{
			base_op<T>::global_w_trainable->insert_v(((variable<T>*)result->w)->var_name, result->w);
			result->dw = (result->w)->copy_zero();
			((variable<T>*)result->dw)->var_name = ((variable<T>*)result->dw)->var_name + "_dw";
			base_op<T>::global_dw_trainable->insert_v(((variable<T>*)result->w)->var_name, result->dw);
		}

		conv_op<T>::global_graph->insert_v_repeat(result->name_of_op, result);//op insert to global_graph
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() 
	{   //cout << "backward  cos start::" << this->name_of_op << endl;
		if(this->sons_num > 0)
		 {   // have sons;
		   //cout << "backward start::" << this->name_of_op << endl;
			for(int i = 0; i < this->sons_num; i++)
			  { //find the index of sons->father
				vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
				int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

				//self->dy=son->dx
				this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
			  }
			//transport dy to dx
			this->sum_dy();
		    
			checkCUDNN(cudnnCreate(&cudnnHandle));

			//dw -------------------------------start--------------------------------------------
			setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, this->h_in_f, this->w_in_f);//x
			setTensorDesc(cudnndydesc, tensorFormat, dataType, n, k, this->h_out_f, this->w_out_f);//dy
			setTensorDesc(cudnndwdesc, tensorFormat, dataType, k, c, this->h_filter, this->w_filter);//dw
		
			int filsize = filterDimA[0] * filterDimA[1] * filterDimA[2] * filterDimA[3];
			cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

			void *workSpace = 0;
			size_t workSpaceSize;

			checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc, cudnndydesc, convDesc,
				filterDesc, algo, &workSpaceSize));

			if (workSpaceSize > 0) {
				checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));
			}
	         
			float  alpha =1.0;
			float  beta =1.0;

			checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
				(void*)(&alpha),
				srcTensorDesc, ((*(this->x))[0])->x,//x
				cudnndydesc, this->dy_sum->x,//dy
				convDesc,
				algo,
				workSpace, workSpaceSize,
				(void*)(&beta),
				filterDesc, this->dw->x));//dw
		   //dw-----------------------over-----------------------------------------------------------

		   //dx----------------------------start-----------------------------------------------------
			(*this->dx)[0] =((constant<T>*)((*this->x)[0]))->copy2zeros();
			cudnnConvolutionBwdDataAlgo_t algo_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
			checkCudaErrors(cudaFree(workSpace));
			checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, cudnndydesc, convDesc, srcTensorDesc, algo_data, &workSpaceSize));
			if (workSpaceSize > 0) {
			   cudaMalloc(&workSpace, workSpaceSize);
			}
			checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
				(void*)(&alpha),
				filterDesc, this->w->x,
				cudnndydesc, this->dy_sum->x,
				convDesc,
				algo_data,
				workSpace, workSpaceSize,
				(void*)(&beta),
				srcTensorDesc, ((constant<T>*)((*this->dx)[0]))->x));
			checkCudaErrors(cudaDeviceSynchronize());
		   //dx----------------------------over-----------------------------------------------------

			if (workSpaceSize != 0)
			{
				checkCudaErrors(cudaFree(workSpace));
			}
		}
		else{
			checkCUDNN(cudnnCreate(&cudnnHandle));

			//dw -------------------------------start--------------------------------------------
			setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, this->h_in_f, this->w_in_f);//x
			setTensorDesc(cudnndydesc, tensorFormat, dataType, n, k, this->h_out_f, this->w_out_f);//dy
			setTensorDesc(cudnndwdesc, tensorFormat, dataType, k, c, this->h_filter, this->w_filter);//dw

			int filsize = filterDimA[0] * filterDimA[1] * filterDimA[2] * filterDimA[3];
			cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

			void *workSpace = 0;
			size_t workSpaceSize;

			constant<T>* temp = this->y->copy2ones();

			checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc, cudnndydesc, convDesc,
				filterDesc, algo, &workSpaceSize));

			if (workSpaceSize > 0) {
				checkCudaErrors(cudaMalloc(&workSpace, workSpaceSize));
			}

			float  alpha = 1.0;
			float  beta = 1.0;

			checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
				(void*)(&alpha),
				srcTensorDesc, ((*(this->x))[0])->x,//x
				cudnndydesc, temp->x,//dy=one
				convDesc,
				algo,
				workSpace, workSpaceSize,
				(void*)(&beta),
				filterDesc, this->dw->x));//dw
		   //dw-----------------------over-----------------------------------------------------------

		   //dx----------------------------start-----------------------------------------------------
			cudnnConvolutionBwdDataAlgo_t algo_data =  CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
			(*this->dx)[0] = ((constant<T>*)((*this->x)[0]))->copy2zeros();
			checkCudaErrors(cudaFree(workSpace));
			checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, cudnndydesc, convDesc, srcTensorDesc, algo_data, &workSpaceSize));
			if (workSpaceSize > 0) {
				cudaMalloc(&workSpace, workSpaceSize);
			}

			checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
				(void*)(&alpha),
				filterDesc, this->w->x,
				cudnndydesc, temp->x,//dy=one
				convDesc,
				algo_data,
				workSpace, workSpaceSize,
				(void*)(&beta),
				srcTensorDesc, ((constant<T>*)((*this->dx)[0]))->x));
			checkCudaErrors(cudaDeviceSynchronize());
			temp->clear();
		  //dx----------------------------over-----------------------------------------------------

			if (workSpaceSize != 0)
			{
				checkCudaErrors(cudaFree(workSpace));
			}
		}

		//------------------------------------------------------------------------------------
		checkCUDNN(cudnnDestroy(cudnnHandle));
		total_not_finish_ops_num -= 1;
		this->backwardover = 1;

		for (int i = 0; i < this->fathers_num; i++)
		{
			((base_op<T>*)(this->fathers[i]))->sons_finshed_size -= 1;//father finished
			if (((base_op<T>*)(this->fathers[i]))->sons_finshed_size == 0 && ((base_op<T>*)(this->fathers[i]))->backwardover != 1)
				queue_forward_canbe_used_ops->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
		}
		//cout << "backward cos over::" << this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() 
	{
		this->createHandles();
		(*this->x)[0] = ((base_op<T>*)(this->fathers[0]))->y;

		this->c = ((base_op<T>*)(this->fathers[0]))->y->x_dim[1];
		this->h_in_f = ((base_op<T>*)(this->fathers[0]))->y->x_dim[2];
		this->w_in_f = ((base_op<T>*)(this->fathers[0]))->y->x_dim[3];
		this->filterDimA[0]=this->k;
		this->filterDimA[1]=this->c;
		this->filterDimA[2]=this->h_filter;
		this->filterDimA[3]=this->w_filter;

		//------------------------------------------------------------------------------
		cudnnConvolutionFwdAlgo_t algo;

		const int tensorDims = 4;

		checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,
			dataType,
			tensorFormat,
			tensorDims,
			filterDimA));
		
		cudnnDataType_t  convDataType = dataType;
		
		if (dataType == CUDNN_DATA_HALF) {
			convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
		}

		checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc,
			convDims,
			padA,
			filterStrideA,
			upscaleA,
			ConvolutionMode,
			convDataType));

		//set n 
		n=((constant<T>*)((*this->x)[0]))->x_dim[0];
		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h_in_f, w_in_f);

		// find dimension of convolution output dim
		checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc,
			srcTensorDesc,
			filterDesc,
			tensorDims,
			tensorOuputDimA));
		
		if (this->tensorFormat == CUDNN_TENSOR_NCHW)
		{
			h_out_f = tensorOuputDimA[2]; w_out_f = tensorOuputDimA[3];
		}
		
		if (this->tensorFormat == CUDNN_TENSOR_NHWC)
		{
			h_out_f = tensorOuputDimA[1]; w_out_f = tensorOuputDimA[2];
		}

		if (this->tensorFormat == CUDNN_TENSOR_NCHW_VECT_C)
		{
			h_out_f = tensorOuputDimA[2]; w_out_f = tensorOuputDimA[3];
		}

		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, k, h_out_f, w_out_f);//y

		if (convAlgorithm < 0)
		{
			// Choose the best according to the preference
			std::cout << "Testing cudnnGetConvolutionForwardAlgorithm ...\n";
			checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				srcTensorDesc,
				filterDesc,
				convDesc,
				dstTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&algo
			));
			std::cout << "Fastest algorithm is Algo " << algo << "\n";
			convAlgorithm = algo;
			// New way of finding the fastest config
			// Setup for findFastest call
			std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
			int requestedAlgoCount = 5;
			int returnedAlgoCount[1];
			cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
			checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
				srcTensorDesc,
				filterDesc,
				convDesc,
				dstTensorDesc,
				requestedAlgoCount,
				returnedAlgoCount,
				results
			));
			for (int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex) {
				printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
			}
			free(results);
		}
		else
		{
			algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
			if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
			{
				std::cout << "Using FFT for convolution\n";
			}
		}

		T* temp;
		checkCudaErrors(cudaMallocManaged((void **)&temp,n*k*h_out_f*w_out_f*sizeof(T)));
		this->y=constant<T>::getObject(this->name_of_op+"_cons",1,tensorDims,tensorOuputDimA,temp);
		checkCudaErrors(cudaFree(temp));

		size_t sizeInBytes = 0;
		void* workSpace = NULL;
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			srcTensorDesc,
			filterDesc,
			convDesc,
			dstTensorDesc,
			algo,
			&sizeInBytes));
		
		if(sizeInBytes != 0)
		{
			checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes));
		}

		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		
		//run forward
		checkCUDNN(cudnnConvolutionForward(cudnnHandle,
			&alpha,
			srcTensorDesc,
			((*(this->x))[0])->x,
			filterDesc,
			(this->w)->x,
			convDesc,
			algo,
			workSpace,
			sizeInBytes,
			&beta,
			dstTensorDesc,
			(this->y)->x));

		checkCudaErrors(cudaDeviceSynchronize());
		if (sizeInBytes != 0)
		{
			checkCudaErrors(cudaFree(workSpace));
		}
		//------------------------------------------------------------------------------

		total_not_finish_ops_num -= 1;
		this->forwardover = 1;
		for (int i = 0; i < this->sons_num; i++)
		{
			((base_op<T>*)(this->sons[i]))->fathers_finshed_size -= 1;//father finished
			if (((base_op<T>*)(this->sons[i]))->fathers_finshed_size == 0 && ((base_op<T>*)(this->sons[i]))->forwardover != 1)
				queue_forward_canbe_used_ops->push(((base_op<T>*)(this->sons[i]))->name_of_op);
		}
	}
};
#endif