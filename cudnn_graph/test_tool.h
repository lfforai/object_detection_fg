#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
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
#include <thread>         // std::thread 
#include <ctime> 
#include "cublas.h"

#include <windows.h>
#include <wincrypt.h>
#include "base_op.cuh"
#include "constant_class.cuh"
#include "weigh_class.cuh"
#include "x_op.cuh"
#include "exp_op.cuh"
#include "sum_op.cuh"
#include "div_op.cuh"
#include "mul_op.cuh"
#include "sin_op.cuh"
#include "cos_op.cuh"
#include "image.h"
#include "paramter.h"

#include "python.h"
using namespace std;
using namespace cv;

template<class T>

void ShowVec(const vector<T>& valList)
{
	for (typename vector<T>::const_iterator iter = valList.cbegin(); iter != valList.cend(); iter++)
	{
		cout << (*iter) << endl;
	}
}

//how to use mutex,no mater who use mutex
class testone {
	
public:	
	testone(){
	};
	int a = 0;
	mutex m;

	void func(int i) {
		printf("func  %d\n",i);
	};

	void sub_test(){
		m.lock();
		func(-1);
		a -= 1;
		printf("a: \n",a);
		m.unlock();
	}

	void add_test(int id){
		cout<<"my="<<id<<endl;
		m.lock();
		func(id);
		float secs =300;      //定义浮点型变量secs
		clock_t delay;  //定义clock_t类型的变量，表示延时时间
		delay = secs * CLOCKS_PER_SEC;  //delay赋值为secs 乘以 CLOCKS_PER_SEC值，将输入的秒数转化系统的时间
		clock_t start = clock();    //定义clock_t类型变量start，并赋值为当前系统的时间
		while (clock() - start < delay);
		for (size_t i = 0; i < 100; i++)
		{
			a += 1;
		}
		m.unlock();
	}

	void printlast() {
		cout<<a<<endl;
	}
};

class loadoperatrion{
public:
	string name;
	loadoperatrion(){
		this->name = "first";
	};

	loadoperatrion(string name_o){
		this->name=name_o;
	};

	friend loadoperatrion  operator+(loadoperatrion& op,loadoperatrion& po2)
	{
		loadoperatrion result;
		result.name = op.name + "+" + po2.name;
		printf("start + \n");
		return result;
	}

	friend loadoperatrion  operator/(loadoperatrion& op,loadoperatrion& po2) 
	{
		loadoperatrion result;
		result.name = op.name+"/"+ po2.name;
		printf("start / \n");
		return result;
	}

	friend loadoperatrion  operator*(loadoperatrion& op,loadoperatrion& op2) {
		loadoperatrion result;
		result.name = op.name + "*" + op2.name;
		printf("* start \n");
		return result;
	}
};
int* find_dy_sum(int* dim_A, int * dim_C, int dim_num = 4)//only support dim_num=4
{   
	int sum_A = 1;
	int * stride_A = (int *)malloc(dim_num * sizeof(int));
	stride_A[0] = 1;
	for (int i = 0; i < dim_num; i++)
	{
		sum_A = sum_A * dim_A[i];
		if (i < dim_num - 1)
			stride_A[i + 1] = sum_A;
	}

	int sum_C = 1;
	int * stride_C = (int *)malloc(dim_num * sizeof(int));
	stride_C[0] = 1;
	for (int i = 0; i < dim_num; i++)
	{
		sum_C = sum_C * dim_C[i];
		if (i < dim_num - 1)
			stride_C[i + 1] = sum_C;
	}

	//sum_C is smaller than sum_A ,by reduced
	int * result = (int *)malloc(sum_A * sizeof(int));

#define A_index(N,H,W,C) result[N*stride_A[0]+ H * stride_A[1]+ W * stride_A[2]+ C * stride_A[3]]
#define C_index(N,H,W,C) N*stride_C[0]+ H * stride_C[1]+ W * stride_C[2]+ C * stride_C[3]

	//NHWC
	int c_N, c_H, c_W, c_C;
	for (int i_N = 0; i_N < dim_A[0]; i_N++)
	{
		for (int i_H = 0; i_H < dim_A[1]; i_H++)
		{
			for (int i_W = 0; i_W < dim_A[2]; i_W++)
			{
				for (int i_C = 0; i_C < dim_A[3]; i_C++)
				{
					c_N = i_N < (dim_C[0] - 1) ? i_N : (dim_C[0] - 1);
					c_H = i_H < (dim_C[1] - 1) ? i_H : (dim_C[1] - 1);
					c_W = i_W < (dim_C[2] - 1) ? i_W : (dim_C[2] - 1);
					c_C = i_C < (dim_C[2] - 1) ? i_C : (dim_C[3] - 1);
					A_index(i_N, i_H, i_W, i_C) = C_index(c_N, c_H, c_W, c_C);
				}
			}
		}
	}
	return result;
}


void tensor_reduce_test()
{   
	//int dimA[4] = { 2,1,4,2 };
	//int dimC[4] = {2,1,1,1 };
	//float srcA[16] ={1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8};
	//float srcC[2] ={1,1};
	//constant<float >* A=constant<float>::getObject("A", 1, 4,dimA,srcA);
	//constant<float >* C= constant<float>::getObject("B", 1, 4, dimC, srcC);
	////constant_math_op_reduce math_op, constant<T>* A,constant<T>* C, T*alpha,T* beta,int* result
	//float alpha = 1.0;
	//float beta = 0.0;
	//int* result=(int *)malloc(sizeof(int)*10);
	//constant<float>::op_math_reduce(CONS_REDUCE_TENSOR_ADD,A,C,&alpha,&beta,result);
	//int* dim_r=find_dy_sum(dimA, dimC);
	//free(result);
	//Py_Initialize();
	//PyRun_SimpleString("print('hello python!') \n");
	//Py_Finalize();
	//Mat a = cv::imread("C:/Users/Administrator/Desktop/lena.jpg");
	//cv::imshow("a",a);
	//uchar* c=a.ptr<uchar>();

	struct cudaDeviceProp prop;
	int device = 0;
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	double globalMem = prop.totalGlobalMem / double(1024 * 1024);

	image<half1>* image_o=new image<half1>;
	image_o->readImage("C:/Users/Administrator/Desktop/lena.jpg");	
	image_o->imgData_h;
	cout << "globalMem::" << globalMem << endl;
    
	float* par;
	cudaMallocManaged((void** )&par,10000000*sizeof(float));
	for (int i = 0; i < 10000000; i++)
	{
		par[i] = (float)i;
	}
	Layer_save_t<float> save("D:/dataimage/paramter/first.bin", par,10000000,FP16_HOST);

	Layer_load_t<half1>* load=new  Layer_load_t<half1>("D:/dataimage/paramter/first.bin",10000000, FP16_CUDA);
	cout << cpu_half2float(load->data_d[1000]) << endl;
	cout << cpu_half2float(load->data_h[1000])<< endl;
	waitKey(0);
};

void test() {

    
}
