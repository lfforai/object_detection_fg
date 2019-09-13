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
#include "hash_globle.cuh"
using namespace std;

#ifndef _BASE_OP_CUH
#define _BASE_OP_CUH

template<class T>
struct constant{
	//input from father 
	T*  x;//forward input
	int* x_stride;
	int  x_dim_num;//length of x_dim
	int* x_dim;
	int  device;
	
	//init in cpu=0,init in gpu=1
	void init(int device_o,int x_dim_num_o,int *x_dim_o,T* x_src){
		device = device_o;
		x_dim_num = x_dim_num_o;

		x_dim = (int *)malloc(x_dim_num * sizeof(int));
		x_stride = (int *)malloc(x_dim_num * sizeof(int));
		memcpy(x_dim, x_dim_o, x_dim_num * sizeof(int));
		
		if(x_dim_num > 1)
		  { x_stride[x_dim_num - 1] = 1;
		    for(int d = x_dim_num - 2; d >= 0; d--){
				x_stride[d] = x_stride[d + 1] * x_dim[d + 1];
		     }
		  }
		else{
			x_stride[x_dim_num - 1] = 1;
		}
		
		int length = x_stride[0] * x_dim[0];

		if (device == 0){
			x = (T*)malloc(length * sizeof(T));
			memcpy(x, x_src, length * sizeof(T));
		}
		else{
		   checkCudaErrors(cudaMemcpy(x,x_src, length * sizeof(T),cudaMemcpyHostToDevice));
		}
	}

	void clear() {
		free(x_stride);
		free(x_dim);
		if (device == 0){
			free(x);
		}
		else {
			checkCudaErrors(cudaFree(x));
		}
	}
};

template<class T>
struct x_dx_father{
	//input from father 
	T*  x;//forward input
	int* x_stride;
	int  x_dim_num;
	int  x_dim;

	// output to father
	T*   dx;//backward output
	int* dx_stride;
	int  dx_dim_num;
	int* dx_dim;
};

template<class T>
struct w_dw_now{
	T* dw=NULL;//backward  w
	int* dw_stride=NULL;
	int  dw_dim_num;
	int* dw_dim=NULL;

	T* w=NULL;//w
	int* w_stride=NULL;
	int  w_dim_num;
	int* w_dim=NULL;
	bool trainable=true;
};

template<class T>
struct y_dy_son{
	//input from son 
	T*    dy;//backward input
	int*  dy_stride;
	int   dy_dim_num;
	int   dy_dim;

	// output to son
	T*   y;//forward output
	int* y_stride;
	int  y_dim_num;
	int* y_dim;
};


template<class T>
class base_op{
 private:
	 
 public:
	 string name_of_op; //mast be unique;

	//N input op of fathers has  N  x_dy_father<T>[0]....x_dy_father<T>[N-1] parameters 
	std::vector<x_dx_father<T>*> xdx_father;

	//1 out op of son  has  1 
	y_dy_son<T>* ydy_son;

	//w
	w_dw_now<T>* w_dw;

	//constant<T>
	constant<T>* constant_N;

	graph<T, base_op>* globle_graph;

	//---------------------------
	std::vector<base_op<T>*> fathers;//up ops
	std::vector<base_op<T>*> sons;//down ops
	
	int fathers_num = 2; //number of the input ops=N
	int sons_num = 0; //number of the input ops=N

	//std::atomic_int forward_over(0);//if this object's all father are all ready(fathers->forward_condition==1),running this forward
	//std::atomic_int backward_over(0);//if this object's all son are all ready(sons->backward_condition==1),running this backward
	
	int forward_over = 0;
	int backward_over = 0;
	
   inline bool if_forward_start() {
		bool result = true;
		for (typename vector<base_op<T>*>::const_iterator iter = this->fathers.cbegin(); iter != this->fathers.cend(); iter++)
			{
				if ((*iter)->forward_over != 1)
				{
					result = false;
					break;
				}
			}
		return result;
	}

	inline bool if_backward_start() {
		bool result = true;
		//must have e tpyename
		for (typename vector<base_op<T>*>::const_iterator iter = this->sons.cbegin(); iter != this->sons.cend(); iter++)
		{
			if ((*iter)->backward_over!= 1)
			{
				result = false;
				break;
			}	   
		}
		return result;
	}
	
	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function(){

		backward_over = 1;
	}

   //reload the forward_function,make sure last of the function must be forward_over = 1
   virtual void forward_function(){
	   
	   forward_over = 1;
	}

	//run forward mark=0,else mark=1 run backward
	void ward_run(int mark) {
		//if(mark == 0) {
		//  
		//}
		//else {
		//
		//}
	}

	base_op(constant<T>* constant_N_o, graph<T, base_op>* globle_graph_o,string name_o)
		:name_of_op(name_o),constant_N(constant_N_o),globle_graph(globle_graph_o)
	{   //create op on grap
		this->globle_graph->insert_v(this->name_of_op, this);
	};

	//parameter=w_dw_o,constant=constant_o;
	base_op(base_op<T>* op, constant<T>* constant_N_o, graph<T, base_op>* globle_graph_o, string name_o) :
		name_of_op(name_o), constant_N(constant_N_o),globle_graph(globle_graph_o)
	{   //create op on graph
		this->fathers.push_back(op);
		op->sons.push_back(this);
		op->sons_num += 1;
		this->globle_graph->insert_v(this->name_of_op, this);
	};

	//parameter=w_dw_o,constant=constant_o;
	base_op(base_op<T>* op1,base_op<T>* op2,constant<T>* constant_N_o,graph<T, base_op>* globle_graph_o,string name_o):
		name_of_op(name_o),constant_N(constant_N_o),globle_graph(globle_graph_o)
	{   //create op on graph
 	    this->fathers.push_back(op1);
		this->fathers.push_back(op2);
		op1->sons.push_back(this);
		op1->sons_num += 1;
		op2->sons.push_back(this);
		op2->sons_num += 1;
		this->globle_graph->insert_v(this->name_of_op, this);
	};
    
	~base_op() {
	 
	};
};
//template class base_op<int>;
#endif // !_BASE_OP_CUH