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
#include "constant_class.cuh"
#include "weigh_class.cuh"
#include "base_op.cuh"

#include <windows.h>
#include <wincrypt.h>
#include "stdarg.h"


using namespace std;
#ifndef _SUM_OP_CUH
#define _SUM_OP_CUH
template<class T>
class sum_op :public base_op<T>
{
public:
	T aphla_sum1;
	T aphla_sum2;
	static sum_op<T>* getObejct(base_op<T>* op1,T aphla1,T aphla2,base_op<T>* op2,string name_o, char* Tensor_des = "")
	{  
		//assume size must be same
		sum_op<T>* result = new sum_op<T>;
		result->aphla_sum1 = aphla1;
		result->aphla_sum2 = aphla2;
		result->name_of_op = name_o;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;


		result->fathers.push_back(op1);
		result->fathers_name.push_back(op1->name_of_op);
		result->fathers.push_back(op2);
		result->fathers_name.push_back(op2->name_of_op);
		result->fathers_num = 2;
		result->xdx_num += 2;

		op1->sons.push_back(result);
		op1->sons_name.push_back(result->name_of_op);
		op1->sons_num += 1;
		op1->ydy_num += 1;

		op2->sons.push_back(result);
		op2->sons_name.push_back(result->name_of_op);
		op2->sons_num += 1;
		op2->ydy_num += 1;

		sum_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}
	
	//reload the backward_function,make sure last of the function must be backward_over = 1
    void backward_function(){
		//transport dy to dx
		this->sum_dy();
		int i = 0;
		for (typename vector<constant<T>*>::const_iterator iter = this->dx->cbegin(); iter != this->dx->cend(); iter++)
		{     //iter is a father->xd;
			if (i == 0)
			{
				(*iter) = ((constant<T>*)(this->dy_sum))->scala_mul(this->aphla_sum1);
				i += 1;
			}
			if (i == 1)
				(*iter) = ((constant<T>*)(this->dy_sum))->scala_mul(this->aphla_sum2);
		}
		backward_over = 1;
		cout <<"backward::"<<this->name_of_op << endl;
	}
	
	//reload the forward_function,make sure last of the function must be forward_over = 1
	void forward_function(){
		//from this->x computer this->y
		int i = 0;
		T beta = 0;
		for(typename vector<constant<T>*>::const_iterator iter = this->x->cbegin(); iter != this->x->cend(); iter++)
		  { //assume op1.y.size==op2.y.size     
			if(i == 0)
			  { ((constant<T>*)(this->y))=((constant<T>*)(*iter));
				i += 1;
			  }
			else
			 {
				constant<T>::op_math(CONSTANT_OP_ADD,((constant<T>*)(this->y)),((constant<T>*)(*iter)),((constant<T>*)(this->y)), &this->aphla_sum1, &this->aphla_sum2, &beta);
			 }
		  }
		forward_over = 1;
		cout <<"forward::"<<this->name_of_op << endl;
	}
};
#endif // !_SUM_OP_CUH

