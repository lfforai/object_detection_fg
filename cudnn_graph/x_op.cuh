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

//all varible and constant must be init by x_op
template<class T>
class x_op :public base_op<T>
{
public:
	//y=alpha*x
	static x_op<T>* getObejct(vector<constant<T>*>* constant_N_o,T alpha_o, string name_o)
	{   //constant_N_o.size()==1
		x_op<T>* result = new x_op<T>;
		result->alpha = alpha_o;
		result->name_of_op = name_o;
		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();
		result->neededBackwark_dw=false;
		result->neededBackwark_dx = false;

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}

		x_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//y=alpha*x
	static x_op<T>* getObejct(vector<varialbe<T>*>* w_o, T alpha_o, string name_o)
	{   //w_o.size()==1
		x_op<T>* result = new x_op<T>;
		result->alpha = alpha_o;
		result->name_of_op = name_o;
		result->w = w_o;
		result->w_num= w_o->size();
		
		result->neededBackwark_dx = false;

		for (typename vector<variable<T>*>::const_iterator iter = w_o->cbegin(); iter != w_o->cend(); iter++)
		{
			if ((*iter)->trainable == true)
				result->neededBackwark_dw = true;
			if ((*iter)->trainable == true && !base_op<T>::global_w_trainable->if_find((*iter)->var_name))
				base_op<T>::global_w_trainable->insert_v((*iter)->var_name, (*iter));
		}

		x_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	void backward_function() {
		//transport dy to dx
		if(this->neededBackwark_dw==true);
		{  this->sum_dy();
		   this->dw = this->dy_sum->scala_mul(this->alpha);
		}
		backward_over = 1;
		cout <<"backward::"<<this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	void forward_function() 
	   { //from this->x computer this->y
		if (result->neededBackwark_dw == false)
		{   //input is constant
			this->y = ((constant<T>*)((*(this->cons))[0]))->scala_mul(this->alpha);
		}
		else {
			//input is varible
			this->y = ((constant<T>*)((*(this->w))[0]))->scala_mul(this->alpha);
		}
		 forward_over = 1;
		 cout <<"forward::"<<this->name_of_op << endl;
	   }
};
#endif // !_SUM_OP_CUH