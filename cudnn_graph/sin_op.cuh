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

#ifndef _SIN_OP_CUH
#define _SIN_OP_CUH
template<class T>
class sin_op :public base_op<T>
{
public:
	static sin_op<T>* getObejct(base_op<T>* op1, T aphla_o, string name_o, char* Tensor_des = "")
	{
		//assume size must be same
		sin_op<T>* result = new sin_op<T>;
		result->aphla = aphla_o;
		result->name_of_op = name_o;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;

		result->fathers.push_back(op1);
		result->fathers_name.push_back(op1->name_of_op);
		result->fathers_num = 1;
		result->xdx_num += 1;

		op1->sons.push_back(result);
		op1->sons_name.push_back(result->name_of_op);
		op1->sons_num += 1;
		op1->ydy_num += 1;

		sin_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function(){
		//transport dy to dx
		for (int i = 0; i < this->sons_num; i++)
		{   //find the index of sons->father
			vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
			int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

			//self->dy=son->dx
			this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
		}

		this->sum_dy();
		int i = 0;
		T beta = 0;
		T apla1 = 1.0;
		T apla2 = 1.0;

		constant<T>* temp_const = ((constant<T>*)(*(this->x))[1])->function_tensor(CONS_SIN, 1, 1);
		constant<T>::op_math(CONSTANT_OP_MUL, temp_const, this->dy_sum, this->dy_sum, &apla1, &apla2, &beta);
		(*(this->dx))[0] = ((constant<T>*)(this->dy_sum))->scala_mul(this->aphla);
		temp_const->clear();
		backward_over = 1;
		cout << "backward::" << this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() {
		//from this->x computer this->y
		for(int i = 0; i < this->fathers_num; i++)
		{
			//self->x=father->y::fathers y be converted to this->x ,vector<constant<T>*>* x , vector<base_op<T>*> fathers
			(*(this->x))[i] = ((base_op<T>*)(this->fathers[i]))->y;
		}

		int i = 0;
		T beta = 0;
		T apla1 = 1.0;
		T apla2 = 1.0;
		//1 no use  //0 ,dy/dx==1
		constant<T>* temp_const = ((constant<T>*)(*(this->x))[1])->function_tensor(CONS_SIN, 1, 0);
		((constant<T>*)(this->y)) = temp_const->scala_mul(this->aphla);
		temp_const->clear();
		forward_over = 1;
		cout << "forward::" << this->name_of_op << " y:" << this->y->x[0] << endl;
	}
};
#endif