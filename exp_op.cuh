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

#ifndef _EXP_OP_CUH
#define _EXP_OP_CUH
template<class T>
class exp_op :public base_op<T>
{
public:
	static exp_op<T>* getObejct(base_op<T>* op1, T aphla_o,string name_o, char* Tensor_des = "")
	{
		//assume size must be same
		exp_op<T>* result = new exp_op<T>;
		result->alpha = aphla_o;
		result->name_of_op = name_o;
		result->x =  new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;

		result->fathers.push_back(op1);
		result->fathers_name.push_back(op1->name_of_op);
		result->fathers_num = 1;
		result->xdx_num += 1;

		op1->sons.push_back(result);
		op1->sons_name.push_back(result->name_of_op);
		op1->sons_num += 1;
		op1->ydy_num += 1;

		exp_op<T>::global_graph->insert_v_repeat(result->name_of_op, result);
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() {
		cout << "backward exp start::" << this->name_of_op << endl;
		if (this->sons_num > 0)
		{   // have sons;
		   //transport dy to dx
			for (int i = 0; i < this->sons_num; i++)
			{   //find the index of sons->father
				vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
				int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

				//self->dy=son->dx
				this->dy->push_back((*(this->sons[i])->dx)[index]);
			}
			this->dx;
			this->sum_dy();

			T beta = 0;
			T apla1 = 1.0;
			T apla2 = 1.0;

			(*this->dx)[0] = ((*this->x)[0])->function_tensor(CONS_EXP, 1, 1);
			 constant<T>::op_math(CONSTANT_OP_MUL, (*this->dx)[0], this->dy_sum, (*this->dx)[0], &apla1, &apla2, &beta);
		}
		else {//no sons
			(*this->dx)[0] = ((*this->x)[0])->function_tensor(CONS_EXP, 1, 1);
		}

		total_not_finish_ops_num -= 1;
		this->backwardover = 1;

		for (int i = 0; i < this->fathers_num; i++)
		{
			((base_op<T>*)(this->fathers[i]))->sons_finshed_size -= 1;//father finished
			if (((base_op<T>*)(this->fathers[i]))->sons_finshed_size == 0 && ((base_op<T>*)(this->fathers[i]))->backwardover == -1)
				queue_forward_canbe_used_ops->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
				//((threadsafe_queue<string>*) base_op<T>::queue_forward_canbe_used_ops)->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
		}

		cout << "backward exp over::" << this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() {
		//from this->x computer this->y
		for (int i = 0; i < this->fathers_num; i++)
		{
			//self->x=father->y::fathers y be converted to this->x ,vector<constant<T>*>* x , vector<base_op<T>*> fathers
			(*(this->x))[i] = ((base_op<T>*)(this->fathers[i]))->y;
		}
		//1 no use  //0 ,dy/dx==1
		this->y = ((constant<T>*)(*(this->x))[0])->function_tensor(CONS_EXP, 1, 0);
		
		total_not_finish_ops_num -= 1;
		this->forwardover = 1;

		for (int i = 0; i < this->sons_num; i++)
			{
				((base_op<T>*)(this->sons[i]))->fathers_finshed_size -= 1;//father finished
				if (((base_op<T>*)(this->sons[i]))->fathers_finshed_size == 0 && ((base_op<T>*)(this->sons[i]))->forwardover != 1)
					queue_forward_canbe_used_ops->push(((base_op<T>*)(this->sons[i]))->name_of_op);
				//((threadsafe_queue<string>*) base_op<T>::queue_forward_canbe_used_ops)->push(((base_op<T>*)(this->sons[i]))->name_of_op);
			}
		cout << "forward::" << this->name_of_op << " y:" << this->y->x[0] << " y:" << this->y->x[1] << endl;
	}
};
#endif