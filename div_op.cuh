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

#ifndef _DIV_OP_CUH
#define _DIV_OP_CUH
template<class T>
class div_op :public base_op<T>
{ 
public:
	static div_op<T>* getObejct(base_op<T>* op1, T alpha_o, base_op<T>* op2, string name_o, char* Tensor_des = "")
	{
		//assume size must be same
		div_op<T>* result = new div_op<T>;
		result->alpha = alpha_o;
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

		div_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() {
		cout << "backward start::" << this->name_of_op << endl;
		//transport dy to dx
		for (int i = 0; i < this->sons_num; i++)
		{   //find the index of sons->father
			vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
			int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

			//self->dy=son->dx
			this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
		}

		this->sum_dy();
		T beta = 0;
		T apla1 = 1.0;
		T apla2 = 1.0;
		int len=(vector<constant<T>*>(*this->dx)).size();
		for (int i=0;i<len;i++)
		{     //iter is a father->xd;
			if(i == 1)
			 {   
				constant<T>* temp_const = ((constant<T>*)(*(this->x))[1])->function_tensor(CONS_REC, 1, 1);
				constant<T>::op_math(CONSTANT_OP_MUL, temp_const,this->dy_sum, this->dy_sum, &apla1, &apla2, &beta);
				(vector<constant<T>*>(*this->dx))[i] = ((constant<T>*)(this->dy_sum))->scala_mul(this->alpha);
				temp_const->clear();
			 }

			if (i == 0)
			{
				constant<T>* temp_const = ((constant<T>*)(*(this->x))[1])->function_tensor(CONS_REC, 1, 0);
				constant<T>::op_math(CONSTANT_OP_MUL, temp_const,this->dy_sum, this->dy_sum, &apla1, &apla2, &beta);
				(vector<constant<T>*>(*this->dx))[i] = ((constant<T>*)(this->dy_sum))->scala_mul(this->alpha);
				temp_const->clear();
			}
		}

		backward_over = 1;
		cout << "backward end::" << this->name_of_op << endl;
	}
	
	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() {
		//from this->x computer this->y
		int i = 0;
        T beta = 0;
		T apla1 = 1.0;
		T apla2 = 1.0;

		for (int i = 0; i < this->fathers_num; i++)
		{
			//self->x=father->y::fathers y be converted to this->x ,vector<constant<T>*>* x , vector<base_op<T>*> fathers
			(*(this->x))[i] = ((base_op<T>*)(this->fathers[i]))->y;
		}
		                                                            //1 no use  //0 ,dy/dx==1
		constant<T>* temp_const=((constant<T>*)(*(this->x))[1])->function_tensor(CONS_REC,1,0);
		constant<T>::op_math(CONSTANT_OP_MUL,temp_const,(constant<T>*)(*(this->x))[0], temp_const, &apla1, &apla2, &beta);
		((constant<T>*)(this->y)) =temp_const->scala_mul(this->alpha);
		temp_const->clear();
	
		forward_over = 1;
		cout << "forward::" << this->name_of_op << " y:" << this->y->x[0] << endl;
	}
};
#endif // !_DIV_OP_CUH
