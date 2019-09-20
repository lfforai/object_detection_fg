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
#ifndef _X_OP_CUH
#define _X_OP_CUH

//all varible and constant must be init by x_op
template<class T>
class x_op:public base_op<T>
{
public:
	//one constant,alpha=1
	static x_op<T>*  convert_cons_to_x_op(string cons_name_o,int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {
		constant<T>* thisobj=constant<T>::getObject(cons_name_o, device_o, x_dim_num_o, x_dim_o, x_src);
		T alpha_o;
		vector<constant<T>*>* vector_temp = new vector<constant<T>*>;
		vector_temp->push_back(thisobj);
		x_op<T>* result = x_op<T>::getconObejct(vector_temp, 1, cons_name_o);
		return result;
	};

	//one variable,alpha=1
	static x_op<T>*  convert_var_to_x_op(bool trainble_o, string var_name_o, int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {
		variable<T>* thisobj= variable<T>::getObject(trainble_o, var_name_o,device_o,x_dim_num_o,x_dim_o, x_src);
		T alpha_o;
		vector<variable<T>*>* vector_temp = new vector<variable<T>*>;
		vector_temp->push_back(thisobj);
		x_op<T>* result = x_op<T>::getvarObejct(vector_temp, 1,var_name_o);
		return result;
	};

	//y=alpha*x
	//vector constant
	static x_op<T>* getconObejct(vector<constant<T>*>* constant_N_o,T alpha_o, string name_o)
	{   //constant_N_o.size()==1
		x_op<T>* result = new x_op<T>;
		result->alpha = alpha_o;
		result->name_of_op = name_o;
		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();
		result->neededBackwark_dw=false;
		result->neededBackwark_dx = false;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}

		x_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//y=alpha*x
	//veector varible
	static x_op<T>* getvarObejct(vector<variable<T>*>* w_o, T alpha_o, string name_o)
	{   //w_o.size()==1
		x_op<T>* result = new x_op<T>;
		result->alpha = alpha_o;
		result->name_of_op = name_o;
		result->w = w_o;
		result->w_num= w_o->size();
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;
		
		result->neededBackwark_dx = false;

		for (typename vector<variable<T>*>::const_iterator iter = w_o->cbegin(); iter != w_o->cend(); iter++)
		{
			if ((*iter)->trainable == false)
				result->neededBackwark_dw = false;

			if ((*iter)->trainable == true && !base_op<T>::global_w_trainable->if_find((*iter)->var_name))
				base_op<T>::global_w_trainable->insert_v((*iter)->var_name, (*iter));
		}
		result->dw = new vector<variable<T>*>;
		x_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function(){
		//cout << "backward  start::" << this->name_of_op << endl;
		//transport dy to dx
		T apla2 = 1;
		T beta = 1;
		if(this->neededBackwark_dw==true)
		{  //self------------------------------excharge-------------------------son
			for (int i = 0; i < this->sons_num; i++)
			{   //find the index of sons->father
				vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
				int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);
				//self->dy=son->dx
				this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
			}
			
		   this->sum_dy();
		   constant<T>::op_math(CONSTANT_OP_ADD,(*this->dw)[0],this->dy_sum,(*this->dw)[0], &this->alpha, &apla2, &beta);
		}
		backward_over = 1;
		//cout << "backward over::" << this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() 
	   { //from this->x computer this->y
		if (this->neededBackwark_dw == false)
		{   //input is constant
			this->y = ((constant<T>*)((*(this->cons))[0]))->scala_mul(this->alpha);
		}
		else {
			//input is variable
			this->y = ((variable<T>*)((*(this->w))[0]))->scala_mul(this->alpha);
		}
		forward_over = 1;
		//cout << "forward::" << this->name_of_op <<" y:"<<this->y->x[0]<<endl;
	   }
};
#endif // !_SUM_OP_CUH