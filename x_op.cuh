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
#include "queue.cuh"


using namespace std;
#ifndef _X_OP_CUH
#define _X_OP_CUH

//all varible and constant must be init by x_op
template<class T>
class x_op:public base_op<T>
{
public:
	//one constant,alpha=1

	static x_op<T>*  convert_Placeholder_to_x_op(string cons_name_o, int device_o, int x_dim_num_o, int *x_dim_o) {
		constant<T>* con_temp = constant<T>::getPlaceholder(cons_name_o, device_o, x_dim_num_o, x_dim_o);
		x_op<T>* result = x_op<T>::getconObejct(con_temp, 1, cons_name_o);
		return result;
	};

	static x_op<T>*  convert_cons_to_x_op(string cons_name_o,int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {
		constant<T>* con_temp=constant<T>::getObject(cons_name_o, device_o,x_dim_num_o,x_dim_o,x_src);
		x_op<T>* result = x_op<T>::getconObejct(con_temp, 1, cons_name_o);
		return result;
	};

	//one variable,alpha=1
	static x_op<T>*  convert_var_to_x_op(bool trainble_o, string var_name_o, int device_o, int x_dim_num_o, int *x_dim_o, T* x_src) {
		variable<T>* thisobj= variable<T>::getObject(trainble_o, var_name_o,device_o,x_dim_num_o,x_dim_o, x_src);
		x_op<T>* result = x_op<T>::getvarObejct(thisobj, 1,var_name_o);
		return result;
	};

	//y=alpha*x
	//vector constant
	static x_op<T>* getconObejct(constant<T>* constant_N_o,T alpha_o, string name_o)
	{   //constant_N_o.size()==1
		x_op<T>* result = new x_op<T>;
		result->alpha = alpha_o;
		result->name_of_op = name_o;
		result->cons = constant_N_o;
		result->neededBackwark_dw=false;
		result->neededBackwark_dx=false;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
        if (((constant<T>*)result->cons)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find(((constant<T>*)result->cons)->con_name))
		   base_op<T>::global_placehold_constant->insert_v(((constant<T>*)result->cons)->con_name, result->cons);
		x_op<T>::global_graph->insert_v_repeat(result->name_of_op, result);
		return result;
	}

	//y=alpha*x
	//veector varible
	static x_op<T>* getvarObejct(variable<T>* w_o, T alpha_o, string name_o)
	{   //w_o.size()==1
		x_op<T>* result = new x_op<T>;
		result->alpha = alpha_o;
		result->name_of_op = name_o;
		result->w = w_o;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		
		result->neededBackwark_dx = false;

		if (result->w->trainable == false)
			result->neededBackwark_dw = false;

		if (result->w->trainable == true && !base_op<T>::global_w_trainable->if_find(((variable<T>*)result->w)->var_name))
		{
			base_op<T>::global_w_trainable->insert_v(((variable<T>*)result->w)->var_name, result->w);
			result->dw = (result->w)->copy_zero();
			base_op<T>::global_dw_trainable->insert_v(((variable<T>*)result->w)->var_name, result->dw);
			
		}

		x_op<T>::global_graph->insert_v_repeat(result->name_of_op, result);//op insert to global_graph
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function(){
		cout << "backward x_op start::" << this->name_of_op << endl;
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
		   constant<T>::op_math(CONSTANT_OP_ADD,this->dw,this->dy_sum,this->dw, &this->alpha, &apla2, &beta);
		}

		total_not_finish_ops_num -= 1;
		this->backwardover = 1;

		for(int i = 0; i < this->fathers_num; i++)
		  {
			((base_op<T>*)(this->fathers[i]))->sons_finshed_size -= 1;//father finished
			if (((base_op<T>*)(this->fathers[i]))->sons_finshed_size == 0 && ((base_op<T>*)(this->fathers[i]))->backwardover!=1)
			    //((threadsafe_queue<string>*) base_op<T>::queue_forward_canbe_used_ops)->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
				queue_forward_canbe_used_ops->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
		  }

		cout << "backward x_op over::" << this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() 
	   { //from this->x computer this->y
		if (this->neededBackwark_dw == false)
			{   //input is constant
				this->y = this->cons;
			}
		else {
				//input is variable
				this->y = this->w;
			}
		
		total_not_finish_ops_num -= 1;
		this->forwardover = 1;

		for (int i = 0; i < this->sons_num; i++)
		    {
				((base_op<T>*)(this->sons[i]))->fathers_finshed_size -= 1;//father finished
				if (((base_op<T>*)(this->sons[i]))->fathers_finshed_size == 0 && ((base_op<T>*)(this->sons[i]))->forwardover != 1)
					queue_forward_canbe_used_ops->push(((base_op<T>*)(this->sons[i]))->name_of_op);
			}
		cout << "forward::" << this->name_of_op << " y:" << this->y->x[0] << " y:" << this->y->x[1] << endl;
	   }
};
#endif // !_SUM_OP_CUH