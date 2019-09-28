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

#include <windows.h>
#include <wincrypt.h>
#include "stdarg.h"


using namespace std;

#ifndef _RUDUCE_SUM_OP_CUH
#define _RUDUCE_SUM_OP_CUH

template<class T>
class reduce_sum_op	:public base_op<T>
{
private:
	//out  dx->dy_sum ,the postion in dy_sum
public:
	int* find_dx2dy_sum(int* dim_A, int * dim_C, int dim_num = 4)//only support dim_num=4
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

		//sum_C is smaller than sum_A ,by reduce_sumd
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
	
	int *dim_C;//c=[1,1,1,1] 

	static reduce_sum_op<T>* getObejct(base_op<T>* op1, int* dim_C_o,string name_o, char* Tensor_des = "")
	{
		//assume size must be same
		reduce_sum_op<T>* result = new reduce_sum_op<T>;
		result->alpha = 1.0;
		result->name_of_op = name_o;
		result->x = new vector<constant<T>*>;
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

		result->dim_C=dim_C_o;
		
		reduce_sum_op<T>::global_graph->insert_v_repeat(result->name_of_op, result);
		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() {
		//cout << "backward  cos start::" << this->name_of_op << endl;
		if(this->sons_num > 0)
		{   
			// have sons;
		    //cout << "backward start::" << this->name_of_op << endl;
			for (int i = 0; i < this->sons_num; i++)
			{   //find the index of sons->father
				vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
				int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

				//self->dy=son->dx
				this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
			}

			//transport dy to dx
			this->sum_dy();

			(*this->dx)[0] = ((constant<T>*)((*this->x)[0]))->copy2zeros();

			int* index_p=find_dx2dy_sum(((constant<T>*)((*this->x)[0]))->x_dim, this->dim_C);
			
			for(int i = 0; i < ((constant<T>*)((*this->x)[0]))->length; i++) 
			   {
				((constant<T>*)(*this->dx)[0])->x[i] = ((constant<T>*)(this->dy_sum))->x[index_p[i]];
			   }
		}
		else {

			(*this->dx)[0] = ((constant<T>*)((*this->x)[0]))->copy2ones();
		}

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
	virtual void forward_function() {
		//self->x=father->y::fathers y be converted to this->x ,vector<constant<T>*>* x , vector<base_op<T>*> fathers
	   (*(this->x))[0] = ((base_op<T>*)(this->fathers[0]))->y;//only one father
		
		//from this->x computer this->y
		T beta = 0;
		T alpha = 1.0;
		
		int sumsize = 1;
		for (int i = 0; i < 4; i++) {
			sumsize = sumsize *(this->dim_C)[i];
		}

		T* src = (T*)malloc(sumsize * sizeof(T));
		this->y = constant<T>::getObject("reduce_sum_y", ((constant<T>*)(*(this->x))[0])->device, ((constant<T>*)(*(this->x))[0])->x_dim_num,this->dim_C, src);
		free(src);

		int* result = (int *)malloc(sizeof(int) * 1);
		constant<T>::op_math_reduce(CONS_REDUCE_TENSOR_ADD, (*(this->x))[0], this->y, &alpha, &beta, result);
		free(result);

		total_not_finish_ops_num -= 1;
		this->forwardover = 1;

		for (int i = 0; i < this->sons_num; i++)
		{
			((base_op<T>*)(this->sons[i]))->fathers_finshed_size -= 1;//father finished
			if (((base_op<T>*)(this->sons[i]))->fathers_finshed_size == 0 && ((base_op<T>*)(this->sons[i]))->forwardover != 1)
				queue_forward_canbe_used_ops->push(((base_op<T>*)(this->sons[i]))->name_of_op);
			//((threadsafe_queue<string>*) base_op<T>::queue_forward_canbe_used_ops)->push(((base_op<T>*)(this->sons[i]))->name_of_op);
		}

		//cout << "forward::" << this->name_of_op << " y:" << this->y->x[0] << " y:" << this->y->x[1] << endl;
	}
};

#endif // !_SUM_OP_CUH