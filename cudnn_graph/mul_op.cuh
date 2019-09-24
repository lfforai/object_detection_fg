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
#ifndef _MUL_OP_CUH
#define _MUL_OP_CUH
template<class T>
class mul_op:public base_op<T>
{
public:


	static mul_op<T>* getObejct(base_op<T>* op1,T aphla_o, base_op<T>* op2, string name_o, char* Tensor_des = "")
	{   
		//assume size must be same
		mul_op<T>* result = new mul_op<T>;
		result->alpha = aphla_o;
		result->name_of_op = name_o;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;


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

		mul_op<T>::global_graph->insert_v_repeat(result->name_of_op, result);
		return result;
	}


	//reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() {
		//cout << "backward mul start::" << this->name_of_op << endl;
		//transport dy to dx

		bool eq=((constant<T>*)(*this->x)[0])->x_stride[0] * ((constant<T>*)(*this->x)[0])->x_dim[0] ==
			    ((constant<T>*)(*this->x)[1])->x_stride[0] * ((constant<T>*)(*this->x)[1])->x_dim[0] ?true : false;
		
		if (eq) 
		{   //op1.size =op2.size
			if (this->sons_num > 0) {
				//has sons
				for (int i = 0; i < this->sons_num; i++)
				{   //find the index of sons->father
					vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
					int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

					//self->dy=son->dx
					this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
				}

				int i = 0;
				T aphla_mul = 1;
				T beta_mul = 0;
				this->sum_dy();

				int len = (vector<constant<T>*>(*this->dx)).size();
				for (int i = 0; i < len; i++)
				{    //iter is a father->xd;
					if (i == 0)
					{
						constant<T>* temp = ((*this->x)[1])->copy();
						constant<T>::op_math(CONSTANT_OP_MUL, (*this->x)[1], this->dy_sum, temp, &this->alpha, &aphla_mul, &beta_mul);
						(*this->dx)[i] = temp;
					}

					if (i == 1)
					{
						constant<T>* temp = ((*this->x)[0])->copy();;
						constant<T>::op_math(CONSTANT_OP_MUL, (*this->x)[0], this->dy_sum, temp, &this->alpha, &aphla_mul, &beta_mul);
						(*this->dx)[i] = temp;

					}
				}
			}
			else {
				// no sons 
				int i = 0;
				int len = (vector<constant<T>*>(*this->dx)).size();
				for (int i = 0; i < len; i++)
				{    //iter is a father->xd;
					if (i == 0)
					{
						(*this->dx)[i] = ((*this->x)[1])->copy();
					}
					if (i == 1)
					{
						(*this->dx)[i] = ((*this->x)[0])->copy();
					}
				}
			}
		}
		else//op1.size!=op2.size!
	    {
	     int lenop1 = ((constant<T>*)(*this->x)[0])->x_stride[0] * ((constant<T>*)(*this->x)[0])->x_dim[0];
		 int lenop2 = ((constant<T>*)(*this->x)[1])->x_stride[0] * ((constant<T>*)(*this->x)[1])->x_dim[0];
		 int maxindex = lenop1 > lenop2 ? 0 : 1;

		 constant<T>* tempbiger = ((constant<T>*)(*this->x)[maxindex])->copy();
		 constant<T>* tempsmall = ((constant<T>*)(*this->x)[1 - maxindex])->copy2zeros();

		 //----------------------------------------------------------------------	 
		 if(this->sons_num > 0) 
			{    //has sons
			     //cout << "no eq mul hase sons" << endl;
				 for (int i = 0; i < this->sons_num; i++)
				  {   //find the index of sons->father
					  vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
					  int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);

					  //self->dy=son->dx
					  this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
				  }

				 this->sum_dy();

				 if (maxindex == 0)
				 {
					 int n = (int)(lenop1 / lenop2);
					 for (int i = 0; i < n; i++)
					 {
						 checkCudaErrors(cudaMemcpy(tempbiger->x + lenop2 * i, ((*this->x)[1])->x, sizeof(T)*lenop2, cudaMemcpyDefault));
						 //memcpy(tempbiger->x + lenop2 * i, ((*this->x)[1])->x, sizeof(T)*lenop2);
					 }

					 for (int i = 0; i < lenop1; i++) {
					     *(((constant<T>*)tempbiger)->x + i) = *(((constant<T>*)tempbiger)->x + i) *(((this->dy_sum)->x)[i]);
					 }


					 for (int i = 0; i < lenop2; i++)
					 {
						 for (int j = 0; j < n; j++)
						 {

							 *(((constant<T>*)tempsmall)->x + i) = *(((constant<T>*)tempsmall)->x + i) + (((this->dy_sum)->x)[lenop2*j + i]) * (((*this->x)[0])->x)[j*lenop2 + i];
						 }
					 }
				 }
				 else
				 {
					 int n = (int)(lenop2 / lenop1);
					 for (int i = 0; i < n; i++)
					 {
						 checkCudaErrors(cudaMemcpy(tempbiger->x + lenop1 * i, ((*this->x)[0])->x, sizeof(T)*lenop1, cudaMemcpyDefault));
						 //memcpy(tempbiger->x + lenop1 * i, ((*this->x)[0])->x, sizeof(T)*lenop1);
					 }

					 for (int i = 0; i < lenop2; i++) {

						 *(((constant<T>*)tempbiger)->x + i) = *(((constant<T>*)tempbiger)->x + i) *(((this->dy_sum)->x)[i]);
					 }

					 for (int i = 0; i < lenop1; i++)
					 {
						 for (int j = 0; j < n; j++) {
							
                            *(((constant<T>*)tempsmall)->x+i) = *(((constant<T>*)tempsmall)->x + i) + (((this->dy_sum)->x)[lenop1*j + i]) * ((((*this->x)[1])->x)[lenop1*j + i]);
						 }
					 }

				 }
				 (*this->dx)[maxindex] = tempbiger;
				 (*this->dx)[1 - maxindex] = tempsmall;
			}
			else 
			{ //no  sons
			 //cout<<"no eq mul no sons"<<endl;
			 if (maxindex == 0)
			 {    
				 int n = (int)(lenop1 / lenop2);
				 for (int i = 0; i < n; i++)
				 {
					 checkCudaErrors(cudaMemcpy(tempbiger->x + lenop2 * i, ((*this->x)[1])->x, sizeof(T)*lenop2, cudaMemcpyDefault));
					 //memcpy(tempbiger->x + lenop2 * i, ((*this->x)[1])->x, sizeof(T)*lenop2);
				 }

				 for (int i = 0; i < lenop2; i++)
				 {
					 for (int j = 0; j < n; j++)
					 {

						 (tempsmall->x)[i] = (tempsmall->x)[i] + (((*this->x)[0])->x)[j*lenop2 + i];
					 }
				 }
			 }
			 else
			 {
				 int n = (int)(lenop2 / lenop1);
				 for (int i = 0; i < n; i++)
				 {
					 checkCudaErrors(cudaMemcpy(tempbiger->x + lenop1 * i, ((*this->x)[0])->x, sizeof(T)*lenop1, cudaMemcpyDefault));
					 //memcpy(tempbiger->x + lenop1 * i, ((*this->x)[0])->x, sizeof(T)*lenop1);
				 }

				 for (int i = 0; i < lenop1; i++)
				 {
					 for (int j = 0; j < n; j++) {

						 (tempsmall->x)[i] = (tempsmall->x)[i] + (((*this->x)[1])->x)[lenop1*j + i];
					 }
				 }

			 }
			  (*this->dx)[maxindex] =     tempbiger;
			  (*this->dx)[1 - maxindex] = tempsmall;
			}
		}
		//-------------------------------------------------------------------------

		total_not_finish_ops_num -= 1;
		this->backwardover = 1;

		for (int i = 0; i < this->fathers_num; i++)
		{
			((base_op<T>*)(this->fathers[i]))->sons_finshed_size -= 1;//father finished
			if (((base_op<T>*)(this->fathers[i]))->sons_finshed_size == 0 && ((base_op<T>*)(this->fathers[i]))->backwardover != 1)
				queue_forward_canbe_used_ops->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
				//((threadsafe_queue<string>*) base_op<T>::queue_forward_canbe_used_ops)->push(((base_op<T>*)(this->fathers[i]))->name_of_op);
		}
	
		//cout << "backward mul over::" << this->name_of_op << endl;
	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	virtual void forward_function() {
		//from this->x computer this->y
		for (int i = 0; i < this->fathers_num; i++)
		{
			//self->x=father->y::fathers y be converted to this->x ,vector<constant<T>*>* x , vector<base_op<T>*> fathers
			(*(this->x))[i] = ((base_op<T>*)(this->fathers[i]))->y;
		}
		
		T aphla_mul = 1;
		T beta_mul= 0;

		int maxindex = ((constant<T>*)(*this->x)[0])->x_stride[0]*((constant<T>*)(*this->x)[0])->x_dim[0] >
			           ((constant<T>*)(*this->x)[1])->x_stride[0] * ((constant<T>*)(*this->x)[1])->x_dim[0] ? 0 : 1;
		constant<T>* temp = ((*this->x)[maxindex])->copy();
		this->y = temp;

		if (maxindex == 0)
			constant<T>::op_math(CONSTANT_OP_MUL, this->y, (*this->x)[1], this->y, &aphla_mul, &aphla_mul, &beta_mul);
		else
			constant<T>::op_math(CONSTANT_OP_MUL, this->y, (*this->x)[0], this->y, &aphla_mul, &aphla_mul, &beta_mul);

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

