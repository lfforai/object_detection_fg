#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "npp.h"
#include "nppdefs.h"
#include "nppcore.h"
#include "nppi.h"
#include "npps.h"

#include <string>
#include <windows.h>
#include <thread>
#include  <ctime>
#include <iostream>

#include <unordered_map>
#include "base_op.cuh"
#include "graph_active_threads.cuh"

#include "constant_class.cuh"
#include "weigh_class.cuh"
#include "x_op.cuh"
#include "exp_op.cuh"
#include "sum_op.cuh"
#include "div_op.cuh"
#include "mul_op.cuh"
#include "sin_op.cuh"
#include "cos_op.cuh"
#include "queue.cuh"
#include "reduce_sum_op.cuh"
#include "reduce_avg_op.cuh"
#include "power_x_op.cuh"
#include "a_pow_x_op.cuh"
#include "conv_op.cuh"

using namespace std;

#ifndef _TF_CUH_
#define _TF_CUH_

//string Convert(float Num)
//{
//	ostringstream oss;
//	oss << Num;
//	string str(oss.str());
//	return str;
//}

template<class T>
class tf {
public:
	tf(){
	//init graph

	}

	void graph_init() {
		base_op<T>::global_graph = new graph<T, base_op>;
		base_op<T>::global_w_trainable = new graph<T, variable>;
		base_op<T>::global_dw_trainable = new graph<T, variable>;
		base_op<T>::global_placehold_constant = new graph<T, constant>;
		//base_op<T>::queue_forward_canbe_used_ops = new threadsafe_queue<string>;
	}
	
	graph_active<T>* session(){   
		 return graph_active<T>::getobject(base_op<T>::global_graph);
	}
	
	//constant
	base_op<T>* constantPlaceholder_o(string name, int device, int x_dim_num_o, int* dim)
	{
		return x_op<T>::convert_Placeholder_to_x_op(name, device, x_dim_num_o, dim);
	}

	base_op<T>* constant_o(string name,int device,int x_dim_num_o,int* dim, T* src)
	{  
		return x_op<T>::convert_cons_to_x_op(name,device,x_dim_num_o,dim, src);
	}

	//variable
	base_op<T>* variable_o(bool trianable,string name,int device,int x_dim_num_o,int* dim, T* src)
	{
		return x_op<T>::convert_var_to_x_op(trianable,name,device,x_dim_num_o,dim,src);
	}
   
	//net_op
	base_op<T>* conv(base_op<T>* op,int k_o,int* filtersize)
	{
	    return conv_op<T>::getObejct(op,"conv(" + op->name_of_op + ")",k_o,filtersize);
	}

	//math_op
	base_op<T>* exp(base_op<T>* op)
	{   
		return exp_op<T>::getObejct(op,1.0,"exp("+op->name_of_op+")");
	}

	base_op<T>* sin(base_op<T>* op)
	{
		return sin_op<T>::getObejct(op, 1.0, "sin(" + op->name_of_op + ")");
	}

	base_op<T>* cos(base_op<T>* op)
	{
		return cos_op<T>::getObejct(op, 1.0, "cos(" + op->name_of_op + ")");
	}

	base_op<T>* power_x(base_op<T>* op,T aphla)
	{   
		return power_x_op<T>::getObejct(op, aphla , "power(" + op->name_of_op +"^"+std::to_string(aphla) +")");
	}

	base_op<T>* a_power_x(base_op<T>* op, T aphla)
	{
		return a_power_x_op<T>::getObejct(op, aphla, "a_power_x(" + std::to_string(aphla)+"^"+ op->name_of_op + ")");
	}

	base_op<T>* reduce_sum(base_op<T>* op,int* dim_c)
	{
		return reduce_sum_op<T>::getObejct(op, dim_c, "reduce_sum(" + op->name_of_op + ")");
	}

	base_op<T>* reduce_avg(base_op<T>* op, int* dim_c)
	{
		return reduce_avg_op<T>::getObejct(op, dim_c, "reduce_avg(" + op->name_of_op + ")");
	}

	template<class S> friend base_op<S>* operator*(base_op<S>& op1, base_op<S>* op2);
	
	template<class S> friend base_op<S>* operator+(base_op<S>& op1, base_op<S>* op2);

	template<class S> friend base_op<S>* operator/(base_op<S>& op1, base_op<S>* op2);

	template<class S> friend base_op<S>* operator-(base_op<S>& op1, base_op<S>* op2);
};

template<class S>
base_op<S>* operator*(base_op<S>& op1, base_op<S>* op2)
{
	return mul_op<S>::getObejct(&op1, 1.0, op2, "("+op1.name_of_op+ ")"+ "*" + "("+op2->name_of_op+")");
}

template<class S>
base_op<S>* operator+(base_op<S>& op1, base_op<S>* op2)
{
	return sum_op<S>::getObejct(&op1, 1.0, 1.0, op2,  op1.name_of_op  + "+" + op2->name_of_op );
}

template<class S>
base_op<S>* operator/(base_op<S>& op1, base_op<S>* op2)
{
	return div_op<S>::getObejct(&op1, 1.0, op2, "(" + op1.name_of_op + ")" + "/" + "(" + op2->name_of_op + ")");
}

template<class S>
base_op<S>* operator-(base_op<S>& op1, base_op<S>* op2)
{
	return sum_op<S>::getObejct(&op1, 1.0, -1.0, op2, op1.name_of_op  + "-"  + op2->name_of_op);
}
#endif // _TF_CUH_
