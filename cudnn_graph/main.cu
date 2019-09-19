#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "npp.h"
#include "nppdefs.h"
#include "nppcore.h"
#include "nppi.h"
#include "npps.h"
#include <iostream>
#include <unordered_map>
#include <string>
#include <windows.h>
#include "base_op.cuh"
#include <thread>
#include "graph_active_threads.cuh"
#include "test_tool.h"


#include "constant_class.cuh"
#include "weigh_class.cuh"
#include "x_op.cuh"
#include "exp_op.cuh"
#include "sum_op.cuh"
#include "div_op.cuh"
#include "mul_op.cuh"


using namespace std;
int main()
{
	int dim[4] = { 1,1,1,1 };
	float src[1] = { 5.0 };
	float a[1] = { 2.0 };
	float one_v[1] = { 1.0 };
	
	//init graph
	base_op<float>::global_graph = new graph<float, base_op>;
    base_op<float>::global_w_trainable = new graph<float, variable>;
	base_op<float>::global_placehold_constant = new graph<float, constant>;

    //graph_create
	base_op<float>*  two_con= x_op<float>::convert_cons_to_x_op("a", 1, 4, dim, a);
	base_op<float>*  X_var = x_op<float>::convert_var_to_x_op(true,"X", 1,4, dim, src);
	base_op<float>*  one_con = x_op<float>::convert_cons_to_x_op("1", 1, 4, dim, one_v);
    
	//put into ops
	base_op<float>* x_1 = sum_op<float>::getObejct(one_con, 1.0, -1.0, X_var, "1-x|");
	base_op<float>* exp_2 = exp_op<float>::getObejct(X_var,2.0,"2*e^x|");
	base_op<float>* x_1_2_exp = mul_op<float>::getObejct(exp_2,1.0,x_1,"2*(1-x)*e^x|");

	base_op<float>* x_add_1 = sum_op<float>::getObejct(one_con, 1.0, 1.0, X_var, "1+x|");
	base_op<float>* exp_1 = exp_op<float>::getObejct(X_var, 1.0, "1*e^x|");
	base_op<float>* x_add_1_1_exp = mul_op<float>::getObejct(exp_1, 1.0, x_add_1,"(1+x)*e^x");
	base_op<float>* x_2= sum_op<float>::getObejct(two_con, 1.0, 1.0, X_var, "2+x|");

	base_op<float>* x_add_1_1_exp_x_2= sum_op<float>::getObejct(x_add_1_1_exp, 1.0, 1.0, x_2, "2+x+(1+x)*e^x|");

	base_op<float>* last = div_op<float>::getObejct(x_1_2_exp, 1.0, x_add_1_1_exp_x_2,"last");

	//active 
	graph_active<float>* graph_ac=graph_active<float>::getobject(base_op<float>::global_graph);
	graph_ac->forward_start(0);
	//if (two_con->sons.empty());
	//cout << "empty" << endl;
	graph_ac->backward_start(0);
	return 0;
}


