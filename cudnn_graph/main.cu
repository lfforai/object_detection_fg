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
#include  <ctime>

#include "constant_class.cuh"
#include "weigh_class.cuh"
#include "x_op.cuh"
#include "exp_op.cuh"
#include "sum_op.cuh"
#include "div_op.cuh"
#include "mul_op.cuh"
#include "sin_op.cuh"
#include "cos_op.cuh"
#include "tf.cuh"

using namespace std;

void addr(float& b) {
	float* c = &b;
	cout<<c<<endl;
}

int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	int dim[4] = { 1,1,1,1 };
	int dim1[4] = { 1,1,1,3};
	float src[3] = {5.0,6.0,7.0};
	float a[1] = { 2.0 };
	float one_v[1] = { 1.0 };

	tf<float> tf; 
	tf.graph_init();
	
	/*base_op<float>* p = tf.constantPlaceholder_o("px",1,4,dim);*/
	base_op<float>* two_con=tf.constant_o("2", 1, 4, dim, a);
	base_op<float>* one_con=tf.constant_o("1", 1, 4, dim, one_v);
	base_op<float>* X    =tf.variable_o(true, "X", 1, 4, dim1, src);
	base_op<float>* exp = tf.exp(X);

	base_op<float>* y1=(*((*one_con) - X))*((*two_con)*exp);//2*exp(x)*(1-x)
	base_op<float>* y2 =*(*X + two_con) + *(*one_con + X)*exp;
	base_op<float>* last = *(tf.exp(*(*y1 + tf.sin(X)) / (*y2*tf.cos(X))))+tf.sin(X);
	
	graph_active<float>* graph_ac=graph_active<float>::getobject(base_op<float>::global_graph);
	graph_ac->ward_start(0,0);
	cout << "---------------------------------------" << endl;
	graph_ac->ward_start(0,1);
	endTime = clock();
	cout <<"--------------------------------------- "<<endl;
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout <<"forward::"<<(last->y)->x[0] << endl;
	cout << "forward::" << (last->y)->x[1] << endl;
	cout << "forward::" << (last->y)->x[2] << endl;
	cout<<"----------------------------------------- "<<endl;
	cout<<"backward::"<< ((variable<float>*)(X->dw))->x[0]<<endl;
	cout << "backward::" << ((variable<float>*)(X->dw))->x[1] << endl;
	cout << "backward::" << ((variable<float>*)(X->dw))->x[2] << endl;
	return 0;
}


