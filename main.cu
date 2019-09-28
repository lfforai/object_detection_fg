#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
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
#include "reduce_sum_op.cuh"
#include "tf.cuh"


using namespace std;
using namespace cv;

void addr(float& b) {
	float* c = &b;
	cout<<c<<endl;
}

int main()
{
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	//int dimfold[4] = {-1, 1, 1, 1};
	//int dimfold2[4] = { 1, 1, 1, 1 };

	//int dim[4] = { 1,1,1,1 };
	//int dim1[4] = { 1,1,1,3};
	//float src[3] = {5.0,6.0,7.0};
	//float a[1] = { 2.0 };
	//float one_v[1] = { 1.0 };

	//tf<float> tf; 
	//tf.graph_init();

	//base_op<float>* two_con = tf.constantPlaceholder_o("2",1,4,dimfold);
	//base_op<float>* one_con=tf.constant_o("1", 1, 4, dim, one_v);
	//base_op<float>* X    =tf.variable_o(true, "X", 1, 4, dim1, src);
	//base_op<float>* exp = tf.exp(X);

	//base_op<float>* y1=(*((*one_con) - X))*((*two_con)*exp);//2*exp(x)*(1-x)
	//base_op<float>* y2 =*(*X + two_con) + *(*one_con + X)*exp;
	//base_op<float>* last = *(tf.exp(*(*y1 + tf.sin(X)) / (*y2*tf.cos(X))))+tf.sin(X);
	//int dim_c[4] = { 1, 1, 1, 1 };
	//base_op<float>* reduce = tf.cos(tf.reduce_avg(tf.a_power_x(last,2.0),dim_c));
	//
	////set sess
	//graph_active<float>* sess=tf.session();
	////init placeholder
	//vector<constant<float>*>*  v = new vector<constant<float>*>;
	//constant<float>* aa = constant<float>::getObject("2", 1, 4, dim,a);
	//v->push_back(aa);
	//sess->Placeholder_assgin(*v);
	//aa->clear();

	////run sess
	//sess->ward_start(1,0);
	//sess->ward_start(0,1);
	//endTime = clock();
	//
	////output 
	//cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	//cout <<"forward::"<<(reduce->y)->x[0] << endl;
	//cout << "forward::" << (last->y)->x[1] << endl;
	//cout << "forward::" << (last->y)->x[2] << endl;
	//cout<<"----------------------------------------- "<<endl;
	//vector<variable<float>*>* list_dw=base_op<float>::global_dw_trainable->getallvalue();
	//for(const auto& e : *list_dw)
	//  {   
	//	for (int i = 0; i < ((variable<float>*) e)->length; i++)
	//	{
	//		cout <<e->var_name<<":" <<((variable<float>*)e)->x[i] << endl;;
	//	}
	//  }
	
	tensor_reduce_test();
	endTime = clock();
	//output 
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}


