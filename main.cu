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
#include "sum_op.cuh"

#include "constant_class.cuh"
#include "weigh_class.cuh"
using namespace std;


int main()
{
	//int *src = (int *)malloc(3 * 6 * 4 * 2 * sizeof(int));
	//int dim[4] = { 3,6,4,2 };
	//constant<int>*  con1=constant<int>::getObject("con1",0, 4, dim, src);
	//constant<int>*  con2 = constant<int>::getObject("con2", 0, 4, dim, src);
	//variable<int>*  w1 = variable<int>::getObject(true,"w1", 0, 4, dim, src);
	//variable<int>*  w2 = variable<int>::getObject(true,"w2", 0, 4, dim, src);
	//vector<constant<int>*>* cons_vector = new vector<constant<int>*>;
	//vector<variable<int>*>* w_vector = new vector<variable<int>*>;
	//cons_vector->push_back(con1);
	//cons_vector->push_back(con2);
	//w_vector->push_back(w1);
	//w_vector->push_back(w2);

 //   base_op<int>::global_graph = new graph<int, base_op>;
	//base_op<int>::global_w_trainable = new graph<int, variable>;
	////base_op<int>* addnew =sum_op<int>::getObejct(cons_vector, "addnew");
	////graph<int, base_op>* global_graph = new graph<int, base_op>;
	//base_op<int>* add = base_op<int>::getObejct(cons_vector,"add");
 //   base_op<int>* sub = base_op<int>::getObejct(cons_vector, "sub");
	//base_op<int>* mul = base_op<int>::getObejct(cons_vector, "mul");
	//base_op<int>* log = base_op<int>::getObejct(add,sub,cons_vector,w_vector,"log");
	//base_op<int>* exp = base_op<int>::getObejct(mul, cons_vector, w_vector, "exp");
	//base_op<int>* log_sub = base_op<int>::getObejct(log,sub, cons_vector, w_vector, "log_sub");
	//base_op<int>* log_exp = base_op<int>::getObejct(log, exp, cons_vector, w_vector, "log_exp");
	//base_op<int>* soft_max = base_op<int>::getObejct(log_sub, log_exp, cons_vector, w_vector,  "soft_max");
	//base_op<int>* cross_entry = base_op<int>::getObejct(soft_max, log_exp, cons_vector, "cross_entry");
	//
	////base_op<int>::global_graph->allvalue();
	////active graph
	//graph_active<int>* gc = graph_active<int>::getobject(base_op<int>::global_graph);
	//cout<<"forward start"<<endl;
	//gc->forward_start(1);
	//cout <<"backward start"<< endl;
 //   gc->backward_start(1);

	//varible out
	//base_op<int>::global_w_trainable->allvalue();
	//test();
    
	int dim[4] = {1,1,2,4};
	int dimC[4] = {1,1,3,1};
	float src[8] = { 4.0,1.0,1.0,3.0,1.0,2.0,2.0,3.0 };
	float srcC[4] ={ 3.0,2.0,1.0};
	constant<float>*  A = constant<float>::getObject("A", 1, 4, dim, src);
	constant<float>*  C = constant<float>::getObject("C", 1, 4, dim, src);
	variable<float>*  B = variable<float>::getObject(true,"B", 1, 4, dimC, srcC);
	float bate = 2.0;
	constant<float>* D= A->scala_mul(bate);
	cout<<D->con_name<<endl;
	constant<float>* D1=D->function_tensor(CONS_LOG,1.0,1);
	cout<<D1->con_name<<endl;

	//float ap0=1.0;
	//float ap1=0.0;
	//float beta=0.0;
	//CONSTANT_OP_ADD = 0,
	//	CONSTANT_OP_MUL = 1,
	//	CONSTANT_OP_TENSOR_MIN = 2,
	//	CONSTANT_OP_TENSOR_MAX = 3,
	//	CONSTANT_OP_TENSOR_SQRT = 4,
	//	CONSTANT_OP_TENSOR_NOT = 5,
	//constant<float>::op_math(CONSTANT_OP_ADD,A, B, C,&ap0,&ap0,&beta);

	//cudnnReduceTensorOp_t
	//  CONS_REDUCE_TENSOR_ADD = 0,
	//	CONS_REDUCE_TENSOR_MUL = 1,
	//	CONS_REDUCE_TENSOR_MIN = 2,
	//	CONS_REDUCE_TENSOR_MAX = 3,
	//	CONS_REDUCE_TENSOR_AMAX = 4,
	//	CONS_REDUCE_TENSOR_AVG = 5,
	//	CONS_REDUCE_TENSOR_NORM1 = 6,
	//	CONS_REDUCE_TENSOR_NORM2 = 7,
	//	CONS_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
	//int* result = (int*)malloc(2*sizeof(int));
	//constant<float>::op_math_reduce(CONS_REDUCE_TENSOR_NORM2, A, C, &ap0, &beta,result);
	//cout<<result[0]<<endl;
	//cout<<result[1]<<endl;
	//constant<float>::add(C, B, C, &ap0, &ap0, &beta);
	//int a;
	return 0;
}


