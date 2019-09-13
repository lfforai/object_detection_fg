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
	int *src = (int *)malloc(3 * 6 * 4 * 2 * sizeof(int));
	int dim[4] = { 3,6,4,2 };
	constant<int>*  con1=constant<int>::getObject("con1",0, 4, dim, src);
	constant<int>*  con2 = constant<int>::getObject("con2", 0, 4, dim, src);
	variable<int>*  w1 = variable<int>::getObject("w1", 0, 4, dim, src);
	variable<int>*  w2 = variable<int>::getObject("w2", 0, 4, dim, src);
	vector<constant<int>*>* cons_vector = new vector<constant<int>*>;
	vector<variable<int>*>* w_vector = new vector<variable<int>*>;
	cons_vector->push_back(con1);
	cons_vector->push_back(con2);
	w_vector->push_back(w1);
	w_vector->push_back(w2);

	//base_op<int>::init_global_graph_ac_varible<int>();
    base_op<int>::global_graph = new graph<int, base_op>;
	base_op<int>::global_w_trainable = new graph<int, variable>;
	base_op<int>* addnew =sum_op<int>::getObejct(cons_vector, "addnew");
	//graph<int, base_op>* global_graph = new graph<int, base_op>;
	base_op<int>* add = base_op<int>::getObejct(cons_vector,"add");
    base_op<int>* sub = base_op<int>::getObejct(cons_vector, "sub");
	base_op<int>* mul = base_op<int>::getObejct(cons_vector, "mul");
	base_op<int>* log = base_op<int>::getObejct(add,sub,cons_vector,w_vector,"log");
	base_op<int>* exp = base_op<int>::getObejct(mul, cons_vector, w_vector, "exp");
	base_op<int>* log_sub = base_op<int>::getObejct(log,sub, cons_vector, w_vector, "log_sub");
	base_op<int>* log_exp = base_op<int>::getObejct(log, exp, cons_vector, w_vector, "log_exp");
	base_op<int>* soft_max = base_op<int>::getObejct(log_sub, log_exp, cons_vector, w_vector,  "soft_max");
	base_op<int>* cross_entry = base_op<int>::getObejct(soft_max, log_exp, cons_vector, "cross_entry");
	
	//base_op<int>::global_graph->allvalue();
	//active graph
	graph_active<int>* gc = graph_active<int>::getobject(base_op<int>::global_graph);
	cout<<"forward start"<<endl;
	gc->forward_start(1);
	cout <<"backward start"<< endl;
    gc->backward_start(1);

	//varible out
	base_op<int>::global_w_trainable->allvalue();
	//test();
	addnew->backward_function();
	return 0;
}


