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
using namespace std;
int main()
{
	constant<int>* op_c = (constant<int> *)malloc(sizeof(constant<int>));
	int *src = (int *)malloc(3 * 6 * 4 * 2 * sizeof(int));
	int dim[4] = { 3,6,4,2 };
	op_c->init(0, 4, dim, src);

	graph<int, base_op>* graph_g = new graph<int, base_op>;
	base_op<int>* add = new base_op<int>(op_c, graph_g, "add");
	base_op<int>* sub = new base_op<int>(op_c, graph_g, "sub");
	base_op<int>* mul = new base_op<int>(op_c, graph_g, "mul");
	base_op<int>* log = new base_op<int>(add, sub, op_c, graph_g, "add_sub");
	base_op<int>* exp = new base_op<int>(mul, op_c, graph_g, "op_mul");
	base_op<int>* simge = new base_op<int>(log, sub,op_c, graph_g, "log_sub");
	base_op<int>* cos = new base_op<int>(exp, log, op_c, graph_g, "log_exp");
	base_op<int>* exten = new base_op<int>(op_c, graph_g, "exten");
	//graph_g->allvalue();

	graph_active<int>* ga = graph_active<int>::getobject(graph_g);
	cout<<"forward start"<<endl;
	ga->forward_start(1);
	cout <<"backward start"<< endl;
	ga->backward_start(1);
	//test();
	return 0;
}

