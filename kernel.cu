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

using namespace std;

DWORD WINAPI Fun(LPVOID lpParamter)
{
	for (int i = 0; i < 10; i++)
		cout << "A Thread Fun Display!" << endl;
	return 0L;
}

template<class T>
struct x_y_w_dx_dy_dw{
	T* x;//forward input
	T* y;//forward output
	T* w;//w
	T* dx;//backward  output
	T* dy;//backward input
	T* dw;
};

template<class T1,class T2>
class father{
public:
	T1 name;
	T2 value;
    father(T1 a,T2 b):name(a),value(b){
	};
	//~father(){};
};

//template<class T1,class T2>
//unordered_map<string, father<T1,T2>*> un_map;

//难度在数据流的记录上，而不是op的记录上
int main()
{   
	//father<string,float>* ok_o=new father<string,float>("luofeng",2.0);
	//unordered_map<string, father<string, float>*> un_map;
	//un_map.insert(std::pair<string,father<string,float>*>{"first", ok_o});
	//father<string, float>* ok_o2 = new father<string, float>("lf", 3.0);
	//un_map.insert(std::pair<string, father<string, float>*>{"second", ok_o2});
	//un_map.erase("second");
	//cout<<"---------------------"<<endl;
	//for (const auto& person : un_map)
	//{
	//	std::cout << person.first << " is " << person.second << std::endl;
	//}
	//
	//HANDLE hThread = CreateThread(NULL, 0, Fun, NULL, 0, NULL);
	//CloseHandle(hThread);
	//for (int i = 0; i < 10; i++)
	//	cout << "Main Thread Display!" << endl;

	constant<int>* op_c = (constant<int> *)malloc(sizeof(constant<int>));
	int *src = (int *)malloc(3 * 6 * 4 * 2 * sizeof(int));
	int dim[4] = { 3,6,4,2 };
	op_c->init(0,4,dim,src);
	graph<int, base_op>* graph_g=new graph<int,base_op>;
	w_dw_now<int>* w_dw_o;
	base_op<int>* baseop=new base_op<int>(op_c,graph_g,"add");
	base_op<int>* baseop1= new base_op<int>(op_c, graph_g,"sub");
	base_op<int>* baseop2 = new base_op<int>(op_c, graph_g,"mul");

	base_op<int>* addop = new base_op<int>(baseop,baseop1,op_c,graph_g,"addop");
	base_op<int>* active = new base_op<int>(baseop, op_c,graph_g,"eule");
	
	op_c->clear();
	for (vector<base_op<int>*>::const_iterator iter = addop->fathers.cbegin(); iter != addop->fathers.cend(); iter++)
	{
		cout << (*iter)->name_of_op << endl;
	}

	//
	//a.insert_v(baseop->name_of_op,baseop);
	//a.insert_v(baseop1->name_of_op, baseop1);
	//a.insert_v(baseop2->name_of_op, baseop2);
	//a.insert_v(addop->name_of_op, addop);
	//a.insert_v(active->name_of_op, active);
	graph_g->allvalue();
    return 0;
}


