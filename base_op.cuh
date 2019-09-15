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

#include <windows.h>
#include <wincrypt.h>
#include "stdarg.h"
#include "gpumathtool.cuh"

//#include "fp16_dev.h"

using namespace std;

#ifndef _BASE_OP_CUH
#define _BASE_OP_CUH

template<class T>
class base_op{
 private:
 public:
	 int randEx() //real random
	 {
		 LARGE_INTEGER seed;
		 QueryPerformanceFrequency(&seed);
		 QueryPerformanceCounter(&seed);
		 srand(seed.QuadPart);

		 return rand();
	 }

	 inline void name_of_op_is_repeat() {
		 if (this->global_graph->un_map.empty())
		 {
			 bool op_name_is_repeat = this->global_graph->if_find(this->name_of_op);
			 assert(op_name_is_repeat == false);
		 }
	 };



	 //-1:father not ready,1:father not ready
	 inline int if_fathers_ready_forward() {
		 int result = 1;
		 if (this->fathers.empty())
		 {
			 return result;
		 }
		 else {
			 for (typename vector<base_op<T>*>::const_iterator iter = this->fathers.cbegin(); iter != this->fathers.cend(); iter++)
			 {
				 if ((*iter)->forward_over != 1)
				 {
					 result = -1;
					 break;
				 }
			 }
		 }
		 return result;
	 }

	 //-1:sons not ready,1:sons not ready
	 inline int if_sons_ready_backward() {
		 int result = 1;
		 if (this->sons.empty())
		 {
			 return result;
		 }
		 else {
			 //must have e tpyename
			 for (typename vector<base_op<T>*>::const_iterator iter = this->sons.cbegin(); iter != this->sons.cend(); iter++)
			 {
				 if ((*iter)->backward_over != 1)
				 {
					 result = -1;
					 break;
				 }
			 }
		 }
		 return result;
	 }

	 //reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() {
		 //input::vector<constant<T>*>* dy|vector<variable<T>*>* w;
		 //output::vector<variable<T>*>* dw|vector<constant<T>*>* dx; 
		 float secs = (float)(randEx() % 5);      //定义浮点型变量secs
		 clock_t delay;  //定义clock_t类型的变量，表示延时时间
		 delay = secs * CLOCKS_PER_SEC;  //delay赋值为secs 乘以 CLOCKS_PER_SEC值，将输入的秒数转化系统的时间
		 clock_t start = clock();    //定义clock_t类型变量start，并赋值为当前系统的时间
	     //cout << this->name_of_op << " is backwarding now......" << endl;
		 while (clock() - start < delay);  // 如果当前时间减去上一刻的系统时间小于延时的系统时间，则执行循环等待，否则跳出循                                                                          环
		 backward_over = 1;
		 cout << this->name_of_op << endl;

	 }

	 //reload the forward_function,make sure last of the function must be forward_over = 1
	 virtual void forward_function() {
		 //input:: vector<constant<T>*>* x|vector<variable<T>*>* w|vector<constant<T>*>* cons;
		 //output::constant<T>* y         
		 float secs = (float)(randEx() % 5);;      //定义浮点型变量secs
		 clock_t delay;  //定义clock_t类型的变量，表示延时时间
		 delay = secs * CLOCKS_PER_SEC;   //delay赋值为secs 乘以 CLOCKS_PER_SEC值，将输入的秒数转化系统的时间
		 clock_t start = clock();    //定义clock_t类型变量start，并赋值为当前系统的时间
		 //cout << this->name_of_op << " is forwarding now......" << endl;
		 while (clock() - start < delay);  // 如果当前时间减去上一刻的系统时间小于延时的系统时间，则执行循环等待，否则跳出循                                                                          环
		 forward_over = 1;
		 cout << this->name_of_op << endl;
	 }

	 bool input_op_eq_const=false;

	 mutex mtx;//互斥量

	 //--------------------------op-value--------------------------------------
	 string name_of_op; //mast be unique;

	//N input ops of fathers has  N  x[0]....x[N-1] input,xd[0]....xd[N-1] output
	//x,dx-
	vector<constant<T>*>* x;  //from faher, forward
	vector<constant<T>*>* dx; //to father, backward
	int xdx_num=0;

	//N output ops of sons has  only 1 y input ,only 1 dy output
	//y,dy,
	constant<T>* y;           //to son ,forward
	constant<T>* dy_sum;
	vector<constant<T>*>* dy; //from son ,backward
	int ydy_num = 0;

	void sum_dy()
	{//sum dy 
		constant<T>* dy_sum=new constant<T>;
		int n = this->dy->size();
		dy_sum = (*dy)[0];
		if(n>1)
		 {  int length = ((constant<T>*)(*dy)[0])->x_stride[0] * ((constant<T>*)(*dy)[0])->x_dim[0];
			if(((constant<T>*)(*dy)[0])->device == 1) //data on gpu
			  {
				for(int i = 1; i < n; i++)
				   {
					add_vector_gpu(length, dy_sum->x,((constant<T>*)(*dy)[i])->x);
				   }
			   }
			else//data on cpu
			  {
				for(int i = 1; i < n; i++)
				  {
					for(int j = 0; j < length; j++)
					  {
						dy_sum->x[j] = dy_sum->x[j] + ((constant<T>*)(*dy)[i])->x[j];
					  }
				 }
			  }
		 }
	}

	//w
	vector<variable<T>*>* w;
	vector<variable<T>*>* dw;
	int w_num=0; 

	//constant<T>  constant :not must be needed
	vector<constant<T>*>* cons;
	int cons_num=0;

	//dstValue =alpha[0] * resultValue + beta[0] * priordstValue
	T alpha;//all most 1.0
	T beta;//all most 0.0

	static graph<T, base_op>* global_graph;//all ops belong to the global_graph
	static graph<T, variable>* global_w_trainable;
	static graph<T, constant>* global_placehold_constant;

    // if neededBackwark_dx dw==true ,computer dx,dw  
	bool neededBackwark_dx=true;
	bool neededBackwark_dw=true;

	std::vector<base_op<T>*> fathers;//up ops
	std::vector<string> fathers_name;

	std::vector<base_op<T>*> sons;//down ops
	std::vector<string> sons_name;
	
	int fathers_num = 0; //number of the input ops=N
	int sons_num = 0; //number of the input ops=N

	//thread safe
    std::atomic<int> forward_over=0;// current op finish forward 
	std::atomic<int> backward_over=0;//current op finish backward 
	std::atomic<bool> is_forwarding = false;//if now thread is forwording now
	std::atomic<bool> is_backwarding = false;//if now thread is backwording now
    
	//--------------------------op-function------------------------------------
	//make sure is_forwarding is charged for every threads
	// return 1:can run, 0:is_forwarding or be finished, -1:wait for son ready
	int if_forward_start_run(){
		int result = 0; 
		if(forward_over == 1 || is_forwarding == true)
           return 0;//be ready to remove the op 
		//one thread  pay attention to  the op, other thread may remove the op from its vector
		if(mtx.try_lock()) 
		  {	result = if_fathers_ready_forward();
			if (result == 1)
				is_forwarding = true;
			mtx.unlock();
		  }
		return result;
	}
	
	//make sure is_backwarding is charged for every threads
	// return 1:can run, 0:is_backwarding or be finished, -1:wait for son ready
	int if_backward_start_run()
	{   
		int result = 0;
		if (backward_over == 1 || is_backwarding == true)
			return result;//be ready to remove the op 
		//one thread  pay attention to  the op, other thread may remove the op from its vector
		if(mtx.try_lock())
		{	
			result = if_sons_ready_backward();
			if (result == 1)
				is_backwarding = true;
			mtx.unlock();
		}
		return result;
	}


	void initvector(){
		for (int i = 0; i < this->xdx_num; i++)
		{
			this->x->push_back(0);
			this->dx->push_back(0);
		}

		for (int i = 0; i < this->ydy_num; i++)
		{
			this->dy->push_back(0);
		}

		for (int i = 0; i < this->w_num; i++)
		{
			this->dw->push_back(0);
		}
	}

	//init all w,dw,x,xd,y,dy
	void initparameter(){
		//cout<<this->name_of_op<<" init"<<endl;
		//self------------------------------excharge-------------------------fahter
		//printf("father num:%d \n", this->xdx_num);
		for (int i = 0; i < this->fathers_num; i++)
		{
			vector<string>::iterator ite1 = find(((base_op<T>*)(this->fathers[i]))->sons_name.begin(), ((base_op<T>*)(this->fathers[i]))->sons_name.end(), this->name_of_op);
			int index = (int)std::distance(std::begin(((base_op<T>*)(this->fathers[i]))->sons_name), ite1);
			//self->x=father->y::fathers y be converted to this->x ,vector<constant<T>*>* x , vector<base_op<T>*> fathers
			(*(this->x))[index]=((base_op<T>*)(this->fathers[i]))->y;
			//father->dy=self->dx ::this->dx be converted to fathrer->dy
			((base_op<T>*)(this->fathers[i]))->dy->push_back((*(this->dx))[i]);
		}

		//printf("sons num:%d \n", this->sons_num);
		//self------------------------------excharge-------------------------son
		for (int i = 0; i < this->sons_num; i++)
		{   //find the index of sons->father
			vector<string>::iterator ite1 = find(((base_op<T>*)(this->sons[i]))->fathers_name.begin(), ((base_op<T>*)(this->sons[i]))->fathers_name.end(), this->name_of_op);
			int index = (int)std::distance(std::begin(((base_op<T>*)(this->sons[i]))->fathers_name), ite1);
			//son->x=self->y
			(*(((base_op<T>*)(this->sons[i]))->x))[index] = this->y;

			//self->dy=son->dx
			this->dy->push_back((*(((base_op<T>*)(this->sons[i]))->dx))[index]);
		}
		//cout << "initparameter over" << endl;
	}

	//run forward mark=0,else mark=1 run backward
	void ward_run(int mark) 
	{
		if(mark == 0) 
		{forward_function();}
		else
		{backward_function();}
	}

	//base_op()
	base_op(){};    
	~base_op() {   
	};

	//static new object, over load
	//no varible input 
	static base_op<T>* getObejct(vector<constant<T>*>* constant_N_o, string name_o, char* Tensor_des = "")
	{	
		base_op<T>* result = new base_op<T>;

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}
		result->name_of_op = name_o;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;
		
		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();

		base_op<T>::global_graph->insert_v(result->name_of_op, result);
		
		return result;
	}

	//has varible input
	static base_op<T>* getObejct(vector<constant<T>*>* constant_N_o, vector<variable<T>*>* w_o, string name_o, char* Tensor_des = "")
	{
		
		base_op<T>* result = new base_op<T>;
		result->name_of_op = name_o;

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;
	
		result->w = w_o;
		result->w_num = w_o->size();//varible number
		
		for (typename vector<variable<T>*>::const_iterator iter = w_o->cbegin(); iter != w_o->cend(); iter++)
		{   if ((*iter)->trainable==true && !base_op<T>::global_w_trainable->if_find((*iter)->var_name))
			    base_op<T>::global_w_trainable->insert_v((*iter)->var_name, (*iter));
		}

		result->cons = constant_N_o;
		result->cons_num = constant_N_o->sise();
		
		result->name_of_op = name_o;
		
		base_op<T>::global_graph->insert_v(result->name_of_op, result);
		result->input_op_eq_const = true;//if this op is tansported from the constant,backward no need to start! 
		return result;
	}

	//1 op, no varible input 
	static base_op<T>* getObejct(base_op<T>* op, vector<constant<T>*>* constant_N_o, string name_o,char* Tensor_des = "")
	{
		base_op<T>* result = new base_op<T>;
		result->name_of_op = name_o;
		
		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;

		result->fathers.push_back(op);
		result->fathers_name.push_back(op->name_of_op);
		result->fathers_num += 1;
		result->xdx_num += 1;

		op->sons.push_back(result);
		op->sons_name.push_back(result->name_of_op);
		op->sons_num += 1;
		op->ydy_num += 1;

		base_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//1 op, varible input 
	static base_op<T>* getObejct(base_op<T>* op, vector<constant<T>*>* constant_N_o,vector<variable<T>*>* w_o, string name_o, char* Tensor_des = "")
	{
		base_op<T>* result = new base_op<T>;
		result->name_of_op = name_o;
	
		result->w = w_o;
		result->w_num = w_o->size();//varible number
		for (typename vector<variable<T>*>::const_iterator iter = w_o->cbegin(); iter != w_o->cend(); iter++)
		{
			if ((*iter)->trainable == true && !base_op<T>::global_w_trainable->if_find((*iter)->var_name))
				base_op<T>::global_w_trainable->insert_v((*iter)->var_name, (*iter));
		}
		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;
		
		result->fathers.push_back(op);
		result->fathers_name.push_back(op->name_of_op);
		result->fathers_num += 1;
		result->xdx_num += 1;

		op->sons.push_back(result);
		op->sons_name.push_back(result->name_of_op);
		op->sons_num += 1;
		op->ydy_num += 1;

		base_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//2 ops, no varible input 
	static base_op<T>* getObejct(base_op<T>* op1, base_op<T>* op2, vector<constant<T>*>* constant_N_o, string name_o, char* Tensor_des = "")
	{   
		base_op<T>* result = new base_op<T>;
		result->name_of_op = name_o;
		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();


		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;

		result->fathers.push_back(op1);
		result->fathers_name.push_back(op1->name_of_op);
		result->fathers.push_back(op2);
		result->fathers_name.push_back(op2->name_of_op);
		result->fathers_num += 2;
		result->xdx_num += 2;

		op1->sons.push_back(result);
		op1->sons_name.push_back(result->name_of_op);
		op1->sons_num += 1;
		op1->ydy_num += 1;

		op2->sons.push_back(result);
		op2->sons_name.push_back(result->name_of_op);
		op2->sons_num += 1;
		op2->ydy_num += 1;

		base_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}

	//2 ops, varible input 
	static base_op<T>* getObejct(base_op<T>* op1, base_op<T>* op2, vector<constant<T>*>* constant_N_o, vector<variable<T>*>* w_o, string name_o,char* Tensor_des="")
	{
		base_op<T>* result = new base_op<T>;
		result->name_of_op = name_o;

		result->w = w_o;
		result->w_num = w_o->size();//varible number
		for (typename vector<variable<T>*>::const_iterator iter = w_o->cbegin(); iter != w_o->cend(); iter++)
		{
			if ((*iter)->trainable == true && !base_op<T>::global_w_trainable->if_find((*iter)->var_name))
				base_op<T>::global_w_trainable->insert_v((*iter)->var_name, (*iter));
		}

		for (typename vector<constant<T>*>::const_iterator iter = constant_N_o->cbegin(); iter != constant_N_o->cend(); iter++)
		{
			if ((*iter)->placeholder == 1 && !base_op<T>::global_placehold_constant->if_find((*iter)->con_name))
				base_op<T>::global_placehold_constant->insert_v((*iter)->con_name, (*iter));
		}

		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;

	
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
	
		base_op<T>::global_graph->insert_v(result->name_of_op, result);
		return result;
	}
};
template<class T>  graph<T, base_op>*   base_op<T>::global_graph;//for static
template<class T>  graph<T, variable>*  base_op<T>::global_w_trainable;  //for static
template<class T>  graph<T, constant>*  base_op<T>::global_placehold_constant;  //for static
#endif // !_BASE_OP_CUH