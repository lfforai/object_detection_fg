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
#include "constant_class.cuh"
#include "weigh_class.cuh"

#include <windows.h>
#include <wincrypt.h>
#include "stdarg.h"
#include "gpumathtool.cuh"
#include "queue.cuh"
#include "cudnn.h"

//#include "fp16_dev.h"

using namespace std;

#ifndef _BASE_OP_CUH
#define _BASE_OP_CUH

//base_op
static std::atomic<int> total_not_finish_ops_num = -1;
threadsafe_queue<string>*  queue_forward_canbe_used_ops=new threadsafe_queue<string>;

//now can be forward or backoward ops namethreadsafe_queue<string>* queue_forward_canbe_used_ops;//now can be forward or backoward ops name
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


	 //reload the backward_function,make sure last of the function must be backward_over = 1
	virtual void backward_function() {

	 }

	 //reload the forward_function,make sure last of the function must be forward_over = 1
	 virtual void forward_function() {

	 }

	 //--------------------------op-value--------------------------------------
	 string name_of_op; //mast be unique;
	 cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc; //for net 

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
		this->dy_sum=(*dy)[0];
		int n = this->dy->size();
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
				//for(int i = 1; i < n; i++)
				//  {
				//	for(int j = 0; j < length; j++)
				//	  {
				//		dy_sum->x[j] = dy_sum->x[j] + ((constant<T>*)(*dy)[i])->x[j];
				//	  }
				// }
				cout<<"sum_dy only suport gpu"<<endl;
				exit(0);
			  }
		 }
	}

	//w 
	variable<T>* w;
	variable<T>* dw;
    //constant
	constant<T>* cons;

	//dstValue =alpha[0] * resultValue + beta[0] * priordstValue
	T alpha;//all most 1.0
	T beta;//all most 0.0

	static graph<T, base_op>* global_graph;//all ops belong to the global_graph
	static graph<T, variable>* global_w_trainable;
	static graph<T, variable>* global_dw_trainable;
	static graph<T, constant>* global_placehold_constant;
	//static threadsafe_queue<string>*  queue_forward_canbe_used_ops;
	
	
	std::vector<base_op<T>*> fathers;//up ops
	std::vector<string> fathers_name;
	int fathers_num = 0; //number of the input ops=N

	std::vector<base_op<T>*> sons;//down ops
	std::vector<string> sons_name;
	int sons_num = 0; //number of the input ops=N

	// if neededBackwark_dx dw==true ,computer dx,dw  
	bool neededBackwark_dx = true;
	bool neededBackwark_dw = true;

	//thread safe
    std::atomic<int> fathers_finshed_size=-1;// if ==0 can be start to forward
	std::atomic<int> sons_finshed_size =-1;//if ==0 can be start to backward
	std::atomic<int> forwardover = -1;
	std::atomic<int> backwardover = -1;

	//--------------------------op-function------------------------------------
    inline void initvector(){
		fathers_finshed_size = fathers.size();//init  
		sons_finshed_size = sons.size();//init  

		if (this->neededBackwark_dx == true)
		{
			//used by index::a[index],so must be inited before backward
			for(int i = 0; i < this->xdx_num; i++)
			{
				this->x->push_back(0);
				this->dx->push_back(0);
			}
		}	
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
};
template<class T>  graph<T, base_op>*   base_op<T>::global_graph;//for static
template<class T>  graph<T, variable>*  base_op<T>::global_w_trainable;  //for static
template<class T>  graph<T, variable>*  base_op<T>::global_dw_trainable;  //for static
template<class T>  graph<T, constant>*  base_op<T>::global_placehold_constant;  //for static
//template<>   threadsafe_queue<string>*  base_op<string>::queue_forward_canbe_used_ops;
#endif // !_BASE_OP_CUH