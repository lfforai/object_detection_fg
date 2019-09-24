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
#include <thread>         // std::thread 
#include <mutex>          // std::mutex

#include "base_op.cuh"
#include "hash_global.cuh"
#include "queue.cuh"

#include <windows.h>
#include <wincrypt.h>

using namespace std;
#ifndef _GRAPH_ACTIVE_THREADS_CU
#define _GRAPH_ACTIVE_THREADS_CU
template<class T>
class graph_active{
private:
	inline int CatchCpuThreadsMaxNum(){
		int result=std::thread::hardware_concurrency();
		return result;
	}

	int randEx() //real random
	{
		LARGE_INTEGER seed;
		QueryPerformanceFrequency(&seed);
		QueryPerformanceCounter(&seed);
		srand(seed.QuadPart);

		return rand();
	}
public:
	mutex m;
	int mark_InitParameter = 0;//make sure the parameters of the ops  are all cennected 
	int forwardover = 0;
	int backwardover = 0;

	graph<T, base_op>* graph_global_active;
	std::vector<string> forwardkeys;//used by forwardstart and backwardstart

	int MaxThreadsNum;
	atomic<int> activeThreadsNum = 0;//active thread

	void init_vector() {
		vector<string> keys = this->graph_global_active->allKeys();
		for (typename vector<string>::const_iterator iter = keys.cbegin(); iter != keys.cend(); iter++)
		{
			((base_op<T>*)this->graph_global_active->un_map[(*iter)])->initvector();
		}
	}

	
	void ward_function(int mark,int num) {
		string value;
		//int numcout = 0;
		while (total_not_finish_ops_num>0) {
			//if(((threadsafe_queue<string>*) base_op<T>::queue_forward_canbe_used_ops)->try_pop(value));
			if (queue_forward_canbe_used_ops->try_pop(value) && this->graph_global_active->if_find(value))	
			{
				((base_op<T>*)this->graph_global_active->un_map[value])->ward_run(mark);
				//numcout++;
			}
		}
		activeThreadsNum -= 1;
		//cout << "myid over:" << num <<"do job:"<<numcout<< endl;
	}
	
	//test all placeholder constants all be assgined
    void Placeholder_assgin_finished() {
		vector<string> keys =  base_op<T>::global_placehold_constant->allKeys();
		for (typename vector<string>::const_iterator iter = keys.cbegin(); iter != keys.cend(); iter++)
		{
		   if (((constant<T>*)((graph<T, constant>*) base_op<T>::global_placehold_constant)->un_map[(*iter)])->x_dim[0] == -1)
		   {
			   cout<<"placehold constant ::"<< *iter<<" not assgian init value!"<<endl;
			   exit(0);
		   }
		}
	}

	void Placeholder_assgin(vector<constant<T>*> cons){
		for (typename vector<constant<T>*>::const_iterator iter = cons.cbegin(); iter != cons.cend(); iter++)
		{
			//cout<< ((constant<T>*)(*iter))->con_name<<endl;
			if (((graph<T, constant>*) base_op<T>::global_placehold_constant)->if_find(((constant<T>*)(*iter))->con_name))
			{
				constant<T> * value = ((graph<T, constant>*) base_op<T>::global_placehold_constant)->un_map[((constant<T>*)(*iter))->con_name];
				value->initPlaceholder(((constant<T>*)(*iter))->x_dim_num, ((constant<T>*)(*iter))->x_dim, ((constant<T>*)(*iter))->x);
			}
			else
			{
				cout<<"no placehold constants name:"<< ((constant<T>*)(*iter))->con_name <<endl;
				exit(0);
			}
		}
	}

	//forward_or_backward=0:forward ,1:=backward 
	void ward_start(int UseMulThread,int forward_or_backward){
		if (this->mark_InitParameter == 0)
		{
			Placeholder_assgin_finished();
			this->init_vector();
			mark_InitParameter = 1;
		}

		//find frist ops ,put them into 
		if (forward_or_backward == 0) {
			vector<string> keys = this->graph_global_active->allKeys();
			for (typename vector<string>::const_iterator iter = keys.cbegin(); iter != keys.cend(); iter++)
			{
				if (((base_op<T>*)this->graph_global_active->un_map[(*iter)])->fathers.empty())
					//((threadsafe_queue<string>*) base_op<T>::
						queue_forward_canbe_used_ops->push(*iter);
			}
		}
		else {
			vector<string> keys = this->graph_global_active->allKeys();
			for (typename vector<string>::const_iterator iter = keys.cbegin(); iter != keys.cend(); iter++)
			{
				if (((base_op<T>*)this->graph_global_active->un_map[(*iter)])->sons.empty())
					//((threadsafe_queue<string>*) base_op<T>::
						queue_forward_canbe_used_ops->push(*iter);
			}
		}

		total_not_finish_ops_num = this->forwardkeys.size();//ops numbers

		//find start ops ,put it into  queue
		this->activeThreadsNum = this->MaxThreadsNum;
		if (UseMulThread == 1){//use mulity threads
		   //dicide threads num;
		   int num=this->MaxThreadsNum;
		   num=num>queue_forward_canbe_used_ops->size()? queue_forward_canbe_used_ops->size():num;
		   this->activeThreadsNum = num;
		   //set threads num
		   std::thread  **p=(std::thread**)malloc(num*sizeof(std::thread**));
		   for(int i = 0; i < num; i++){ 
			   p[i] = new std::thread(&graph_active<T>::ward_function,this,forward_or_backward,i);
		   }
		   for (int i = 0; i < num; i++)
		   {
			   if (i != num - 1)
				   p[i]->detach();
			   else
				   p[i]->join();
		   }

		   while (this->activeThreadsNum>0) {
			  // printf("this->activeThreadsNum %d \n", this->activeThreadsNum);
		   }

		   for (int i = 0; i < num; i++) {
			   free(p[i]);
		   }
		   free(p);
		}
		else 
		{   //use only one thread
			ward_function(forward_or_backward,1);
		}
		
	}

	static graph_active<T>* getobject(graph<T, base_op>* graph_global_active_o){
		//create new object
		graph_active<T>* object_ga =new graph_active<T>;
		
		//catch graph 
		object_ga->graph_global_active = graph_global_active_o;
		
		//thread nums
		int maxthreadnum = object_ga->CatchCpuThreadsMaxNum();
		//cout<<"cpu max thread::"<<maxthreadnum <<endl;
		object_ga->MaxThreadsNum = (maxthreadnum>1)?(int)(maxthreadnum/2):1; //use half threadnum 
		object_ga->activeThreadsNum=object_ga->MaxThreadsNum;
		
		object_ga->forwardkeys=object_ga->graph_global_active->allKeys();//init forwardkeys
		
		return object_ga;
	}
};
#endif // !_GRAPH_ACTIVE_THREADS_CU