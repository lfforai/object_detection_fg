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

	void init_parameter() {
		vector<string> keys = this->graph_global_active->allKeys();
		for(typename vector<string>::const_iterator iter = keys.cbegin(); iter != keys.cend(); iter++)
		   {
			  ((base_op<T>*)this->graph_global_active->un_map[(*iter)])->initparameter();
		   }
	}

	void forward_function(int id){
		vector<string> v_1;
		v_1.assign(this->forwardkeys.begin(), this->forwardkeys.end());
		//cout << "my id :" << id << endl;
		//ShowVec(v_1);

		int n=0;//random num;
		int mark;// return 1:can run, 0:is_backwarding or be finished, -1:wait for son ready

		int size_now = v_1.size();
		int outbreak = 0;

		while (outbreak==0){
			  if (n > size_now - 1)
				  n = 0;
		      if(size_now>1)
				  { 
				    // return 1:can run, 0:is_backwarding or be finished, -1:wait for son ready
				    mark=this->graph_global_active->un_map[v_1[n]]->if_forward_start_run();
					if (mark == 1)//run op forward
					   {this->graph_global_active->un_map[v_1[n]]->ward_run(0);
						 v_1.erase(v_1.begin()+n);
						 size_now = v_1.size();//aftering be removed the value,charge size of the vector 
						 n = 0;
					   }
					else if(mark== 0)// give up the op
					{
						v_1.erase(v_1.begin()+n);
					    size_now = v_1.size();//aftering be removed the value,charge size of the vector
						n = 0;
					}
				  }
				else //if only 1 op left,wait until deal with it, or it be dealed by other thread
				 {  
					while(true){
						mark = this->graph_global_active->un_map[v_1[0]]->if_forward_start_run();
						if(mark == 1)//run op forward
						  {
							this->graph_global_active->un_map[v_1[0]]->ward_run(0);
							outbreak = 1;
							break;
						  }
						else if(mark==0){
							outbreak = 1;
							break;
						}
					}
					
				 }
			    n = n + 1;
			}
		activeThreadsNum -= 1;
	}

	void backward_function(int id) {
		vector<string> v_1;
		v_1.assign(this->forwardkeys.begin(), this->forwardkeys.end());

		int mark;// return 1:can run, 0:is_backwarding or be finished, -1:wait for son ready

		int size_now = v_1.size();
		int outbreak = 0;

		int n = 0;//random num;

		while (outbreak == 0) {
			if (n > size_now - 1)
				n = 0;
			if (size_now > 1)
			{
				// return 1:can run, 0:is_backwarding or be finished, -1:wait back son ready
				mark = this->graph_global_active->un_map[v_1[n]]->if_backward_start_run();
				//cout<<mark<<endl;
				if (mark == 1)//run op backward
				{
					this->graph_global_active->un_map[v_1[n]]->ward_run(1);
					v_1.erase(v_1.begin() + n);
					size_now = v_1.size();//aftering be removed the value,charge size of the vector 
					n = 0;
				}
				else if (mark == 0)// give up the op
				{
					v_1.erase(v_1.begin() + n);
					size_now = v_1.size();//aftering be removed the value,charge size of the vector
					n = 0;
				}
			}
			else //if only 1 op left,wait until deal with it, or it be dealed by other thread
			{
				while (true) {
					mark = this->graph_global_active->un_map[v_1[0]]->if_backward_start_run();
					if (mark == 1)//run op backward
					{
						this->graph_global_active->un_map[v_1[0]]->ward_run(1);
						outbreak = 1;
						break;
					}
					else if (mark == 0) {
						outbreak = 1;
						break;
					}
				}

			}
			n +=1;
		}
		activeThreadsNum -= 1;
	}

	void forward_start(int UseMulThread){
		if (this->mark_InitParameter == 0)
		{
			this->init_vector();
			this->init_parameter();
		}
		this->activeThreadsNum = this->MaxThreadsNum;
		if (UseMulThread == 1){//use mulity threads
		   int num=this->MaxThreadsNum;
		   std::thread  **p=(std::thread**)malloc(num*sizeof(std::thread**));
		   for(int i = 0; i < num; i++){ 
			   p[i] = new std::thread(&graph_active<T>::forward_function,this,i);
		   }
		   for (int i = 0; i < num; i++)
		   {   
			   p[i]->detach();
		   }
		   while (activeThreadsNum > 0) {
		       //main thread 
		   }
		   for (int i = 0; i < num; i++) {
			   free(p[i]);
		   }
		   free(p);
		}
		else 
		{   //use only one thread
			forward_function(1);
		}

		this->forwardover = 1;
	}

	void backward_start(int UseMulThread) {
		if (forwardover != 1)
		{
			this->forward_start(UseMulThread);
		}
		this->activeThreadsNum = this->MaxThreadsNum;
		if (UseMulThread == 1) {//use mulity threads
			int num = this->MaxThreadsNum;
			std::thread  **p = (std::thread**)malloc(num * sizeof(std::thread**));
			for (int i = 0; i < num; i++) {
				p[i] = new std::thread(&graph_active<T>::backward_function, this, i);
			}
			for (int i = 0; i < num; i++)
			{
				p[i]->detach();
			}
			while (activeThreadsNum > 0) {
				//main thread 
			}
			for (int i = 0; i < num; i++) {
				free(p[i]);
			}
			free(p);
		}
		else
		{   //use only one thread
			backward_function(1);
		}
		this->backwardover = 1;
	}

	static graph_active<T>* getobject(graph<T, base_op>* graph_global_active_o){
		//create new object
		graph_active<T>* object_ga =new graph_active<T>;
		
		//catch graph 
		object_ga->graph_global_active = graph_global_active_o;
		
		//thread nums
		int maxthreadnum = object_ga->CatchCpuThreadsMaxNum();
		object_ga->MaxThreadsNum = (maxthreadnum>1)?(int)(maxthreadnum/2):1; //use half threadnum 
		object_ga->activeThreadsNum=object_ga->MaxThreadsNum;
		
		//forwardkeys= backwardkeys
		object_ga->forwardkeys=object_ga->graph_global_active->allKeys();//init forwardkeys
		//object_ga->backwardkeys.assign(object_ga->backwardkeys.begin(), object_ga->backwardkeys.end());//assign backwardkeys
		
		return object_ga;
	}
};
#endif // !_GRAPH_ACTIVE_THREADS_CU