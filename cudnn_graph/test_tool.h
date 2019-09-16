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
#include<mutex>
#include <thread>         // std::thread 
#include<ctime> 

#include <windows.h>
#include <wincrypt.h>
using namespace std;
template<class T>

void ShowVec(const vector<T>& valList)
{
	for (typename vector<T>::const_iterator iter = valList.cbegin(); iter != valList.cend(); iter++)
	{
		cout << (*iter) << endl;
	}
}

//how to use mutex,no mater who use mutex
class testone {
	
public:	
	testone(){
	};
	int a = 0;
	mutex m;

	void func(int i) {
		printf("func  %d\n",i);
	};

	void sub_test(){
		m.lock();
		func(-1);
		a -= 1;
		printf("a: \n",a);
		m.unlock();
	}

	void add_test(int id){
		cout<<"my="<<id<<endl;
		m.lock();
		func(id);
		float secs =300;      //定义浮点型变量secs
		clock_t delay;  //定义clock_t类型的变量，表示延时时间
		delay = secs * CLOCKS_PER_SEC;  //delay赋值为secs 乘以 CLOCKS_PER_SEC值，将输入的秒数转化系统的时间
		clock_t start = clock();    //定义clock_t类型变量start，并赋值为当前系统的时间
		while (clock() - start < delay);
		for (size_t i = 0; i < 100; i++)
		{
			a += 1;
		}
		m.unlock();
	}

	void printlast() {
		cout<<a<<endl;
	}
};

void test() {
	//testone* obj=new testone();
	//int num = 4;
	//std::thread  **p = (std::thread**)malloc(num * sizeof(std::thread**));
	//for (int i = 0; i < num; i++) {
	//	p[i] = new std::thread(&testone::add_test,obj,i);
	//}
	//for (int i = 0; i < num; i++)
	//{
	//	p[i]->detach();
	//}
	//std::thread tj(&testone::sub_test, obj);
	//tj.join();
	//obj->printlast();

	//vector test
	vector<string>  avector;
	
	avector.push_back("a");
	avector.push_back("b");
	avector.push_back("c");
	avector.push_back("d");
	vector<string>::iterator ite = find(avector.begin(), avector.end(), "d");
	int index = (int)std::distance(std::begin(avector), ite);
	printf("index:=%d \n", index);
	avector[3] = "w";

}
