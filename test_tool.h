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
		float secs =300;      //���帡���ͱ���secs
		clock_t delay;  //����clock_t���͵ı�������ʾ��ʱʱ��
		delay = secs * CLOCKS_PER_SEC;  //delay��ֵΪsecs ���� CLOCKS_PER_SECֵ�������������ת��ϵͳ��ʱ��
		clock_t start = clock();    //����clock_t���ͱ���start������ֵΪ��ǰϵͳ��ʱ��
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

class loadoperatrion{
public:
	string name;
	loadoperatrion(){
		this->name = "first";
	};

	loadoperatrion(string name_o){
		this->name=name_o;
	};

	friend loadoperatrion  operator+(loadoperatrion& op,loadoperatrion& po2)
	{
		loadoperatrion result;
		result.name = op.name + "+" + po2.name;
		printf("start + \n");
		return result;
	}

	friend loadoperatrion  operator/(loadoperatrion& op,loadoperatrion& po2) 
	{
		loadoperatrion result;
		result.name = op.name+"/"+ po2.name;
		printf("start / \n");
		return result;
	}

	friend loadoperatrion  operator*(loadoperatrion& op,loadoperatrion& op2) {
		loadoperatrion result;
		result.name = op.name + "*" + op2.name;
		printf("* start \n");
		return result;
	}
};


void test() {

	//vector test
	//vector<string>  avector;
	
	//avector.push_back("a");
	//avector.push_back("b");
	//avector.push_back("c");
	//avector.push_back("d");
	//vector<string>::iterator ite = find(avector.begin(), avector.end(), "a");
	//int index = (int)std::distance(std::begin(avector), ite);
	//printf("index:=%d \n", index);
	//avector[3] = "w";
	loadoperatrion A("A");
	loadoperatrion B("B");
	loadoperatrion C("C");
	B+A*(B/C);
}
