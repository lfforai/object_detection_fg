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
#include <helper_string.h>     // helper for string parsing
#include "base_op.cuh"
using namespace std;

#ifndef _HASH_global_CUH
#define  _HASH_global_CUH
//#define NDEBUG 

template<class vlaueT, template<class vlaueT> typename classobject>
//template<class vlaueT, typename classobject = base_op<vlaueT>>
class graph {
private:
public:
	unordered_map<string, classobject<vlaueT>*> un_map;
	
	void insert_v(string str, classobject<vlaueT>* value)
	{
		if (!this->if_find(str))
			un_map.insert(std::pair<string, classobject<vlaueT>*>{str, value});
		else
		{    
			cout<<"op name repeat:"<<str<<endl;
			exit(0);
		}
	}

	void insert_v_repeat(string& str, classobject<vlaueT>* value)
	{
		if (!this->if_find(str))
			un_map.insert(std::pair<string, classobject<vlaueT>*>{str, value});
		else
		{
			while (this->if_find(str)) {
				str = str + "_1";
			}
			un_map.insert(std::pair<string, classobject<vlaueT>*>{str, value});
			//cout << "op name repeat:" << str << endl;
		}
	}

	void showvalue()
	{
		for (const auto& e : this->un_map)
		{
			std::cout << e.first << " is " << e.second << std::endl;
		}
	}

	vector<classobject<vlaueT>*>* getallvalue()
	{
		vector<classobject<vlaueT>*>* result=new vector<classobject<vlaueT>*>;
		for(const auto& e : this->un_map)
		  {
			result->push_back((classobject<vlaueT>*)e.second);
		  }
		return result;
	}

	vector<string> allKeys(){
		vector<string> result;
		for (const auto& e : this->un_map)
			result.push_back(e.first);
		return result;
	}

	bool if_find(string key) {
		bool result = true;
		if (un_map.find(key) != un_map.end())
			return result;
		else
			result = false;
		return result;
	}
};

//template class graph<int,base_op>;
#endif // !_HASH_global_CUH
