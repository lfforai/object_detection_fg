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
#include "base_op.cuh"

#include <windows.h>
#include <wincrypt.h>
#include "stdarg.h"


using namespace std;
#ifndef _SUM_OP_CUH
#define _SUM_OP_CUH
template<class T>
class sum_op :public base_op<T>
{
public:
	static sum_op<T>* getObejct(vector<constant<T>*>* constant_N_o, string name_o)
	{
		sum_op<T>* result = new sum_op<T>;

		result->name_of_op = name_o;
		result->x = new vector<constant<T>*>;
		result->dx = new vector<constant<T>*>;
		result->dy = new vector<constant<T>*>;
		result->dw = new vector<variable<T>*>;

		result->cons = constant_N_o;
		result->cons_num = constant_N_o->size();

		sum_op<T>::global_graph->insert_v(result->name_of_op, result);

		return result;
	}

	//reload the backward_function,make sure last of the function must be backward_over = 1
    void backward_function() {
		 

	}

	//reload the forward_function,make sure last of the function must be forward_over = 1
	void forward_function() {
                                                                        
		forward_over = 1;
		//cout << this->name_of_op << endl;
	}

};
#endif // !_SUM_OP_CUH

