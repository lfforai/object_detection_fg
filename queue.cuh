#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <atomic>
#include <string>
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
#include <assert.h>
#include <mutex>
#include <ctime> 
#include "fp16_dev.h"
using namespace std;

#ifndef _QUEUE_CUH_
#define _QUEUE_CUH_

template<class T>
class threadsafe_queue
{
public:
	threadsafe_queue()
	{
		m_bTermination = false;
	}
	~threadsafe_queue(void)
	{

	}
	//（1）没有调用termination时，每调用一次出队一个元素，直到队列为空本方法阻塞线程。
	//（2）在调用了termination后，本方法永不阻塞，如果原本已经处于阻塞状态，接触阻塞状态。
	//（3）返回true时，value值有效。返回false时，value值无效。调用了termination且队列为空时返回false.
	bool wait_and_pop(T& value)
	{
		unique_lock<mutex> lk(mut);
		data_cond.wait(lk, [this] {return ((!data_queue.empty()) || m_bTermination); });
		//不为空则出队
		if (!data_queue.empty())
		{
			value = move(*data_queue.front());
			data_queue.pop();
			return true;
		}
		//队列为空则返回失败
		return false;
	}

	//队列为空返回false
	bool try_pop(T& value)
	{
		lock_guard<mutex> lk(mut);
		if (data_queue.empty())
		{
			return false;
		}
		value = move(*data_queue.front());
		data_queue.pop();
		return true;
	}

	std::shared_ptr<T> wait_and_pop()
	{
		unique_lock<mutex> lk(mut);
		data_cond.wait(lk, [this] {return ((!data_queue.empty()) || m_bTermination); });
		if (!data_queue.empty())
		{
			shared_ptr<T> res = data_queue.front();
			data_queue.pop();
			return res;
		}
		return nullptr;
	}

	//队列为空返回null
	std::shared_ptr<T> try_pop()
	{
		lock_guard<mutex> lk(mut);
		if (data_queue.empty())
		{
			return nullptr;
		}
		shared_ptr<T> res = data_queue.front();
		data_queue.pop();
		return res;
	}
	//插入一项
	void push(T new_value)
	{
		if (m_bTermination)
		{
			return;
		}
		shared_ptr<T> data(make_shared<T>(move(new_value)));
		lock_guard<mutex> lk(mut);
		data_queue.push(data);
		data_cond.notify_one();
	}
	bool empty() const
	{
		lock_guard<mutex> lk(mut);
		return data_queue.empty();
	}
	int size()
	{
		lock_guard<mutex> lk(mut);
		return data_queue.size();
	}
	//设置队列为退出状态。在退出状态下，忽略入队，可以执行出队，但当队列为空时，wait_and_pop不会阻塞。
	void termination()
	{
		lock_guard<mutex> lk(mut);
		m_bTermination = true;
		data_cond.notify_all();
	}
	//是退出状态吗
	bool is_termination()
	{
		return m_bTermination;
	}
private:
	mutable mutex mut;
	queue<shared_ptr<T>> data_queue;
	condition_variable data_cond;
	atomic<bool> m_bTermination;
};
#endif //!_QUEUE_CUH_