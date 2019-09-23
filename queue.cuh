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
	//��1��û�е���terminationʱ��ÿ����һ�γ���һ��Ԫ�أ�ֱ������Ϊ�ձ����������̡߳�
	//��2���ڵ�����termination�󣬱������������������ԭ���Ѿ���������״̬���Ӵ�����״̬��
	//��3������trueʱ��valueֵ��Ч������falseʱ��valueֵ��Ч��������termination�Ҷ���Ϊ��ʱ����false.
	bool wait_and_pop(T& value)
	{
		unique_lock<mutex> lk(mut);
		data_cond.wait(lk, [this] {return ((!data_queue.empty()) || m_bTermination); });
		//��Ϊ�������
		if (!data_queue.empty())
		{
			value = move(*data_queue.front());
			data_queue.pop();
			return true;
		}
		//����Ϊ���򷵻�ʧ��
		return false;
	}

	//����Ϊ�շ���false
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

	//����Ϊ�շ���null
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
	//����һ��
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
	//���ö���Ϊ�˳�״̬�����˳�״̬�£�������ӣ�����ִ�г��ӣ���������Ϊ��ʱ��wait_and_pop����������
	void termination()
	{
		lock_guard<mutex> lk(mut);
		m_bTermination = true;
		data_cond.notify_all();
	}
	//���˳�״̬��
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