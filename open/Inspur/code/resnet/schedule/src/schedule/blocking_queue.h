#pragma once
#include <queue>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "../common/macro.h"

using namespace std;


namespace schedule {

	template<typename T>
	class BlockingQueue {
	public:
		BlockingQueue();
		~BlockingQueue();

		void Push(const T& t);
		T Pop();

		bool TryPeek(T* t);
		bool TryPop(T* t);

		T Peek();
		size_t Size();
		bool NonBlockingSize(size_t* size);

		void Clear();

	private:
		mutable mutex m_mutex;
		condition_variable m_condition_var;
		queue<T> m_queue;
	};

}