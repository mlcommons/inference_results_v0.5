#include "settings/mlperf_settings.h"
#include "../inference/inference.h"
#include "data_set.h"
#include "inner_thread.h"
#include "blocking_queue.h"


namespace schedule {

	template<typename T>
	BlockingQueue<T>::BlockingQueue() {}

	template<typename T>
	BlockingQueue<T>::~BlockingQueue() {}

	template<typename T>
	void BlockingQueue<T>::Push(const T& t) {
		unique_lock<mutex> lock(m_mutex);
		m_queue.push(t);
		lock.unlock();
		m_condition_var.notify_one();
	}

	template<typename T>
	T BlockingQueue<T>::Pop() {
		unique_lock<mutex> lock(m_mutex);
		while (m_queue.empty()) {
			m_condition_var.wait(lock);
		}
		T t(m_queue.front());
		m_queue.pop();
		return t;
	}

	template<typename T>
	bool BlockingQueue<T>::TryPeek(T* t) {
		unique_lock<mutex> lock(m_mutex);
		if (m_queue.empty()) {
			return false;
		}
		*t = m_queue.front();
		return true;
	}

	template<typename T>
	bool BlockingQueue<T>::TryPop(T* t) {
		unique_lock<mutex> lock(m_mutex);
		if (m_queue.empty()) {
			return false;
		}
		*t = m_queue.front();
		m_queue.pop();
		return true;
	}

	template<typename T>
	T BlockingQueue<T>::Peek() {
		unique_lock<mutex> lock(m_mutex);
		while (m_queue.empty()) {
			m_condition_var.wait(lock);
		}
		return m_queue.front();
	}

	template<typename T>
	size_t BlockingQueue<T>::Size() {
		unique_lock<mutex> lock(m_mutex);
		size_t size = m_queue.size();
		return size;
	}

	template<typename T>
	bool BlockingQueue<T>::NonBlockingSize(size_t* size) {
		unique_lock<mutex> lock(m_mutex);
		if (lock.owns_lock()) {
			*size = m_queue.size();
			return true;
		}
		return false;
	}

	template<typename T>
	void BlockingQueue<T>::Clear() {
		unique_lock<mutex> lock(m_mutex);
		while (!m_queue.empty()) {
			m_queue.pop();
		}
	}

	template class BlockingQueue<shared_ptr<Batch<QuerySample>>>;
	template class BlockingQueue<shared_ptr<Batch<ImageData>>>;
	template class BlockingQueue<shared_ptr<Batch<MemoryData>>>;
	template class BlockingQueue<shared_ptr<Batch<PredictResult<float>>>>;
	template class BlockingQueue<shared_ptr<Batch<PredictResult<inference::ResultTensor*>>>>;

}
