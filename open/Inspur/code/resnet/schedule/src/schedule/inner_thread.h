#pragma once
#include <map>
#include <vector>
#include <cmath>
#include <ctime>
#include <memory>
#include <thread>
#include <atomic>

#include "../common/macro.h"

using namespace std;


namespace schedule {

	class InnerThread {
		CREATE_SIMPLE_ATTR_SET_GET(m_thread_num, size_t)
		CREATE_SIMPLE_ATTR_GET(m_threads, vector<thread>)

	public:
		InnerThread(size_t thread_num);
		virtual ~InnerThread() {}

		void Start();
		void Stop(bool wait_all=true);
		void WaitAll();

		bool IsStarted(size_t id = 0) const {
			return m_threads[id].joinable();
		}

		size_t ThreadsNum() const {
			return m_thread_num;
		}

	protected:
		virtual void EntrySingle() { EntryMulti(0); }
		virtual void EntryMulti(size_t id) {}
			
		void Entry(size_t thread_id);

		bool MustStop(size_t id);
	};

}