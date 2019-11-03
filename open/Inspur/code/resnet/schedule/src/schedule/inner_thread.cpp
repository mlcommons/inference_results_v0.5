#include <sstream>

#include "inner_thread.h"


namespace schedule {

	InnerThread::InnerThread(size_t thread_num) : m_thread_num(thread_num), m_threads(thread_num) {
	}

	void InnerThread::Start() {
		try {
			m_threads.resize(m_thread_num);
			for (size_t id = 0; id < m_thread_num; ++id) {
				m_threads[id] = thread(&InnerThread::Entry, this, id);
			}
		}
		catch (exception e) {
		}
	}

	void InnerThread::Stop(bool wait_all) {
	}

	void InnerThread::Entry(size_t thread_id) {
		if (m_thread_num == 1) {
			EntrySingle();
		}
		else {
			EntryMulti(thread_id);
		}
	}

	void InnerThread::WaitAll() {
		try {
			for (size_t id = 0; id < m_thread_num; ++id) {
				if (IsStarted(id)) {
					m_threads[id].join();
				}
			}
		}
		catch (exception e) {
		}
	}

	bool InnerThread::MustStop(size_t thread_id) {
		return false;
	}

}