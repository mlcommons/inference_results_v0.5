#pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "inner_thread.h"
#include "data_set.h"
#include "blocking_queue.h"
#include "schedule.h"
#include "inference_node.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	class MemoryCopyParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_thread_num, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)

	public:
		void Construct(MemoryCopyParam& param) {
			m_thread_num = param.m_thread_num;
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
		}
		MemoryCopyParam(size_t thread_num, size_t queue_depth, size_t batch_size) {
			m_thread_num = thread_num;
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
		}
		MemoryCopyParam(MemoryCopyParam& param) {
			Construct(param);
		}
		MemoryCopyParam(MemoryCopyParam&& param) noexcept {
			Construct(param);
		}
		MemoryCopyParam& operator=(MemoryCopyParam&& param) noexcept {
			Construct(param);
			return *this;
		}
		virtual ~MemoryCopyParam() {};
	};

	class MemoryCopyNode : public ScheduleNode, public InnerThread {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, MemoryCopyParam)
		CREATE_SIMPLE_ATTR_GET(m_queues_free, vector<shared_ptr<BlockingQueue<shared_ptr<Batch<ImageData>>>>>)
		CREATE_SIMPLE_ATTR_GET(m_queues_full, vector<shared_ptr<BlockingQueue<shared_ptr<Batch<ImageData>>>>>)

	public:
		MemoryCopyNode(string name, string type, vector<string> bottom, vector<string> top,
			MemoryCopyParam param)
			: ScheduleNode(name, type, bottom, top),
			InnerThread(param.get_m_thread_num()),
			m_param(param) {};
		~MemoryCopyNode() {};

		void Init();
		void Run();
		void Finalize();
		void EntryMulti(size_t thread_id);
		void* GetFreeQueue() { return &m_queues_free; };
		void* GetFullQueue() { return &m_queues_full; };
	};

	class InputFile;
	class MemoryCopyNodeParser : public NodeParser {
	public:
		MemoryCopyNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~MemoryCopyNodeParser() {};

		shared_ptr<MemoryCopyNode> Run(Node node);
		MemoryCopyParam ParseParam();
	};

}
