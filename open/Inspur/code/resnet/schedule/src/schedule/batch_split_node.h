#pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "inner_thread.h"
#include "blocking_queue.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	class BatchSplitParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_thread_num, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)

	public:
		void Construct(BatchSplitParam& param) {
			m_thread_num = param.m_thread_num;
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
		}
		BatchSplitParam(size_t thread_num, size_t queue_depth, size_t batch_size) {
			m_thread_num = thread_num;
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
		};
		BatchSplitParam(BatchSplitParam& param) {
			Construct(param);
		};
		BatchSplitParam(BatchSplitParam&& param) noexcept {
			Construct(param);
		};
		BatchSplitParam& operator=(BatchSplitParam&& param) noexcept {
			Construct(param);
			return *this;
		};
		virtual ~BatchSplitParam() {};
	};

	class BatchSplitNode : public ScheduleNode, public InnerThread {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, BatchSplitParam)
		CREATE_SIMPLE_ATTR_GET(m_queue_free, BlockingQueue<shared_ptr<Batch<QuerySample>>>)
		CREATE_SIMPLE_ATTR_GET(m_queue_full, BlockingQueue<shared_ptr<Batch<QuerySample>>>)

	public:
		BatchSplitNode(string name, string type, vector<string> bottom, vector<string> top,
			BatchSplitParam param)
			: ScheduleNode(name, type, bottom, top),
			InnerThread(param.get_m_thread_num()),
			m_param(param) {};
		~BatchSplitNode() {};

		void Init();
		void Run();
		void Finalize();
		void EntryMulti(size_t thread_id);
		void* GetFreeQueue() { return &m_queue_free; };
		void* GetFullQueue() { return &m_queue_full; };
	};

	class InputFile;
	class BatchSplitNodeParser : public NodeParser {
	public:
		BatchSplitNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~BatchSplitNodeParser() {};

		shared_ptr<BatchSplitNode> Run(Node node);
		BatchSplitParam ParseParam();
	};

}
