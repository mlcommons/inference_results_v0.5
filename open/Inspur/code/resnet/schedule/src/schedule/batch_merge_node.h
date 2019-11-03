#pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "inner_thread.h"
#include "data_set.h"
#include "blocking_queue.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	class BatchMergeParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_thread_num, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_merge_time_threshold_ns, size_t)

	public:
		void Construct(BatchMergeParam& param) {
			m_thread_num = param.m_thread_num;
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
			m_merge_time_threshold_ns = param.m_merge_time_threshold_ns;
		}
		BatchMergeParam(size_t thread_num, size_t queue_depth, size_t batch_size, size_t merge_time_threshold_ns) {
			m_thread_num = thread_num;
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
			m_merge_time_threshold_ns = merge_time_threshold_ns;
		};
		BatchMergeParam(BatchMergeParam& param) {
			Construct(param);
		};
		BatchMergeParam(BatchMergeParam&& param) noexcept {
			Construct(param);
		};
		BatchMergeParam& operator=(BatchMergeParam&& param) noexcept {
			Construct(param);
			return *this;
		};
		virtual ~BatchMergeParam() {};
	};

	class BatchMergeNode : public ScheduleNode, public InnerThread {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, BatchMergeParam)
		CREATE_SIMPLE_ATTR_GET(m_queue_free, BlockingQueue<shared_ptr<Batch<QuerySample>>>)
		CREATE_SIMPLE_ATTR_GET(m_queue_full, BlockingQueue<shared_ptr<Batch<QuerySample>>>)

	public:
		BatchMergeNode(string name, string type, vector<string> bottom, vector<string> top,
			BatchMergeParam param)
			: ScheduleNode(name, type, bottom, top),
			InnerThread(param.get_m_thread_num()),
			m_param(param) {};
		~BatchMergeNode() {};

		void Init();
		void Run();
		void Finalize();
		void EntryMulti(size_t thread_id);
		void* GetFreeQueue() { return &m_queue_free; };
		void* GetFullQueue() { return &m_queue_full; };
	};

	class InputFile;
	class BatchMergeNodeParser : public NodeParser {
	public:
		BatchMergeNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~BatchMergeNodeParser() {};

		shared_ptr<BatchMergeNode> Run(Node node);
		BatchMergeParam ParseParam();
	};

}
