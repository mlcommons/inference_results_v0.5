#pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "../inference/inference.h"
#include "inner_thread.h"
#include "data_set.h"
#include "blocking_queue.h"
#include "schedule.h"

using namespace std;
using namespace schedule;


namespace schedule {

	class InferenceParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_gpu_engine_num, size_t)

	public:
		void Construct(InferenceParam& param) {
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
			m_gpu_engine_num = param.m_gpu_engine_num;
		}
		InferenceParam(size_t queue_depth, size_t batch_size, size_t gpu_engine_num) {
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
			m_gpu_engine_num = gpu_engine_num;
		};
		InferenceParam(InferenceParam& param) {
			Construct(param);
		};
		InferenceParam(InferenceParam&& param) noexcept {
			Construct(param);
		};
		InferenceParam& operator=(InferenceParam&& param) noexcept {
			Construct(param);
			return *this;
		};
		virtual ~InferenceParam() {};
	};

	class InferenceNode : public ScheduleNode, public InnerThread {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, InferenceParam)
		CREATE_SIMPLE_ATTR_SET_GET(m_engines, vector<shared_ptr<inference::GpuEngine>>)
		CREATE_SIMPLE_ATTR_GET(m_queues_free, vector<shared_ptr<BlockingQueue<shared_ptr<Batch<MemoryData>>>>>)
		CREATE_SIMPLE_ATTR_GET(m_queues_full, vector<shared_ptr<BlockingQueue<shared_ptr<Batch<MemoryData>>>>>)

	public:
		InferenceNode(string name, string type, vector<string> bottom, vector<string> top,
			InferenceParam param)
			: ScheduleNode(name, type, bottom, top),
			InnerThread(param.get_m_gpu_engine_num()),
			m_param(param) {};
		~InferenceNode() {};

		void Init();
		void Run();
		void Finalize();
		void EntryMulti(size_t thread_id);
		void* GetFreeQueue() { return &m_queues_free; };
		void* GetFullQueue() { return &m_queues_full; };
		string GetDataFormat();
	};

	class InputFile;
	class InferenceNodeParser : public NodeParser {
	public:
		InferenceNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~InferenceNodeParser() {};

		shared_ptr<InferenceNode> Run(Node node);
		InferenceParam ParseParam();
	};

}
