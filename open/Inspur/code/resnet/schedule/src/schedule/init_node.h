#pragma once
#include <vector>
#include <memory>
#include <functional>

#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "settings/mlperf_settings.h"
#include "inner_thread.h"
#include "data_set.h"
#include "blocking_queue.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	class InitParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)

	public:
		void Construct(InitParam& param) {
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
		}
		InitParam(size_t queue_depth, size_t batch_size) {
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
		};
		InitParam(InitParam& param) {
			Construct(param);
		};
		InitParam(InitParam&& param) noexcept {
			Construct(param);
		};
		InitParam& operator=(InitParam&& param) noexcept {
			Construct(param);
			return *this;
		};
		virtual ~InitParam() {};
	};

	class InitNode : public ScheduleNode {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, InitParam)
		CREATE_SIMPLE_ATTR_GET(m_queue_free, BlockingQueue<shared_ptr<Batch<QuerySample>>>)
		CREATE_SIMPLE_ATTR_GET(m_queue_full, BlockingQueue<shared_ptr<Batch<QuerySample>>>)

	public:
		InitNode(string name, string type, vector<string> bottom, vector<string> top,
			InitParam param)
			: ScheduleNode(name, type, bottom, top),
			m_param(param) {};
		~InitNode() {};

		void Init();
		void Finalize();
		void* GetFreeQueue() { return &m_queue_free; };
		void* GetFullQueue() { return &m_queue_full; };
	};

	class InputFile;
	class InitNodeParser : public NodeParser {
	public:
		InitNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~InitNodeParser() {};

		shared_ptr<InitNode> Run(Node node);
		InitParam ParseParam();
	};

}
