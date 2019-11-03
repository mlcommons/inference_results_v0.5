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

	class GpuScheduleParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_thread_num, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)

	public:
		void Construct(GpuScheduleParam& param) {
			m_thread_num = param.m_thread_num;
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
		}
		GpuScheduleParam(size_t thread_num, size_t queue_depth, size_t batch_size) {
			m_thread_num = thread_num;
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
		}
		GpuScheduleParam(GpuScheduleParam& param) {
			Construct(param);
		}
		GpuScheduleParam(GpuScheduleParam&& param) noexcept {
			Construct(param);
		}
		GpuScheduleParam& operator=(GpuScheduleParam&& param) noexcept {
			Construct(param);
			return *this;
		}
		virtual ~GpuScheduleParam() {};
	};

	class GpuScheduleNode : public ScheduleNode, public InnerThread {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, GpuScheduleParam)

	public:
		GpuScheduleNode(string name, string type, vector<string> bottom, vector<string> top,
			GpuScheduleParam param)
			: ScheduleNode(name, type, bottom, top),
			InnerThread(param.get_m_thread_num()),
			m_param(param) {};
		~GpuScheduleNode() {};

		void Init();
		void Run();
		void Finalize();
		void EntryMulti(size_t thread_id);
	};

	class InputFile;
	class GpuScheduleNodeParser : public NodeParser {
	public:
		GpuScheduleNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~GpuScheduleNodeParser() {};

		shared_ptr<GpuScheduleNode> Run(Node node);
		GpuScheduleParam ParseParam();
	};

}
