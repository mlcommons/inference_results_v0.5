#pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "../postprocess/postprocess_base.h"
#include "inner_thread.h"
#include "data_set.h"
#include "blocking_queue.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	class PostProcessParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_thread_num, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_queue_depth, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_batch_size, size_t)

	public:
		void Construct(PostProcessParam& param) {
			m_thread_num = param.m_thread_num;
			m_queue_depth = param.m_queue_depth;
			m_batch_size = param.m_batch_size;
		}
		PostProcessParam(size_t thread_num, size_t queue_depth, size_t batch_size) {
			m_thread_num = thread_num;
			m_queue_depth = queue_depth;
			m_batch_size = batch_size;
		};
		PostProcessParam(PostProcessParam& param) {
			Construct(param);
		};
		PostProcessParam(PostProcessParam&& param) noexcept {
			Construct(param);
		};
		PostProcessParam& operator=(PostProcessParam&& param) noexcept {
			Construct(param);
			return *this;
		};
		virtual ~PostProcessParam() {};
	};

	class PostProcessNode : public ScheduleNode, public InnerThread {

		CREATE_SIMPLE_ATTR_SET_GET(m_param, PostProcessParam)
		CREATE_SIMPLE_ATTR_SET_GET(m_p_post, shared_ptr<postprocess::PostProcessBase<float>>)
		CREATE_SIMPLE_ATTR_GET(m_p_queue_free, void*)
		CREATE_SIMPLE_ATTR_GET(m_p_queue_full, void*)
			
	public:
		PostProcessNode(string name, string type, vector<string> bottom, vector<string> top,
			PostProcessParam param)
			: ScheduleNode(name, type, bottom, top),
			InnerThread(param.get_m_thread_num()),
			m_param(param) {};
		~PostProcessNode() {};

		void Init();
		void Run();
		void Finalize();
		void EntryMulti(size_t thread_id);
		void* GetFreeQueue() { return m_p_queue_free; };
		void* GetFullQueue() { return m_p_queue_full; };
	};

	class InputFile;
	class PostProcessNodeParser : public NodeParser {
	public:
		PostProcessNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~PostProcessNodeParser() {};

		shared_ptr<PostProcessNode> Run(Node node);
		PostProcessParam ParseParam();
	};

}
