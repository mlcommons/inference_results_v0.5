#include "../common/logging.h"
#include "../config/input_file.h"
#include "image_node.h"
#include "memory_copy_node.h"


namespace schedule {

	using Logger = common::Logger;

	shared_ptr<MemoryCopyNode> MemoryCopyNodeParser::Run(Node node) {
		std::string line = m_pfile->ReadLine();
		if (line != "memory_copy_param {")
			throw "read memory_copy_param fail";
		MemoryCopyParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<MemoryCopyNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	MemoryCopyParam MemoryCopyNodeParser::ParseParam() {
		size_t thread_num = 2;
		size_t queue_depth = 100;
		size_t batch_size = 32;

		std::map<std::string, size_t*> key_value;
		key_value["thread_num:"] = &thread_num;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return MemoryCopyParam(thread_num, queue_depth, batch_size);
	}

	void MemoryCopyNode::Init() {
		size_t engine_num = dynamic_cast<InferenceNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByType("Inference")[0])->get_m_param().get_m_gpu_engine_num();
		size_t batch_size = m_param.get_m_batch_size();
		for (size_t i = 0; i < engine_num; i++) {
			m_queues_free.push_back(make_shared<BlockingQueue<shared_ptr<Batch<ImageData>>>>());
			m_queues_full.push_back(make_shared<BlockingQueue<shared_ptr<Batch<ImageData>>>>());
			for (size_t j = 0; j < m_param.get_m_queue_depth(); j++) {
				shared_ptr<Batch<ImageData>> bsp = make_shared<Batch<ImageData>>();
				bsp->set_m_p_data(new ImageData[2 * batch_size]);
				bsp->set_m_size(2 * batch_size);
				bsp->set_m_p_data_org(bsp->get_m_p_data());
				bsp->set_m_size_org(bsp->get_m_size());
				m_queues_free[i]->Push(bsp);
			}
		}
	}

	void MemoryCopyNode::Run() {
		Start();
	}

	void MemoryCopyNode::Finalize() {
		Stop();

		size_t engine_num = dynamic_cast<InferenceNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByType("Inference")[0])->get_m_param().get_m_gpu_engine_num();
		for (size_t j = 0; j < engine_num; j++) {
			m_queues_free[j]->Clear();
			m_queues_full[j]->Clear();
		}
		m_queues_free.clear();
		m_queues_full.clear();
	}

	void MemoryCopyNode::EntryMulti(size_t thread_id) {
		using queue_next = BlockingQueue<shared_ptr<Batch<MemoryData>>>;
		auto engines = dynamic_cast<InferenceNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByType("Inference")[0])->get_m_engines();
		size_t batch_size = dynamic_cast<InferenceNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByType("Inference")[0])->get_m_param().get_m_batch_size();
		auto input_bytes = engines[thread_id]->GetInputSize();
		auto single_input_bytes = input_bytes / batch_size;
		auto input_buff = new float[input_bytes];
		int gpu_num = 0;
		cudaGetDeviceCount(&gpu_num);
		cudaSetDevice(int(thread_id % gpu_num));
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		try {
			ScheduleNode* top = GetTopNode();
			auto q_free_prev = m_queues_free[thread_id];
			auto q_full_prev = m_queues_full[thread_id];
			auto q_free_next = (*static_cast<vector<shared_ptr<queue_next>>*>(top->GetFreeQueue()))[thread_id];
			auto q_full_next = (*static_cast<vector<shared_ptr<queue_next>>*>(top->GetFullQueue()))[thread_id];

			while (!MustStop(thread_id)) {
				auto log_start = common::ScheduleClock::now();
				auto batch_prev = q_full_prev->Pop();

				size_t size = batch_prev->get_m_size();
				auto pp = batch_prev->get_m_p_data();

				auto batch_next = q_free_next->Pop();
				assert(size <= batch_next->get_m_size_org());
				batch_next->set_m_size(size);
				batch_next->set_m_tp(batch_prev->get_m_tp());
				auto& p = batch_next->get_m_p_data();
				void* buff = batch_next->get_m_p_buff();
				/*if (size == 1) {
					p[0].set_m_p_sample(pp[0].get_m_p_sample());
					p[0].get_m_label() = pp[0].get_m_label();
				        auto log_mem_start = common::ScheduleClock::now();
					cudaMemcpy(static_cast<float*>(buff), pp[0].get_m_p_mat().get(), single_input_bytes, cudaMemcpyHostToDevice);
				        auto log_mem_end = common::ScheduleClock::now();
				        Logger::LogDuration(log_mem_end - log_mem_start, __FILE__, __FUNCTION__, "cudaMemcpy", "size", size);
				}
				else if (size < 3) {*/
				auto log_mem_start = common::ScheduleClock::now();
				for (size_t i = 0; i < size; i++) {
					p[i].set_m_p_sample(pp[i].get_m_p_sample());
					p[i].get_m_label() = pp[i].get_m_label();
					//cudaMemcpy(static_cast<float*>(buff) + i * single_input_bytes / sizeof(float), pp[i].get_m_p_mat().get(), single_input_bytes, cudaMemcpyHostToDevice);
					cudaMemcpyAsync(static_cast<float*>(buff) + i * single_input_bytes / sizeof(float), pp[i].get_m_p_mat().get(), single_input_bytes, cudaMemcpyHostToDevice, stream);
				}
				cudaStreamSynchronize(stream);
				auto log_mem_end = common::ScheduleClock::now();
				Logger::LogDuration(log_mem_end - log_mem_start, __FILE__, __FUNCTION__, "cudaMemcpy", "size", size);
				/*}
				else {
				        auto log_cpumem_start = common::ScheduleClock::now();
					for (size_t i = 0; i < size; i++) {
						p[i].set_m_p_sample(pp[i].get_m_p_sample());
						p[i].get_m_label() = pp[i].get_m_label();
						//memcpy(input_buff + i * single_input_bytes / sizeof(float), pp[i].get_m_p_mat().get(), single_input_bytes);
						cudaMemcpy(static_cast<float*>(buff) + i * single_input_bytes / sizeof(float), pp[i].get_m_p_mat().get(), single_input_bytes, cudaMemcpyHostToDevice);
					}
				        auto log_cpumem_end = common::ScheduleClock::now();
				        Logger::LogDuration(log_cpumem_end - log_cpumem_start, __FILE__, __FUNCTION__, "cudaMemcpy", "size", size);

				        //auto log_mem_start = common::ScheduleClock::now();
					//cudaMemcpy(static_cast<float*>(buff), input_buff, single_input_bytes * size, cudaMemcpyHostToDevice);
				        //auto log_mem_end = common::ScheduleClock::now();
				        //Logger::LogDuration(log_mem_end - log_mem_start, __FILE__, __FUNCTION__, "cudaMemcpy", "size", size);
				}*/

				q_full_next->Push(batch_next);
				auto log_end = common::ScheduleClock::now();
				q_free_prev->Push(batch_prev);
				Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch", "size", size);
			}
			if (input_buff) {
				delete input_buff;
				input_buff = nullptr;
			}
		}
		catch (exception&) {
			if (input_buff) {
				delete input_buff;
				input_buff = nullptr;
			}
		}
	}

}
