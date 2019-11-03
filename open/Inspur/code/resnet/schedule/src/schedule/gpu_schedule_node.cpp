#include "../common/logging.h"
#include "../config/input_file.h"
#include "image_node.h"
#include "gpu_schedule_node.h"


namespace schedule {

	using Logger = common::Logger;

	shared_ptr<GpuScheduleNode> GpuScheduleNodeParser::Run(Node node) {
		std::string line = m_pfile->ReadLine();
		if (line != "gpu_schedule_param {")
			throw "read gpu_schedule_param fail";
		GpuScheduleParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<GpuScheduleNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	GpuScheduleParam GpuScheduleNodeParser::ParseParam() {
		size_t thread_num = 10;
		size_t queue_depth = 100;
		size_t batch_size = 16;

		std::map<std::string, size_t*> key_value;
		key_value["thread_num:"] = &thread_num;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return GpuScheduleParam(thread_num, queue_depth, batch_size);
	}

	void GpuScheduleNode::Init() {
	}

	void GpuScheduleNode::Run() {
		Start();
	}

	void GpuScheduleNode::Finalize() {
		Stop();
	}

	// There should be only one Gpu Schedule (1 thread).
	void GpuScheduleNode::EntryMulti(size_t thread_id) {
		using queue_prev = BlockingQueue<shared_ptr<Batch<QuerySample>>>;
		using queue_next = BlockingQueue<shared_ptr<Batch<ImageData>>>;
		auto& ds = dynamic_cast<ImageNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByTypeAll("Image")[0])->get_m_data_set();;
		try {
			ScheduleNode* bottom = GetBottomNode();
			ScheduleNode* top = GetTopNode();
			auto q_free_prev = static_cast<queue_prev*>(bottom->GetFreeQueue());
			auto q_full_prev = static_cast<queue_prev*>(bottom->GetFullQueue());
			auto qs_free_next = static_cast<vector<shared_ptr<queue_next>>*>(top->GetFreeQueue());
			auto qs_full_next = static_cast<vector<shared_ptr<queue_next>>*>(top->GetFullQueue());

			while (!MustStop(thread_id)) {
				auto log_start = common::ScheduleClock::now();
				auto batch_prev = q_full_prev->Pop();
				auto log_start_1 = common::ScheduleClock::now();
				Logger::LogDuration(log_start_1 - log_start, __FILE__, __FUNCTION__, "batch_prev full pop");

				size_t size = batch_prev->get_m_size();
				QuerySample* pp = batch_prev->get_m_p_data();
				bool has_send = false;
				while (!has_send) {
					for (size_t i = 0; i < qs_free_next->size(); i++) {
						auto q_free_next = (*qs_free_next)[i];
						auto q_full_next = (*qs_full_next)[i];
						size_t queue_depth = q_free_next->Size();
						// idle gpu
						if (queue_depth >= 2) {
							auto batch_next = q_free_next->Pop();
							batch_next->set_m_size(size);
							batch_next->set_m_tp(batch_prev->get_m_tp());
							auto& p = batch_next->get_m_p_data();
							for (size_t j = 0; j < size; j++) {
								p[j].set_m_p_sample(pp + j);
								p[j].set_m_p_mat(ds.GetSampleInput(pp[j].index));
								p[j].get_m_label() = ds.GetSampleLable(pp[j].index);
							}
							q_full_next->Push(batch_next);
							auto log_end = common::ScheduleClock::now();
							q_free_prev->Push(batch_prev);
							has_send = true;
							Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch", "size", size);
							break;
						}
					}
					if (!has_send) {
						for (size_t i = 0; i < qs_free_next->size(); i++) {
							auto q_free_next = (*qs_free_next)[i];
							auto q_full_next = (*qs_full_next)[i];
							size_t queue_depth = q_free_next->Size();
							// gpu has vacant buffer
							if (queue_depth >= 1) {
								auto batch_next = q_free_next->Pop();
								batch_next->set_m_size(size);
								batch_next->set_m_tp(batch_prev->get_m_tp());
								for (size_t j = 0; j < size; j++) {
									auto& p = batch_next->get_m_p_data();
									p[j].set_m_p_sample(pp + j);
									p[j].set_m_p_mat(ds.GetSampleInput(pp[j].index));
									p[j].get_m_label() = ds.GetSampleLable(pp[j].index);
								}
								q_full_next->Push(batch_next);
								auto log_end = common::ScheduleClock::now();
								q_free_prev->Push(batch_prev);
								has_send = true;
								Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch", "size", size);
								break;
							}
						}
					}
				}
			}
		}
		catch (exception&) {
		}
	}

}
