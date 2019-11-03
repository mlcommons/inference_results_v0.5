#include "../common/logging.h"
#include "../config/input_file.h"
#include "data_set.h"
#include "inference_node.h"
#include "batch_split_node.h"


namespace schedule {

	using Logger = common::Logger;

	shared_ptr<BatchSplitNode> BatchSplitNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "batch_split_param {")
			throw "read batch_split_param fail";
		BatchSplitParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<BatchSplitNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	BatchSplitParam BatchSplitNodeParser::ParseParam() {
		size_t thread_num = 1;
		size_t queue_depth = 100;
		size_t batch_size = 32;

		map<string, size_t*> key_value;
		key_value["thread_num:"] = &thread_num;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return BatchSplitParam(thread_num, queue_depth, batch_size);
	}

	void BatchSplitNode::Init() {
		for (size_t i = 0; i < m_param.get_m_queue_depth(); i++) {
			shared_ptr<Batch<QuerySample>> bsp = make_shared<Batch<QuerySample>>();
			m_queue_free.Push(bsp);
		}
	}

	void BatchSplitNode::Run() {
		Start();
	}

	void BatchSplitNode::Finalize() {
		Stop();
		m_queue_free.Clear();
		m_queue_full.Clear();
	}

	void BatchSplitNode::EntryMulti(size_t thread_id) {
		using queue_prev = BlockingQueue<shared_ptr<Batch<QuerySample>>>;
		try {
			ScheduleNode* bottom = GetBottomNode();
			//size_t gpu_engine_num = dynamic_cast<InferenceNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByType("Inference")[0])->get_m_param().get_m_gpu_engine_num();
			size_t batch_size = m_param.get_m_batch_size();
			queue_prev* queue_free_prev = static_cast<queue_prev*>(bottom->GetFreeQueue());
			queue_prev* queue_full_prev = static_cast<queue_prev*>(bottom->GetFullQueue());
			while (!MustStop(thread_id)) {
				auto log_start = common::ScheduleClock::now();
				shared_ptr<Batch<QuerySample>> batch_prev = queue_full_prev->Pop();
				auto log_full_pop = common::ScheduleClock::now();
				Logger::LogDuration(log_full_pop - log_start, __FILE__, __FUNCTION__, "batch_prev full pop");

				size_t size = batch_prev->get_m_size();
				QuerySample* pp = batch_prev->get_m_p_data();
				size_t left = size;
				//size_t dev = size / gpu_engine_num;
				//size_t remainder = size % gpu_engine_num;
				//size_t batch_size_next = 0;
				//if (remainder > 0) {
				//	batch_size_next = batch_size <= dev + 1 ? batch_size : dev + 1;
				//}
				//else {
				//	batch_size_next = batch_size <= dev ? batch_size : dev;
				//}

				while (left > 0) {
				  auto log_free_pop = common::ScheduleClock::now();
					shared_ptr<Batch<QuerySample>> batch_next = m_queue_free.Pop();
				  auto log_free_pop_end = common::ScheduleClock::now();
				  Logger::LogDuration(log_free_pop_end - log_free_pop, __FILE__, __FUNCTION__, "batch_next queue free pop");

					size_t this_batch_size = 0;
					if (left >= batch_size) {
						this_batch_size = batch_size;
					}
					else {
						this_batch_size = left;
					}

					batch_next->set_m_size(this_batch_size);
					batch_next->set_m_p_data(pp);
					batch_next->set_m_tp(batch_prev->get_m_tp());

				  auto log_full_push = common::ScheduleClock::now();
					m_queue_full.Push(batch_next);
				  auto log_full_push_end = common::ScheduleClock::now();
				  Logger::LogDuration(log_full_push_end - log_full_push, __FILE__, __FUNCTION__, "batch_next queue full push", "size", this_batch_size);


					pp += this_batch_size;
					left -= this_batch_size;
				}
				auto log_free_pop = common::ScheduleClock::now();
				queue_free_prev->Push(batch_prev);
				auto log_free_pop_end = common::ScheduleClock::now();
				Logger::LogDuration(log_free_pop_end - log_free_pop, __FILE__, __FUNCTION__, "batch_prev queue free push");
				Logger::LogDuration(log_free_pop_end - log_start, __FILE__, __FUNCTION__, "Send one batch");
			}
		}
		catch (exception&) {
		}
	}

}
