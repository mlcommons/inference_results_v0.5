#include "../common/types.h"
#include "../common/logging.h"
#include "../config/input_file.h"
#include "batch_merge_node.h"


namespace schedule {

	using Logger = common::Logger;

	shared_ptr<BatchMergeNode> BatchMergeNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "batch_merge_param {")
			throw "read batch_merge_param fail";
		BatchMergeParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<BatchMergeNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	BatchMergeParam BatchMergeNodeParser::ParseParam() {
		size_t thread_num = 1;
		size_t queue_depth = 100;
		size_t batch_size = 32;
		size_t merge_time_threshold_ns = 1000000;

		map<string, size_t*> key_value;
		key_value["thread_num:"] = &thread_num;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		key_value["merge_time_threshold_ns:"] = &merge_time_threshold_ns;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return BatchMergeParam(thread_num, queue_depth, batch_size, merge_time_threshold_ns);
	}

	void BatchMergeNode::Init() {
		auto batch_size = m_param.get_m_batch_size();
		for (size_t i = 0; i < m_param.get_m_queue_depth(); i++) {
			shared_ptr<Batch<QuerySample>> bsp = make_shared<Batch<QuerySample>>();
			bsp->set_m_p_data(new QuerySample[2 * batch_size]);
			bsp->set_m_size(2 * batch_size);
			bsp->set_m_p_data_org(bsp->get_m_p_data());
			bsp->set_m_size_org(bsp->get_m_size());
			m_queue_free.Push(bsp);
		}
	}

	void BatchMergeNode::Run() {
		Start();
	}

	void BatchMergeNode::Finalize() {
		Stop();
		m_queue_free.Clear();
		m_queue_full.Clear();
	}

	void BatchMergeNode::EntryMulti(size_t thread_id) {
		using queue_prev = BlockingQueue<shared_ptr<Batch<QuerySample>>>;
		try {
			ScheduleNode* bottom = GetBottomNode();
			bool wait_flag = false;
			size_t accumulated = 0;
			vector<shared_ptr<Batch<QuerySample>>> batches;
			size_t batch_size = m_param.get_m_batch_size();
			int64_t threshold = static_cast<int64_t>(m_param.get_m_merge_time_threshold_ns());
			queue_prev* queues_free = static_cast<queue_prev*>(bottom->GetFreeQueue());
			queue_prev* queues_full = static_cast<queue_prev*>(bottom->GetFullQueue());
			shared_ptr<Batch<QuerySample>> batch_prev;
			common::ScheduleClock::time_point begin = common::ScheduleClock::now();
			common::ScheduleClock::time_point now;
			while (!MustStop(thread_id)) {
				bool has_item = queues_full->TryPop(&batch_prev);
				common::ScheduleClock::time_point log_start;
				common::ScheduleClock::time_point log_end;
				if (!has_item) {
					now = common::ScheduleClock::now();
					if (wait_flag && (now - begin).count() >= threshold) {
						wait_flag = false;
						accumulated = 0;
					}
					else
						continue;
				}
				else {
					size_t size = batch_prev->get_m_size();
					batches.push_back(batch_prev);

					if (!wait_flag) {
						log_start = common::ScheduleClock::now();
						begin = batch_prev->get_m_tp();
						wait_flag = true;
					}
					if (wait_flag) {
						accumulated += size;
					}
					now = common::ScheduleClock::now();
					// Meets the waiting time or accumulated samples number.
					if (wait_flag && ((now - begin).count() >= threshold || accumulated >= batch_size)) {
						wait_flag = false;
						accumulated = 0;
					}
				}

				if (!wait_flag && batches.size() > 0) {
					shared_ptr<Batch<QuerySample>> batch_next = m_queue_free.Pop();

					size_t b_size = 0;
					for (auto& b : batches) {
						b_size += b->get_m_size();
					}

					batch_next->set_m_size(b_size);
					batch_next->set_m_tp(batches[0]->get_m_tp());

					if (batches.size() == 1) {
						batch_next->set_m_p_data(batches[0]->get_m_p_data());
					}
					else {
						auto p_data_org = batch_next->get_m_p_data_org();
						auto p_data_org_end = p_data_org + batch_next->get_m_size_org();
						batch_next->set_m_p_data(p_data_org);
						QuerySample* p = batch_next->get_m_p_data();
						for (auto& bb : batches) {
							QuerySample* pp = bb->get_m_p_data();
							size_t ss = bb->get_m_size();
							memcpy(p, pp, sizeof(QuerySample) * ss);
							p += ss;
							assert(p < p_data_org_end);
						}
					}
					log_end = common::ScheduleClock::now();
					m_queue_full.Push(batch_next);
					Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch", "size", b_size);
					for (auto& bb : batches) {
						queues_free->Push(bb);
					}
					batches.clear();
				}
			}
		}
		catch (exception&) {
		}
	}

}