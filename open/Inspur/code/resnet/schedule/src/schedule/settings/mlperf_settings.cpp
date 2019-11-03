#include <memory>

#include "../../common/logging.h"
#include "../graph.h"
#include "../init_node.h"
#include "../image_node.h"
#include "mlperf_settings.h"
#include "../schedule.h"


namespace schedule {

	namespace mp {

		using Logger = common::Logger;

		schedule::Graph* m_p_graph = nullptr;
		InitNode* m_p_node = nullptr;
		BlockingQueue<shared_ptr<Batch<QuerySample>>>* m_p_queue_free;
		BlockingQueue<shared_ptr<Batch<QuerySample>>>* m_p_queue_full;
		auto m_start = common::ScheduleClock::now();

		void SetGraph(void* p_graph) {
			m_p_graph = (schedule::Graph*)p_graph;
			m_p_node = dynamic_cast<InitNode*>(m_p_graph->GetNodesByType("Init")[0]);
			m_p_queue_free = static_cast<BlockingQueue<shared_ptr<Batch<QuerySample>>>*>(m_p_node->GetFreeQueue());
			m_p_queue_full = static_cast<BlockingQueue<shared_ptr<Batch<QuerySample>>>*>(m_p_node->GetFullQueue());
		}

		common::ScheduleClock::time_point GetStart() {
			return m_start;
		}

		void ReportLatencyResults(ClientData client, const int64_t* data, size_t size) {
			vector<int64_t> results;
			for (size_t i = 0; i < size; i++) {
				results.push_back(data[i]);
			}
			schedule::Schedule::GetSchedule()->get_m_settings().get_m_report_latency_results()(results);
		}

		void IssueQuery(ClientData client,
			const QuerySample* samples,
			size_t size) {
			if (m_p_graph) {
				m_start = common::ScheduleClock::now();
//				auto batch_size = m_p_node->get_m_param().get_m_batch_size();
//				auto left = size;
				static size_t count = 0;
				count++;
				stringstream ss;
				for (size_t i = 0; i < size; i++) {
					ss << samples[i].index << ", ";
				}
				Logger::Log(__FILE__, __FUNCTION__, "Receive one batch, count", count, "size", size, "data", ss.str());

//				while (left > 0)
//				{
//					size_t this_batch_size = 0;
//					if (left >= batch_size) {
//						this_batch_size = batch_size;
//					}
//					else {
//						this_batch_size = left;
//					}

				shared_ptr<Batch<QuerySample>> batch_next = m_p_queue_free->Pop();
				auto log_start = common::ScheduleClock::now();

//				batch_next->set_m_p_data(const_cast<QuerySample*>(samples + size - left));
				batch_next->set_m_p_data(const_cast<QuerySample*>(samples));
//				batch_next->set_m_size(this_batch_size);
				batch_next->set_m_size(size);
				batch_next->set_m_tp(m_start);

				auto log_end = common::ScheduleClock::now();
				m_p_queue_full->Push(batch_next);
//				left -= this_batch_size;
//				Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch, size", this_batch_size);
				Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch, size", size);
//				}
			}
		}

		void FlushQuery() {

		}

		void LoadQuerySamples(ClientData client,
			const QuerySampleIndex* samples,
			size_t size) {

			ImageNode* node = dynamic_cast<ImageNode*>(m_p_graph->GetNodesByTypeAll("Image")[0]);
			node->get_m_data_set().LoadQuerySamples(samples, size);
		}

		void UnloadQuerySamples(ClientData client,
			const QuerySampleIndex* samples,
			size_t size) {

			ImageNode* node = dynamic_cast<ImageNode*>(m_p_graph->GetNodesByTypeAll("Image")[0]);
			node->get_m_data_set().UnloadQuerySamples(samples, size);
		}

	}

}