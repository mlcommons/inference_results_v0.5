#include <set>

#include "../common/logging.h"
#include "../config/input_file.h"
#include "../postprocess/postproc_common.h"
#include "../postprocess/postproc_argmax.h"
#include "../postprocess/postproc_coco.h"
#include "post_process_node.h"


namespace schedule {

	using Logger = common::Logger;

	shared_ptr<PostProcessNode> PostProcessNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "post_process_param {")
			throw "read post_process_param fail";
		PostProcessParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<PostProcessNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	PostProcessParam PostProcessNodeParser::ParseParam() {
		size_t thread_num = 1;
		size_t queue_depth = 100;
		size_t batch_size = 32;

		map<string, size_t*> key_value;
		key_value["thread_num:"] = &thread_num;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return PostProcessParam(thread_num, queue_depth, batch_size);
	}

	void PostProcessNode::Init() {
		auto batch_size = m_param.get_m_batch_size();
		string ds = Schedule::GetSchedule()->get_m_settings().get_m_data_set();
		using BQResnet = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultImageNet>>>;
		using BQCoco = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultCoco>>>;
		if (ds == "imagenet" || ds == "imagenet_mobilenet") {
			m_p_queue_free = new BQResnet;
			m_p_queue_full = new BQResnet;
			m_p_post = make_shared<postprocess::PostProcCommon<float>>(0);
			for (size_t i = 0; i < m_param.get_m_queue_depth(); i++) {
				auto bsp = make_shared<Batch<postprocess::PredictResultImageNet>>();
				bsp->set_m_p_data(new postprocess::PredictResultImageNet[2 * batch_size]);
				bsp->set_m_size(2 * batch_size);
				bsp->set_m_p_data_org(bsp->get_m_p_data());
				bsp->set_m_size_org(bsp->get_m_size());
				((BQResnet*)m_p_queue_free)->Push(bsp);
			}
		}
		else if (ds == "coco-300") {
			m_p_queue_free = new BQCoco;
			m_p_queue_full = new BQCoco;
			m_p_post = make_shared<postprocess::PostProcessCoco<float>>();
			for (size_t i = 0; i < m_param.get_m_queue_depth(); i++) {
				auto bsp = make_shared<Batch<postprocess::PredictResultCoco>>();
				bsp->set_m_p_data(new postprocess::PredictResultCoco[2 * batch_size]);
				bsp->set_m_size(2 * batch_size);
				bsp->set_m_p_data_org(bsp->get_m_p_data());
				bsp->set_m_size_org(bsp->get_m_size());
				((BQCoco*)m_p_queue_free)->Push(bsp);
			}
		}
		else if (ds == "coco-1200-tf") {
			m_p_queue_free = new BQCoco;
			m_p_queue_full = new BQCoco;
			m_p_post = make_shared<postprocess::PostProcessCocoTf<float>>();
			for (size_t i = 0; i < m_param.get_m_queue_depth(); i++) {
				auto bsp = make_shared<Batch<postprocess::PredictResultCoco>>();
				bsp->set_m_p_data(new postprocess::PredictResultCoco[2 * batch_size]);
				bsp->set_m_size(2 * batch_size);
				bsp->set_m_p_data_org(bsp->get_m_p_data());
				bsp->set_m_size_org(bsp->get_m_size());
				((BQCoco*)m_p_queue_free)->Push(bsp);
			}
		}
		Logger::Log(__FILE__, __FUNCTION__, ds);
	}

	void PostProcessNode::Run() {
		Start();
	}

	void PostProcessNode::Finalize() {
		Stop();

		using BQImagenet = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultImageNet>>>;
		using BQCoco = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultCoco>>>;
		
		string ds = Schedule::GetSchedule()->get_m_settings().get_m_data_set();
		if (ds == "imagenet") {
			delete static_cast<BQImagenet*>(m_p_queue_free);
			delete static_cast<BQImagenet*>(m_p_queue_full);
		}
		else
		{
			delete static_cast<BQCoco*>(m_p_queue_free);
			delete static_cast<BQCoco*>(m_p_queue_full);
		}
	}

	struct SampleMetadata {
		void* query_metadata;
		uint64_t sequence_id;
		QuerySampleIndex sample_index;
		double accuracy_log_val;
	};

	void PostProcessNode::EntryMulti(size_t thread_id) {
		auto batch_size = m_param.get_m_batch_size();
		bool accuracy = Schedule::GetSchedule()->get_m_settings().get_m_accuracy();
		string ds = Schedule::GetSchedule()->get_m_settings().get_m_data_set();
		QuerySampleResponse* p_responses = new QuerySampleResponse[2 * batch_size];
		shared_ptr<QuerySampleResponse> sp(p_responses, QuerySampleResponseDeleter);
		try {
			while (!MustStop(thread_id)) {
				using BQResnet = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultImageNet>>>;
				using BQCoco = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultCoco>>>;
				static size_t count = 0;
				if (ds == "imagenet" || ds == "imagenet_mobilenet") {
					auto log_start = common::ScheduleClock::now();
					auto batch_prev = ((BQResnet*)m_p_queue_full)->Pop();
					count++;
					size_t size = batch_prev->get_m_size();
					auto pp = batch_prev->get_m_p_data();
					auto log_comm_start = common::ScheduleClock::now();
					auto processed_result = m_p_post->RunCommon(batch_prev);
					auto log_comm_end = common::ScheduleClock::now();
					Logger::LogDuration(log_comm_end - log_comm_start, __FILE__, __FUNCTION__, "RunCommon");

					auto log_res_start = common::ScheduleClock::now();
					if (accuracy)
						m_p_post->AddResults(processed_result);
					for (size_t i = 0; i < size; i++) {
						p_responses[i].id = pp[i].get_m_p_sample()->id;
						p_responses[i].data = (int64_t)& processed_result[i];
						p_responses[i].size = sizeof(int64_t);
					}
					auto log_end = common::ScheduleClock::now();
					Logger::LogDuration(log_end - log_res_start, __FILE__, __FUNCTION__, "AddResults");

					mlperf::c::QuerySamplesComplete(p_responses, size);
					auto log_query = common::ScheduleClock::now();
					Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one response", "size", size);
					Logger::LogDuration(log_end - mp::GetStart(), __FILE__, __FUNCTION__, "One round time");
					Logger::LogDuration(log_query - log_end, __FILE__, __FUNCTION__, "QuerySamplesComplete");
					((BQResnet*)m_p_queue_free)->Push(batch_prev);

					// To be deleted
					{
						set<ResponseId> respond_ids;
						for (size_t i = 0; i < size; i++) {
							auto s = reinterpret_cast<SampleMetadata*>(p_responses[i].id);
							//Logger::Log(__FILE__, __FUNCTION__, "accuracy_log_val", s->accuracy_log_val);
							//Logger::Log(__FILE__, __FUNCTION__, "query_metadata", s->query_metadata);
							//Logger::Log(__FILE__, __FUNCTION__, "sample_index", s->sample_index);
							//Logger::Log(__FILE__, __FUNCTION__, "pp[i].get_m_p_sample()->index", pp[i].get_m_p_sample()->index);
							//Logger::Log(__FILE__, __FUNCTION__, "sequence_id", s->sequence_id);
							assert(s->sample_index == pp[i].get_m_p_sample()->index);
							assert(p_responses[i].id = pp[i].get_m_p_sample()->id);
							respond_ids.emplace(p_responses[i].id);
						}
					}

				}
				else if (ds == "coco-300" || ds == "coco-1200-tf") {
					auto batch_prev = ((BQCoco*)m_p_queue_full)->Pop();
					auto log_start = common::ScheduleClock::now();
					count++;
					size_t size = batch_prev->get_m_size();
					auto pp = batch_prev->get_m_p_data();
					auto processed_result = m_p_post->RunCoco(batch_prev);
					if (accuracy)
						m_p_post->AddResultsCoco(processed_result);
					for (size_t i = 0; i < size; i++) {
						p_responses[i].id = pp[i].get_m_p_sample()->id;
						p_responses[i].data = (int64_t)& processed_result[i];
						p_responses[i].size = sizeof(int64_t);
					}
					auto log_end = common::ScheduleClock::now();
					mlperf::c::QuerySamplesComplete(p_responses, size);
					Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one response", "size", size);
					((BQCoco*)m_p_queue_free)->Push(batch_prev);

					// To be deleted
					{
						set<ResponseId> respond_ids;
						for (size_t i = 0; i < size; i++) {
							auto s = reinterpret_cast<SampleMetadata*>(p_responses[i].id);
							//Logger::Log(__FILE__, __FUNCTION__, "accuracy_log_val", s->accuracy_log_val);
							//Logger::Log(__FILE__, __FUNCTION__, "query_metadata", s->query_metadata);
							//Logger::Log(__FILE__, __FUNCTION__, "sample_index", s->sample_index);
							//Logger::Log(__FILE__, __FUNCTION__, "pp[i].get_m_p_sample()->index", pp[i].get_m_p_sample()->index);
							//Logger::Log(__FILE__, __FUNCTION__, "sequence_id", s->sequence_id);
							assert(s->sample_index == pp[i].get_m_p_sample()->index);
							assert(p_responses[i].id = pp[i].get_m_p_sample()->id);
							respond_ids.emplace(p_responses[i].id);
						}
					}

				}
			}
		}
		catch (exception&) {
		}
	}

}
