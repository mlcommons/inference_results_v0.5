#include "../common/logging.h"
#include "../common/types.h"
#include "../config/input_file.h"
#include "../inference/inference.h"
#include "post_process_node.h"
#include "inference_node.h"


namespace schedule {

	using Logger = common::Logger;

	shared_ptr<InferenceNode> InferenceNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "inference_param {")
			throw "read inference_param fail";
		InferenceParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<InferenceNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	InferenceParam InferenceNodeParser::ParseParam() {
		size_t queue_depth = 100;
		size_t batch_size = 32;
		size_t gpu_engine_num = 10;

		map<string, size_t*> key_value;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		key_value["gpu_engine_num:"] = &gpu_engine_num;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return InferenceParam(queue_depth, batch_size, gpu_engine_num);
	}

	void InferenceNode::Init() {
		mp::MLPerfSettings& settings = Schedule::GetSchedule()->get_m_settings();
		auto batch_size = m_param.get_m_batch_size();
		auto engine_num = m_param.get_m_gpu_engine_num();
		cout << "model_path: " << settings.get_m_model_path().data() << endl;
		m_engines = inference::GpuEngine::InitGpuEngines(
			settings.get_m_model_path(),
			settings.get_m_data_dir(),
			settings.get_m_profile_name(),
			(unsigned int)batch_size,
			settings.get_m_profile_name(),
			(int)engine_num);
		for (size_t i = 0; i < engine_num; i++) {
			m_queues_free.push_back(make_shared<BlockingQueue<shared_ptr<Batch<MemoryData>>>>());
			m_queues_full.push_back(make_shared<BlockingQueue<shared_ptr<Batch<MemoryData>>>>());
			for (size_t j = 0; j < m_param.get_m_queue_depth(); j++) {
				auto bsp = make_shared<Batch<MemoryData>>();
				bsp->set_m_p_data(new MemoryData[2 * batch_size]);
				bsp->set_m_size(2 * batch_size);
				bsp->set_m_p_data_org(bsp->get_m_p_data());
				bsp->set_m_size_org(bsp->get_m_size());
				void*& buff = bsp->get_m_p_buff();
				int gpu_num = 0;
				cudaGetDeviceCount(&gpu_num);
				cudaSetDevice(int(i % gpu_num));
				cudaMalloc(&buff, m_engines[i]->GetInputSize());
				m_queues_free[i]->Push(bsp);
			}
		}
	}

	void InferenceNode::Run() {
		Start();
	}

	void InferenceNode::Finalize() {
		Stop();

		for (size_t j = 0; j < m_param.get_m_gpu_engine_num(); j++) {
			m_queues_free[j]->Clear();
			m_queues_full[j]->Clear();
		}
		m_queues_free.clear();
		m_queues_full.clear();
	}

	string InferenceNode::GetDataFormat() {
		return "NCHW";
	}

	void InferenceNode::EntryMulti(size_t thread_id) {
		string ds = Schedule::GetSchedule()->get_m_settings().get_m_data_set();
		shared_ptr<GpuEngine> engine = m_engines[thread_id];
		auto batch_size = m_param.get_m_batch_size();
		int gpu_num = 0;
		cudaGetDeviceCount(&gpu_num);
		cudaSetDevice(int(thread_id % gpu_num));

		try {
			ScheduleNode* top = GetTopNode();
			using BQResnet = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultImageNet>>>;
			using BQCoco = BlockingQueue<shared_ptr<Batch<postprocess::PredictResultCoco>>>;
			auto q_free_prev = m_queues_free[thread_id];
			auto q_full_prev = m_queues_full[thread_id];
			while (!MustStop(thread_id)) {
				if (ds == "imagenet" || ds == "imagenet_mobilenet") {
					auto log_start = common::ScheduleClock::now();
					auto q_free_next = static_cast<BQResnet*>(top->GetFreeQueue());
					auto q_full_next = static_cast<BQResnet*>(top->GetFullQueue());
					auto batch_prev = q_full_prev->Pop();
					auto size = batch_prev->get_m_size();
					auto& pp = batch_prev->get_m_p_data();
					auto batch_next = q_free_next->Pop();
					auto log_pop = common::ScheduleClock::now();
					Logger::LogDuration(log_pop - log_start, __FILE__, __FUNCTION__, "pop");

					assert(size <= batch_next->get_m_size_org());
					assert(size <= batch_size);
					auto& p = batch_next->get_m_p_data();
					batch_next->set_m_size(size);
					batch_next->set_m_tp(batch_prev->get_m_tp());
					auto log_start3 = common::ScheduleClock::now();
					vector<ResultTensor>* p_result = engine->Predict(batch_prev);
					auto log_end3 = common::ScheduleClock::now();
					for (size_t i = 0; i < size; i++) {
						p[i].set_m_p_sample(pp[i].get_m_p_sample());
						p[i].set_m_result((float)(*p_result)[i].num);
						p[i].set_m_label(pp[i].get_m_label());
					}
					auto log_push = common::ScheduleClock::now();
					q_full_next->Push(batch_next);
					q_free_prev->Push(batch_prev);
					auto log_end = common::ScheduleClock::now();
					Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch", "size", size);
					Logger::LogDuration(log_end3 - log_start3, __FILE__, __FUNCTION__, "Predict");
					Logger::LogDuration(log_end - log_push, __FILE__, __FUNCTION__, "push");
				}
				else {
					auto log_start = common::ScheduleClock::now();
					auto q_free_next = static_cast<BQCoco*>(top->GetFreeQueue());
					auto q_full_next = static_cast<BQCoco*>(top->GetFullQueue());
					auto batch_prev = q_full_prev->Pop();
					auto size = batch_prev->get_m_size();
					auto& pp = batch_prev->get_m_p_data();
					auto batch_next = q_free_next->Pop();
					assert(size <= batch_next->get_m_size_org());
					assert(size <= batch_size);
					auto& p = batch_next->get_m_p_data();
					batch_next->set_m_size(size);
					batch_next->set_m_tp(batch_prev->get_m_tp());
					auto log_start3 = common::ScheduleClock::now();
					vector<ResultTensor>* p_result = engine->Predict(batch_prev);
					auto log_end3 = common::ScheduleClock::now();
					for (size_t i = 0; i < size; i++) {
						p[i].set_m_p_sample(pp[i].get_m_p_sample());
						p[i].set_m_result(&(*p_result)[i]);
						p[i].set_m_label(pp[i].get_m_label());
						Logger::Log(__FILE__, __FUNCTION__, "num", (*p_result)[i].num);
						for (int j = 0; j < (*p_result)[i].num; j++) {
							Logger::Log(__FILE__, __FUNCTION__, "detected_classes", ((*p_result)[i].detected_classes)[j]);
							Logger::Log(__FILE__, __FUNCTION__, "score", ((*p_result)[i].scores)[j]);
							Logger::Log(__FILE__, __FUNCTION__, "box0", ((*p_result)[i].boxes)[j * 4 + 0]);
							Logger::Log(__FILE__, __FUNCTION__, "box1", ((*p_result)[i].boxes)[j * 4 + 1]);
							Logger::Log(__FILE__, __FUNCTION__, "box2", ((*p_result)[i].boxes)[j * 4 + 2]);
							Logger::Log(__FILE__, __FUNCTION__, "box3", ((*p_result)[i].boxes)[j * 4 + 3]);
						}
					}
					q_full_next->Push(batch_next);
					q_free_prev->Push(batch_prev);
					auto log_end = common::ScheduleClock::now();
					Logger::LogDuration(log_end - log_start, __FILE__, __FUNCTION__, "Send one batch", "size", size);
					Logger::LogDuration(log_end3 - log_start3, __FILE__, __FUNCTION__, "Predict");
				}
			}
		}
		catch (exception&) {
		}
	}

}
