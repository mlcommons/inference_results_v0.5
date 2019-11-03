#include <unistd.h>

#include "../common/logging.h"
#include "../common/utils.h"
#include "graph.h"
#include "image_node.h"
#include "post_process_node.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	using Logger = common::Logger;

	Schedule* Schedule::m_p_schedule = nullptr;
	mutex Schedule::m_mutex;

	Schedule* Schedule::GetSchedule() {
		if (!m_p_schedule) {
			unique_lock<mutex> lock(m_mutex);
			if (!m_p_schedule) {
				m_p_schedule = new Schedule;
			}
			lock.unlock();
		}
		return m_p_schedule;
	}

	TestSettings& Schedule::GetInferenceSettings() {
		return m_settings.get_m_test_settings();
	}

	void Schedule::InitSchedule(string node_conf_path,
		TestSettings test_settings,
		string data_set,
		string data_dir,
		string cache_dir,
		string model_path,
		string profile_name,
		string backend,
		bool accuracy,
		vector<string> inputs,
		vector<string> outputs,
		vector<string> image_list,
		vector<vector<float>> label_list) {

		m_settings.set_m_test_settings(test_settings);
		m_settings.set_m_data_set(data_set);
		m_settings.set_m_data_dir(data_dir);
		m_settings.set_m_cache_dir(cache_dir);
		m_settings.set_m_model_path(model_path);
		m_settings.set_m_profile_name(profile_name);
		m_settings.set_m_backend(backend);
		m_settings.set_m_accuracy(accuracy);
		m_settings.get_m_inputs().swap(inputs);
		m_settings.get_m_outputs().swap(outputs);
		m_settings.get_m_image_list().swap(image_list);
		m_settings.get_m_label_list().swap(label_list);

		Logger::Log(m_settings);

		m_graph.ReadConfigFile(node_conf_path);
		m_graph.Optimize();
		m_graph.Init();
		m_graph.Run();
		mp::SetGraph(&m_graph);
	}

	void Schedule::DestroySchedule() {
		if (m_p_schedule) {
			unique_lock<mutex> lock(m_mutex);
			if (m_p_schedule) {
				delete m_p_schedule;
			}
			lock.unlock();
		}
	}

	void Schedule::InitMLPerf(ReportLatencyResultsCallbackPython report_latency_results_cb) {

		m_settings.set_m_report_latency_results(report_latency_results_cb);

		const char* name = "cScheduleSUT";
		size_t name_length = string(name).length();
		m_settings.get_m_sut() = mlperf::c::ConstructSUT(0, name, name_length,
			&mp::IssueQuery,
			&mp::FlushQuery,
			&mp::ReportLatencyResults
		);

		name = "cScheduleQSL";
		name_length = string(name).length();
		ImageNode* node = dynamic_cast<ImageNode*>(m_graph.GetNodesByTypeAll("Image")[0]);
		size_t total_count = node->get_m_param().get_m_total_count();
		size_t loadable_set_size = node->get_m_param().get_m_loadable_set_size();
		m_settings.set_m_qsl(mlperf::c::ConstructQSL(0, name, name_length,
			total_count, loadable_set_size,
			&mp::LoadQuerySamples,
			&mp::UnloadQuerySamples
		));
	}

	void Schedule::FinalizeMLPerf() {
		mlperf::c::DestroyQSL(m_settings.get_m_qsl());
		mlperf::c::DestroySUT(m_settings.get_m_sut());
//		m_graph.Stop();
	}

	void Schedule::StartTest() {
		mlperf::c::StartTest(m_settings.get_m_sut(), m_settings.get_m_qsl(), m_settings.get_m_test_settings());
	}

	map<string, size_t> Schedule::UploadResults() {
		vector<shared_ptr<ScheduleNode>> nodes = m_graph.get_m_nodes();
		auto p_post = (dynamic_cast<PostProcessNode*>(m_graph.GetNodesByType("PostProcess")[0]))->get_m_p_post();
		return p_post->UploadResults();
	}

	vector<vector<vector<vector<float>>>> Schedule::UploadResultsCoco() {
		auto p_post = (dynamic_cast<PostProcessNode*>(m_graph.GetNodesByType("PostProcess")[0]))->get_m_p_post();
		return p_post->UploadResultsCoco();
	}

	double Schedule::GetLastLoad() {
		auto ds = (dynamic_cast<ImageNode*>(m_graph.GetNodesByTypeAll("Image")[0]))->get_m_data_set();
		auto time = ds.get_m_last_loaded().time_since_epoch().count();
		double time_s = (double)time / 1000000000;
		return time_s;
	}

}
