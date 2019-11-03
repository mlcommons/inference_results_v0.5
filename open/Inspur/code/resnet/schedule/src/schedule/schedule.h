# pragma once
#include <mutex>
#include <functional>

#include "../common/macro.h"
#include "../config/config.h"
#include "../config/parser.h"
#include "../config/node.h"
#include "../config/input_file.h"
#include "settings/mlperf_settings.h"
#include "graph.h"

using namespace std;


namespace schedule {

	class Schedule {
		CREATE_SIMPLE_ATTR_SET_GET(m_graph, Graph)
		CREATE_SIMPLE_ATTR_SET_GET(m_settings, mp::MLPerfSettings)

	public:
		Schedule() {};
		Schedule(Schedule&) = delete;
		Schedule(Schedule&&) = delete;
		Schedule& operator=(Schedule&) = delete;
		Schedule& operator=(Schedule&&) = delete;
		~Schedule() {};

		static Schedule* GetSchedule();
		static void DestroySchedule();

		void InitSchedule(string node_conf_path,
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
			vector<vector<float>> label_list);

		TestSettings& GetInferenceSettings();

		void InitMLPerf(ReportLatencyResultsCallbackPython report_latency_results_cb);

		void StartTest();

		void FinalizeMLPerf();

		map<string, size_t> UploadResults();

		vector<vector<vector<vector<float>>>> UploadResultsCoco();

		double GetLastLoad();

	private:
		static Schedule* m_p_schedule;
		static mutex m_mutex;
	};

}