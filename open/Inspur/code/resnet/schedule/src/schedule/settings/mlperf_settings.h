#pragma once
#include <vector>
#include <iostream>
#include <functional>

#include "../../common/macro.h"
#include "../../common/types.h"
//#include "../../third_party/loadgen/bindings/c_api.h"
#include "../../loadgen/bindings/c_api.h"

using namespace std;


namespace schedule {

	using ClientData = mlperf::c::ClientData;
	using TestSettings = mlperf::TestSettings;
	using TestScenario = mlperf::TestScenario;
	using TestMode = mlperf::TestMode;
	using QuerySample = mlperf::QuerySample;
	using QuerySampleIndex = mlperf::QuerySampleIndex;
	using QuerySampleResponse = mlperf::QuerySampleResponse;
	using ResponseId = mlperf::ResponseId;
	using ReportLatencyResultsCallback = mlperf::c::ReportLatencyResultsCallback;
	using ReportLatencyResultsCallbackPython = function<void(vector<int64_t>)>;

	namespace mp {

		auto GetScenarioStr = [](TestScenario scenario) {
			switch (scenario) {
			case TestScenario::SingleStream:
				return "SingleStream";
			case TestScenario::MultiStream:
				return "MultiStream";
			case TestScenario::MultiStreamFree:
				return "MultiStreamFree";
			case TestScenario::Server:
				return "Server";
			case TestScenario::Offline:
				return "Offline";
			default:
				return "";
			}
		};

		auto GetModeStr = [](TestMode mode) {
			switch (mode) {
			case TestMode::SubmissionRun:
				return "SubmissionRun";
			case TestMode::AccuracyOnly:
				return "AccuracyOnly";
			case TestMode::PerformanceOnly:
				return "PerformanceOnly";
			case TestMode::FindPeakPerformance:
				return "FindPeakPerformance";
			default:
				return "";
			}
		};

		class MLPerfSettings {
			CREATE_SIMPLE_ATTR_SET_GET(m_test_settings, TestSettings)
			CREATE_SIMPLE_ATTR_SET_GET(m_data_set, string)
			CREATE_SIMPLE_ATTR_SET_GET(m_data_dir, string)
			CREATE_SIMPLE_ATTR_SET_GET(m_cache_dir, string)
			CREATE_SIMPLE_ATTR_SET_GET(m_model_path, string)
			CREATE_SIMPLE_ATTR_SET_GET(m_profile_name, string)
			CREATE_SIMPLE_ATTR_SET_GET(m_inputs, vector<string>)
			CREATE_SIMPLE_ATTR_SET_GET(m_outputs, vector<string>)
			CREATE_SIMPLE_ATTR_SET_GET(m_image_list, vector<string>)
			CREATE_SIMPLE_ATTR_SET_GET(m_label_list, vector<vector<float>>)
			CREATE_SIMPLE_ATTR_SET_GET(m_backend, string)
			CREATE_SIMPLE_ATTR_SET_GET(m_accuracy, bool)
			CREATE_SIMPLE_ATTR_SET_GET(m_sut, void*)
			CREATE_SIMPLE_ATTR_SET_GET(m_qsl, void*)
			CREATE_SIMPLE_ATTR_SET_GET(m_report_latency_results, ReportLatencyResultsCallbackPython)

		public:
			MLPerfSettings() {
				m_accuracy = false;
				m_sut = nullptr;
				m_qsl = nullptr;
				m_report_latency_results = nullptr;
			}

			~MLPerfSettings() {
			}

			friend ostream& operator<<(ostream& stream, const MLPerfSettings& ms) {
				stream << "m_data_set: " << ms.m_data_set << endl;
				stream << "m_data_dir: " << ms.m_data_dir << endl;
				stream << "m_cache_dir: " << ms.m_cache_dir << endl;
				stream << "m_model_path: " << ms.m_model_path << endl;
				stream << "m_profile_name: " << ms.m_profile_name << endl;
				stream << "m_inputs[0]: " << ms.m_inputs[0] << endl;
				stream << "m_outputs[0]: " << ms.m_outputs[0] << endl;
				stream << "m_image_list.size(): " << ms.m_image_list.size() << endl;
				stream << "m_label_list.size(): " << ms.m_label_list.size() << endl;
				stream << "m_label_list[0].size(): " << ms.m_label_list[0].size() << endl;
				stream << "m_backend: " << ms.m_backend << endl;
				stream << "m_accuracy: " << ms.m_accuracy << endl;
				stream << "m_sut: " << ms.m_sut << endl;
				stream << "m_qsl: " << ms.m_qsl << endl;
				stream << "ms.m_test_settings.scenario: " << GetScenarioStr(ms.m_test_settings.scenario) << endl;
				stream << "ms.m_test_settings.mode: " << GetModeStr(ms.m_test_settings.mode) << endl;
				stream << "ms.m_test_settings.min_duration_ms: " << ms.m_test_settings.min_duration_ms << endl;
				stream << "ms.m_test_settings.max_duration_ms: " << ms.m_test_settings.max_duration_ms << endl;
				stream << "ms.m_test_settings.min_query_count: " << ms.m_test_settings.min_query_count << endl;
				stream << "ms.m_test_settings.max_query_count: " << ms.m_test_settings.max_query_count << endl;
				stream << "ms.m_test_settings.qsl_rng_seed: " << ms.m_test_settings.qsl_rng_seed << endl;
				stream << "ms.m_test_settings.sample_index_rng_seed: " << ms.m_test_settings.sample_index_rng_seed << endl;
				stream << "ms.m_test_settings.schedule_rng_seed: " << ms.m_test_settings.schedule_rng_seed << endl;
				return stream;
			}
		};

		void SetGraph(void* p_graph);
		common::ScheduleClock::time_point GetStart();
		void ReportLatencyResults(ClientData client, const int64_t* data, size_t size);
		void IssueQuery(ClientData client, const QuerySample* samples, size_t size);
		void FlushQuery();
		void LoadQuerySamples(ClientData client, const QuerySampleIndex* samples, size_t size);
		void UnloadQuerySamples(ClientData client, const QuerySampleIndex* samples, size_t size);
	}

}