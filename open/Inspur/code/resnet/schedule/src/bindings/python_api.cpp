/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
	http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/// \file
/// \brief Python bindings for the schedule using pybind11.
#ifndef SCHEDULE_PYTHON_BINDINGS_H
#define SCHEDULE_PYTHON_BINDINGS_H

//#include <map>
//#include <string>
//#include <vector>
#include <functional>

#include "../third_party/pybind/include/pybind11/functional.h"
#include "../third_party/pybind/include/pybind11/pybind11.h"
#include "../third_party/pybind/include/pybind11/stl.h"
#include "../third_party/pybind/include/pybind11/stl_bind.h"

#include "../loadgen/loadgen.h"
#include "../loadgen/query_sample.h"
#include "../loadgen/query_sample_library.h"
#include "../loadgen/system_under_test.h"
#include "../loadgen/test_settings.h"

#include "../schedule/schedule.h"


namespace shedule {

	namespace {

	}  // namespace

	/// \brief Python bindings.
	namespace py {

		using Schedule = schedule::Schedule;
		using ReportLatencyResultsCallback = schedule::ReportLatencyResultsCallback;
		using ReportLatencyResultsCallbackPython = schedule::ReportLatencyResultsCallbackPython;
		using TestScenario = mlperf::TestScenario;
		using TestMode = mlperf::TestMode;
		using TestSettings = mlperf::TestSettings;
		using QuerySample = mlperf::QuerySample;
		using QuerySampleResponse = mlperf::QuerySampleResponse;
		using QuerySampleIndex = mlperf::QuerySampleIndex;
		using ResponseId = mlperf::ResponseId;
		using LoggingMode = mlperf::LoggingMode;
		using LogSettings = mlperf::LogSettings;
		using LogOutputSettings = mlperf::LogOutputSettings;

		TestSettings GetInferenceSettings() {
			return Schedule::GetSchedule()->GetInferenceSettings();
		}

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
			vector<vector<float>> label_list) {

			Schedule::GetSchedule()->InitSchedule(node_conf_path,
				test_settings,
				data_set,
				data_dir,
				cache_dir,
				model_path,
				profile_name,
				backend,
				accuracy,
				inputs,
				outputs,
				image_list,
				label_list);
		}

		void DestroySchedule() {
			Schedule::GetSchedule()->DestroySchedule();
		}

		void InitMLPerf(ReportLatencyResultsCallbackPython report_latency_results_cb) {
			Schedule::GetSchedule()->InitMLPerf(report_latency_results_cb);
		}

		void FinalizeMLPerf() {
			Schedule::GetSchedule()->FinalizeMLPerf();
		}

		void StartTest() {
			Schedule::GetSchedule()->StartTest();
		}

		map<string, size_t> UploadResults() {
			return Schedule::GetSchedule()->UploadResults();
		}

		vector<vector<vector<vector<float>>>> UploadResultsCoco() {
			return Schedule::GetSchedule()->UploadResultsCoco();
		}

		double GetLastLoad() {
			return Schedule::GetSchedule()->GetLastLoad();
		}

		PYBIND11_MODULE(mlperf_schedule, m) {
			m.doc() = "MLPerf Schedule.";

			pybind11::enum_<TestScenario>(m, "TestScenario")
				.value("SingleStream", TestScenario::SingleStream)
				.value("MultiStream", TestScenario::MultiStream)
				.value("MultiStreamFree", TestScenario::MultiStreamFree)
				.value("Server", TestScenario::Server)
				.value("Offline", TestScenario::Offline);

			pybind11::enum_<TestMode>(m, "TestMode")
				.value("SubmissionRun", TestMode::SubmissionRun)
				.value("AccuracyOnly", TestMode::AccuracyOnly)
				.value("PerformanceOnly", TestMode::PerformanceOnly)
				.value("FindPeakPerformance", TestMode::FindPeakPerformance);

			pybind11::class_<TestSettings>(m, "TestSettings")
				.def(pybind11::init<>())
				.def_readwrite("scenario", &TestSettings::scenario)
				.def_readwrite("mode", &TestSettings::mode)
				.def_readwrite("single_stream_expected_latency_ns",
					&TestSettings::single_stream_expected_latency_ns)
				.def_readwrite("single_stream_target_latency_percentile",
					&TestSettings::single_stream_target_latency_percentile)
				.def_readwrite("multi_stream_target_qps",
					&TestSettings::multi_stream_target_qps)
				.def_readwrite("multi_stream_target_latency_ns",
					&TestSettings::multi_stream_target_latency_ns)
				.def_readwrite("multi_stream_target_latency_percentile",
					&TestSettings::multi_stream_target_latency_percentile)
				.def_readwrite("multi_stream_samples_per_query",
					&TestSettings::multi_stream_samples_per_query)
				.def_readwrite("multi_stream_max_async_queries",
					&TestSettings::multi_stream_max_async_queries)
				.def_readwrite("server_target_qps", &TestSettings::server_target_qps)
				.def_readwrite("server_target_latency_ns",
					&TestSettings::server_target_latency_ns)
				.def_readwrite("server_target_latency_percentile",
					&TestSettings::server_target_latency_percentile)
				.def_readwrite("server_coalesce_queries",
					&TestSettings::server_coalesce_queries)
				.def_readwrite("server_find_peak_qps_decimals_of_precision",
					&TestSettings::server_find_peak_qps_decimals_of_precision)
				.def_readwrite("server_find_peak_qps_boundary_step_size",
					&TestSettings::server_find_peak_qps_boundary_step_size)
				.def_readwrite("server_max_async_queries",
					&TestSettings::server_max_async_queries)
				.def_readwrite("offline_expected_qps",
					&TestSettings::offline_expected_qps)
				.def_readwrite("min_duration_ms", &TestSettings::min_duration_ms)
				.def_readwrite("max_duration_ms", &TestSettings::max_duration_ms)
				.def_readwrite("min_query_count", &TestSettings::min_query_count)
				.def_readwrite("max_query_count", &TestSettings::max_query_count)
				.def_readwrite("qsl_rng_seed", &TestSettings::qsl_rng_seed)
				.def_readwrite("sample_index_rng_seed",
					&TestSettings::sample_index_rng_seed)
				.def_readwrite("schedule_rng_seed", &TestSettings::schedule_rng_seed)
				.def_readwrite("accuracy_log_rng_seed",
					&TestSettings::accuracy_log_rng_seed)
				.def_readwrite("accuracy_log_probability",
					&TestSettings::accuracy_log_probability)
				.def_readwrite("print_timestamps", &TestSettings::print_timestamps)
				.def_readwrite("performance_issue_unique",
					&TestSettings::performance_issue_unique)
				.def_readwrite("performance_issue_same",
					&TestSettings::performance_issue_same)
				.def_readwrite("performance_issue_same_index",
					&TestSettings::performance_issue_same_index)
				.def_readwrite("performance_sample_count_override",
					&TestSettings::performance_sample_count_override)
				.def("FromConfig", &TestSettings::FromConfig, "FromConfig.");

			pybind11::enum_<LoggingMode>(m, "LoggingMode")
				.value("AsyncPoll", LoggingMode::AsyncPoll)
				.value("EndOfTestOnly", LoggingMode::EndOfTestOnly)
				.value("Synchronous", LoggingMode::Synchronous);

			pybind11::class_<LogOutputSettings>(m, "LogOutputSettings")
				.def(pybind11::init<>())
				.def_readwrite("outdir", &LogOutputSettings::outdir)
				.def_readwrite("prefix", &LogOutputSettings::prefix)
				.def_readwrite("suffix", &LogOutputSettings::suffix)
				.def_readwrite("prefix_with_datetime",
					&LogOutputSettings::prefix_with_datetime)
				.def_readwrite("copy_detail_to_stdout",
					&LogOutputSettings::copy_detail_to_stdout)
				.def_readwrite("copy_summary_to_stdout",
					&LogOutputSettings::copy_summary_to_stdout);

			pybind11::class_<LogSettings>(m, "LogSettings")
				.def(pybind11::init<>())
				.def_readwrite("log_output", &LogSettings::log_output)
				.def_readwrite("log_mode", &LogSettings::log_mode)
				.def_readwrite("log_mode_async_poll_interval_ms",
					&LogSettings::log_mode_async_poll_interval_ms)
				.def_readwrite("enable_trace", &LogSettings::enable_trace);

			pybind11::class_<QuerySample>(m, "QuerySample")
				.def(pybind11::init<>())
				.def(pybind11::init<ResponseId, QuerySampleIndex>())
				.def_readwrite("id", &QuerySample::id)
				.def_readwrite("index", &QuerySample::index);

			pybind11::class_<QuerySampleResponse>(m, "QuerySampleResponse")
				.def(pybind11::init<>())
				.def(pybind11::init<ResponseId, uintptr_t, size_t>())
				.def_readwrite("id", &QuerySampleResponse::id)
				.def_readwrite("data", &QuerySampleResponse::data)
				.def_readwrite("size", &QuerySampleResponse::size);

			// TODO: Use PYBIND11_MAKE_OPAQUE for the following vector types.
			pybind11::bind_vector<std::vector<QuerySample>>(m, "VectorQuerySample");
			pybind11::bind_vector<std::vector<QuerySampleResponse>>(m, "VectorQuerySampleResponse");

			m.def("GetInferenceSettings", &py::GetInferenceSettings, "Get MLPerf Inference settings struct.");
			m.def("InitSchedule", &py::InitSchedule, "Init MLPerf Inference Schedule.");
			m.def("DestroySchedule", &py::DestroySchedule, "Destroy MLPerf Inference Schedule.");
			m.def("InitMLPerf", &py::InitMLPerf, "Init MLPerf Inference Loadgen.");
			m.def("FinalizeMLPerf", &py::FinalizeMLPerf, "Stop MLPerf Inference Loadgen.");
			m.def("StartTest", &py::StartTest, "Stop MLPerf Inference Loadgen.");
			m.def("UploadResults", &py::UploadResults, "Upload results for finalize.");
			m.def("UploadResultsCoco", &py::UploadResultsCoco, "Upload Coco results for finalize.");
			m.def("GetLastLoad", &py::GetLastLoad, "Upload Coco results for finalize.");
		}

	}  // namespace py
}  // namespace schedule

#endif  // SCHEDULE_PYTHON_BINDINGS_H
