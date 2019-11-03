// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"

#include "common/logger.h"
#include "test_settings.h"
#include "loadgen.h"
#include "params.h"

#include "gnmt_server.h"

DEFINE_string(engine, "", "Engine");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "gnmt", "Model name");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(single_stream_expected_latency_ns, 100000, "Inverse of desired target QPS");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// gnmt specific arguments
DEFINE_string(input_file, "newstest2014.tok.bpe.32000.en", "Path to input file. Default file 'newstest2014.tok.bpe.32000.en' is provided with GNMT");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream}
};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}
};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous}
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "GNMT_OFFLINE_AND_SINGLE_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);

    // Scope to force all smart objects destruction before CUDA context resets
    {

        int num_gpu;
        cudaGetDeviceCount(&num_gpu);
        LOG(INFO) << "Found " << num_gpu << " GPUs";

        std::string jsonString = FLAGS_engine + "/" + "config.json";
        LOG(INFO) << jsonString;
        auto config = std::make_shared<Config>(jsonString);
        config->printConfig();

        std::vector<int> gpus;
        for(int i = 0; i < num_gpu; ++i)
            gpus.push_back(i);
        auto gnmt_server = std::make_shared<GNMTServer>(
            "GNMT SERVER",
            FLAGS_input_file,
            config,
            FLAGS_engine,
            gpus);

        // Configure the test settings
        mlperf::TestSettings testSettings;
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.mode = testModeMap[FLAGS_test_mode];
        testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;

        // Configure the logging settings
        mlperf::LogSettings logSettings;
        logSettings.log_output.outdir = FLAGS_logfile_outdir;
        logSettings.log_output.prefix = FLAGS_logfile_prefix;
        logSettings.log_output.suffix = FLAGS_logfile_suffix;
        logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
        logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
        logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
        logSettings.log_mode = logModeMap[FLAGS_log_mode];
        logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
        logSettings.enable_trace = FLAGS_log_enable_trace;

        StartTest(gnmt_server.get(), gnmt_server.get(), testSettings, logSettings);
    }

    return 0;
}
