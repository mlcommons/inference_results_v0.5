
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <dlfcn.h>
#include <iostream>

#include "loadgen.h"
#include "test_settings.h"
#include "GNMTSUT.h"
#include "GNMTQSL.h"

using mlperf::TestSettings;
using mlperf::LogSettings;
using mlperf::TestScenario;
using mlperf::TestMode;

using mlperf::QuerySample;
using mlperf::QuerySampleResponse;
using mlperf::QuerySampleLatency;

using mlperf::StartTest;

// General harness arguments
DEFINE_string(engines, "", "Comma-separated list of engines");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");

// GNMT specific arguments
DEFINE_int32(concurrency, 1, "Number of concurrency execution streams");
DEFINE_int32(timeout, 5000, "Batching Timeout in microseconds");
DEFINE_int32(min_batch_size, 1, "Minimum batch_size to benchmark for batcher cutoffs");
DEFINE_int32(batch_size_inc, 1, "batch_size increment benchmark for batcher cutoffs");
DEFINE_string(input_file, "newstest2014.tok.bpe.32000.en", "Path to input file. Default file 'newstest2014.tok.bpe.32000.en' is provided with GNMT");
DEFINE_bool(homogeneous_cluster, false, "Flag suggesting that all gpus in the cluster are similar");

// Loadgen test settings
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

std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t next = 0;
    while(next != std::string::npos)
    {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

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
    const std::string gSampleName = "GNMT_SERVER_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);


    // Creating settings for server mode
    TestSettings testSettings;
    testSettings.scenario = mlperf::TestScenario::Server;
    testSettings.mode = testModeMap[FLAGS_test_mode];
    testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, "Server");
    testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, "Server");
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

    // The target latency bound (converted to microseconds).
    std::chrono::microseconds latency_bound(testSettings.server_target_latency_ns/1000);

    // Initialize TensorRT plugins.
    nvinfer1::ILogger* trt_logger = getLogger();
    initLibNvInferPlugins(trt_logger, "");

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            LOG(FATAL) << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    std::vector<std::string> engine_dirs = splitString(FLAGS_engines, ",");

    // create sut
    int num_gpu;
    cudaGetDeviceCount(&num_gpu);
    LOG(INFO) << "Found " << num_gpu << " GPUs";
    auto sut = std::make_shared<Server>("GNMT SUT", latency_bound, num_gpu, FLAGS_homogeneous_cluster);

    // create qsl
    auto qsl = std::make_shared<SampleLibrary>("GNMT QSL", FLAGS_input_file);

    for(int device_id = 0; device_id < num_gpu; device_id ++)
    {
        sut->SingleGPUExecution(device_id, FLAGS_concurrency, engine_dirs, qsl, FLAGS_timeout);
        LOG(INFO) << "Launching device " << device_id << "...";
    }

    StartTest(sut.get(), qsl.get(), testSettings, logSettings);
    sut->running = false;

    return 0;
}
