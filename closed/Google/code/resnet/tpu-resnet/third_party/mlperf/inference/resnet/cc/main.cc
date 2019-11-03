#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "mlperf/inference/loadgen/loadgen.h"
#include "mlperf/inference/loadgen/test_settings.h"
#include "mlperf/inference/runner/options.h"
#include "mlperf/inference/runner/query_sample_library_tpu.h"
#include "mlperf/inference/runner/runner.h"
#include "mlperf/inference/runner/system_under_test_tpu.h"

using int64 = long long int;

ABSL_FLAG(std::string, tpu_name, "", "The name of the TPU instance.");
ABSL_FLAG(int, num_worker_threads, std::thread::hardware_concurrency() * 2,
          "Number of threads to use.");
ABSL_FLAG(std::string, export_model_path, "",
          "The directory that includes the frozen model.");
ABSL_FLAG(std::string, preprocessed_dataset_path, "",
          "The directory that includes the preprocessed dataset.");
// Loadgen test settings.
ABSL_FLAG(bool, accuracy_mode, false, "Running in accuracy model.");
ABSL_FLAG(bool, init_tpu, true, "Initialize TPU.");
ABSL_FLAG(std::string, scenario, std::string("Server"), "Test scenario.");
ABSL_FLAG(int, total_sample_count, 50000, "Number of samples to test.");
ABSL_FLAG(int, performance_sample_count, 3072, "Number of samples to test.");
ABSL_FLAG(int, qps, 18000, "Target qps.");
ABSL_FLAG(int, time, 60, "Time to run in seconds.");
ABSL_FLAG(int, max_latency, 15, "Latency target in milliseconds.");
ABSL_FLAG(std::string, outdir, "/tmp", "Output directory for Loadgen.");
ABSL_FLAG(int64, qsl_rng_seed, 0x2b7e151628aed2a6L, "QSL rng seed.");
ABSL_FLAG(int64, sample_index_rng_seed, 0x093c467e37db0c7aL,
          "Sample index rng seed.");
ABSL_FLAG(int64, schedule_rng_seed, 0x3243f6a8885a308dL,
          "Schedule rng seed.");
// Batching settings.
ABSL_FLAG(std::string, batch_size, std::string("16"),
          "comma-separated list that specifies allowed batch sizes.");
ABSL_FLAG(std::string, batching_thread_pool_name,
          std::string("shared_batch_scheduler"),
          "The name of the thread pool for batching.");
ABSL_FLAG(int, batching_num_batch_threads, 16,
          "The number of thread for batching.");
ABSL_FLAG(int, batching_batch_timeout_micros, 2000,
          "The timeout in microseconds for batching.");
ABSL_FLAG(int, batching_max_enqueued_batches, 1 << 21,
          "The maximum batches in the queue.");
// Performance settings.
ABSL_FLAG(int, space_to_depth_block_size, 0, "conv0 space-to-depth block size");

namespace {

tensorflow::Status ParseFlagBatchSize(absl::string_view text,
                                      std::vector<int>& batch_size) {
  for (absl::string_view part :
     absl::StrSplit(text, ',', absl::SkipEmpty())) {
    // Let flag module parse the element type for us.
    int element;
    std::string error;
    if (!absl::ParseFlag(std::string(part), &element, &error)) {
      return tensorflow::Status(tensorflow::errors::InvalidArgument("Invalid"));
    }
    batch_size.emplace_back(element);
  }
  return tensorflow::Status::OK();
}

std::unique_ptr<mlperf::QuerySampleLibraryTpu> ConstructQsl(
    const mlperf::Option& option, mlperf::Runner* runner) {
  const int kChannelDim = 3;
  const int kImageSize = 224;
  int kSpaceToDepth = absl::GetFlag(FLAGS_space_to_depth_block_size);
  int channel, height, width;
  if (kSpaceToDepth == 0) {
    // When space-to-depth block size is zero, the packed channel dimension
    // has paddings. See imagenet.py and packing_utils.py.
    channel = 1;
    height = width = kImageSize;
  } else {
    channel = kChannelDim;
    height = width = kImageSize / kSpaceToDepth;
  }
  std::vector<int64> shape({channel, height, width});
  std::unique_ptr<mlperf::QuerySampleLibraryTpu> qsl =
      absl::make_unique<mlperf::QuerySampleLibraryTpu>(
          "qsl", runner, absl::GetFlag(FLAGS_preprocessed_dataset_path), shape,
          tensorflow::DT_INT32, option.total_sample_count,
          std::min(option.total_sample_count, option.performance_sample_count));
  return qsl;
}

void RunInference(const mlperf::Option& option) {
  // Set up runner.
  std::unique_ptr<mlperf::Runner> runner = absl::make_unique<mlperf::Runner>(
      option.batching_option, option.num_worker_threads,
      /*export_model_path*/ option.export_model_path,
      /*tpu_targets*/ option.tpu_target, /*standalone*/ false, option.init_tpu);

  // Offline mode: use the max batch size; server mode: use 1.
  // TODO(chiachenc): change this logic when server coelescing is enbaled.
  int issue_batch_size = 1;
  if (option.scenario == mlperf::TestScenario::Offline) {
    issue_batch_size =
        *std::max_element(option.batching_option.batch_size.begin(),
                          option.batching_option.batch_size.end());
  }
  LOG(INFO) << "issue_batch_size: " << issue_batch_size << std::endl;
  // Set up sut.
  std::unique_ptr<mlperf::SystemUnderTestTpu> sut =
      absl::make_unique<mlperf::SystemUnderTestTpu>("sut", issue_batch_size,
                                                    runner.get());
  // Set up qsl.
  std::unique_ptr<mlperf::QuerySampleLibraryTpu> qsl =
      ConstructQsl(option, runner.get());

  // Set up the loadgen.
  mlperf::TestSettings requested_settings;
  requested_settings.scenario = option.scenario;
  requested_settings.qsl_rng_seed = absl::GetFlag(FLAGS_qsl_rng_seed);
  requested_settings.schedule_rng_seed = absl::GetFlag(FLAGS_schedule_rng_seed);
  requested_settings.sample_index_rng_seed =
      absl::GetFlag(FLAGS_sample_index_rng_seed);
  int64 qps = option.qps;
  auto time = std::chrono::seconds(option.time);

  requested_settings.min_duration_ms = 60 * std::milli::den;
  requested_settings.max_duration_ms = 0;
  requested_settings.min_query_count = qps * time.count();
  requested_settings.max_query_count = 0;

  requested_settings.server_target_qps = qps;
  requested_settings.offline_expected_qps = qps;
  requested_settings.server_target_latency_ns =
      std::chrono::milliseconds(option.max_latency) /
      std::chrono::nanoseconds(1);
  if (absl::GetFlag(FLAGS_accuracy_mode)) {
    requested_settings.mode = mlperf::TestMode::AccuracyOnly;
  } else {
    requested_settings.mode = mlperf::TestMode::PerformanceOnly;
  }

  mlperf::LogSettings log_settings;
  log_settings.log_output.outdir = option.outdir;
  log_settings.log_output.copy_detail_to_stdout = true;
  log_settings.log_output.copy_summary_to_stdout = true;
  log_settings.enable_trace = false;

  // Warm up the system.
  qsl->LoadSamplesToRam({0});
  runner->WarmUp();

  // After warmup, give the system a moment to quiesce before putting it under
  // load.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Start test.
  mlperf::StartTest(sut.get(), qsl.get(), requested_settings, log_settings);
  runner->Finish();
}
}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::vector<int> batch_size;
  TF_CHECK_OK(ParseFlagBatchSize(absl::GetFlag(FLAGS_batch_size), batch_size));
  mlperf::BatchingOption batching_option(
      absl::GetFlag(FLAGS_batching_thread_pool_name),
      absl::GetFlag(FLAGS_batching_num_batch_threads),
      absl::GetFlag(FLAGS_batching_batch_timeout_micros),
      absl::GetFlag(FLAGS_batching_max_enqueued_batches), batch_size);
  mlperf::Option option(
      absl::GetFlag(FLAGS_num_worker_threads), batching_option,
      absl::GetFlag(FLAGS_total_sample_count),
      absl::GetFlag(FLAGS_performance_sample_count), absl::GetFlag(FLAGS_qps),
      absl::GetFlag(FLAGS_time), absl::GetFlag(FLAGS_max_latency),
      absl::GetFlag(FLAGS_outdir), absl::GetFlag(FLAGS_export_model_path),
      absl::GetFlag(FLAGS_tpu_name), absl::GetFlag(FLAGS_scenario),
      absl::GetFlag(FLAGS_init_tpu));

  mkdir(option.outdir.c_str(), 0777);
  RunInference(option);
  return 0;
}
