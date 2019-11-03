#include <string> 
#include <vector>

// loadgen integration
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <assert.h>

#include "multi_instance.h"

// loadgen integration
static size_t inference_samples = 0;
static size_t number_response_samples = 0;
std::chrono::time_point<std::chrono::high_resolution_clock> sort_start;
std::chrono::time_point<std::chrono::high_resolution_clock> sort_end;	
std::chrono::duration<double, std::milli> sort_time(0);

bool stop_process = false;
bool run_as_instance = false;
bool interactive = false;
bool offline_single_response = false;
// sleep for a while after check each instances' response
bool loadrun_sleep_after_each_response_check = true; 
// sort samples before dispatch
bool sort_samples = false;
// handle queries number indivisable by batch size
bool handle_flush_queries = true;
bool loadgen_test_end = false;

std::chrono::time_point<std::chrono::high_resolution_clock> query_complete_start;
std::chrono::time_point<std::chrono::high_resolution_clock> query_complete_end;	
std::chrono::duration<double, std::milli> query_complete_time(0);

std::chrono::time_point<std::chrono::high_resolution_clock> issue_query_start;
std::chrono::time_point<std::chrono::high_resolution_clock> issue_query_end;	
std::chrono::duration<double, std::milli> issue_query_time(0);

loadrun::LoadrunSettings loadrun_args;


////////////////////////////////
// Runner library integration //
////////////////////////////////


#include "runner_lib.h"
#include "nnpi_infer_job.h"
#include "multi_thread_job_sched.h"
#include "time_logger.h"
#include "cv_image_utils.h"
#include "nnpi_infer_mlperf_job.h"
#include "runner_lib_utils.h"


#include <atomic>
#include <list>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include <chrono>
#include <numeric>
#include <future>
#include <signal.h>

#define nnpiHostResourceBindToMem_NOT_SUPPORTED_WA

std::string blob_name;
std::string input_directory;

template<class E>
class fastResizeFifo {
public:
    fastResizeFifo() : m_size(0) , m_pool_size(16) {
        reserve(m_pool_size);
    }
    void reserve(size_t size) {
        m_pool_size = size;
        m_pool.resize(m_pool_size);
    }
    size_t size() const noexcept { return m_size; };
    void push_back(const E& e) {
        if (m_pool.empty()) {
            m_pool_size *= 2;
            m_pool.resize(m_pool_size);
        }

        m_fifo.splice(m_fifo.end(), m_pool, m_pool.begin());
        m_fifo.back() = e;
        m_size++;
    }
    const E& front() const {
        return m_fifo.front();
    }
    E& front() {
        return m_fifo.front();
    }
    void pop_front() noexcept {
        m_pool.splice(m_pool.end(), m_fifo, m_fifo.begin());
        m_size--;
    }
    void clear () noexcept {
        while(!empty()) {
            pop_front();
        }
    }
    bool empty() const noexcept { return m_size == 0; }
private:
    std::list<E> m_pool;
    std::list<E> m_fifo;
    size_t m_size;
    size_t m_pool_size;
};

class MlPerfFileIO {
public:
    using files_map = std::map <std::string, std::vector<uint8_t> >;
    typedef mlperf::QuerySample file_handle;
public:
    static void Init() {
        std::lock_guard<std::mutex> guard(queueMutex);
        m_fifo.reserve(1024);
        isStop = false;
    }

    static file_handle Open(const std::string& name) {
        return {0,0};
    }

    static const std::vector<std::vector<float>>& GetImageDataDb() {
        return image_data;
    }

    static void OpenMLPerf(const mlperf::QuerySampleIndex* samples, size_t sample_size) {
        auto max_index = std::max_element(samples, samples + sample_size);
        image_data.resize(std::max(*max_index + 1, image_data.size()));
        size_t concThreadsMax = std::thread::hardware_concurrency();
        size_t concThreads = 0;
        std::vector<std::future<void>> ct;
        ct.resize(concThreadsMax);
        std::cout << "Load images using " << concThreadsMax << " Threads\n";

        for (size_t i = 0; i < sample_size; i++) {
            auto file_index = samples[i];
            size_t data_size = 0;
            std::string file_name = runner_lib::format_text("%s/ILSVRC2012_val_%08u.JPEG", input_directory.c_str(), file_index + 1);

            auto& id = image_data[file_index];
            if(concThreads < concThreadsMax) {
                ct[concThreads++] = std::async(std::launch::async, [&id, file_name] {
                    std::vector<float> m_scales;
                    std::vector<float> m_means;
                    ///uint32_t imageSize = 0;
                    uint32_t imageSize = (uint32_t)sizeof(float) * 224 * 224 * 3;

                    const bool singed_int = false;///desc.dataType == NNPI_INF_PRECISION_INT8;
                    bool m_invertChannels = false;
                    //auto image = LoadImage(file_name.c_str(), 224, m_means, m_scales, imageSize, m_invertChannels, singed_int);
                    auto image = LoadImageOpenCVPreProcessing(file_name);
                    id = std::move(image);
                } );
            }
            if(concThreads == concThreadsMax || i+1 == sample_size) {
                for(size_t thi = 0; thi < concThreads; thi++)
                    ct[thi].get();
                concThreads = 0;
            }
            ///assert(imageSize == sizePerBatch);
            ///std::copy_n((uint8_t*)image.data(), imageSize, blob.data());
            ///assert(desc.layout == NNPI_INF_LAYOUT_NCHW);
            ///assert(desc.dataType == NNPI_INF_PRECISION_FLOAT32);
            //image_data[file_index] = std::move(image);
            // TODO: add reading of image
        }
    }

    static void CloseMLPerf(const mlperf::QuerySampleIndex* samples, size_t sample_size) {

        for (size_t i = 0; i < sample_size; i++) {
            image_data[samples[i]].clear();
        }
    }

    static void Stop() {
        std::cout << "MlPerfFileIO stop command send\n";
        isStop = true;
        fifoIsPushed.notify_all();
    }

    static void GetData(file_handle& handle, void* data, size_t size) {
        mlperf::QuerySample sample;
        {
            ///std::cout << "GetData from :" << std::this_thread::get_id() << std::endl;
            std::unique_lock<std::mutex> lk(queueMutex);
            fifoIsPushed.wait(lk, [] {if(isStop) return true; else return !m_fifo.empty();});
            THROW_IF_EQ_T(isStop, true, "MlPerfFileIO is stopped by command");
            sample = m_fifo.front();
            m_fifo.pop_front();

            if (!m_fifo.empty()) {
                lk.unlock();
                fifoIsPushed.notify_one();
                fifoIsPopped.notify_one();
            } else {
                lk.unlock();
                fifoIsPopped.notify_one();
            }
        }

        handle = sample;
        if(data != nullptr) {
            auto& vec = image_data[sample.index];
            void* image_data = vec.data();
            size_t image_size = vec.size() * sizeof(vec[0]);

            std::memcpy(data, image_data, std::min(image_size, size));
        }
    }

    static void WaitQueueEmpty() {
        while(!m_fifo.empty()); //TODO: find better way
    }

    static void QueueData(const mlperf::QuerySample *samples, size_t samples_size) {
        ///std::cout << "QueueData from :" << std::this_thread::get_id() <<  " sample_size: " << samples_size << std::endl;
        if (!isBackPressure) {
            {
                std::lock_guard<std::mutex> guard(queueMutex);

                for (size_t i = 0; i < samples_size; i++) {
                    m_fifo.push_back(samples[i]);
                }
            }
            fifoIsPushed.notify_one();
        } else {
            int can_be_queued_before_wait = 0;
            int to_queued_before_wait = 0;
            size_t i = 0;
            {
                std::lock_guard<std::mutex> guard(queueMutex);
                can_be_queued_before_wait = max_queue_size > m_fifo.size() ? max_queue_size - m_fifo.size() : 0;
                to_queued_before_wait = samples_size > can_be_queued_before_wait ? can_be_queued_before_wait : samples_size;
                for (; i < to_queued_before_wait; i++) {
                    m_fifo.push_back(samples[i]);
                }
            }
            ///std::cout << "Queued : to_queued_before_wait" << to_queued_before_wait << std::endl;
            fifoIsPushed.notify_one();
            if(i < samples_size) {
                std::unique_lock<std::mutex> lk(queueMutex);
                fifoIsPopped.wait(lk, [] {if(isStop) return true; else return m_fifo.empty();});
                for (; i < samples_size; i++) {
                    m_fifo.push_back(samples[i]);
                }
                lk.unlock();
                fifoIsPushed.notify_one();
                ///std::cout << "Queued : after_wait" << samples_size << std::endl;
            }
        }
    }

    static void ClearAll() {
        m_fifo.clear();
        image_data.clear();
    }
private:
    static std::vector<std::vector<float>> image_data;
    static fastResizeFifo<mlperf::QuerySample> m_fifo;
    static std::mutex queueMutex;
    static std::condition_variable fifoIsPushed;
    static std::condition_variable fifoIsPopped;
    static bool isStop;
public:
    static int max_queue_size;
    static bool isBackPressure;
};

std::vector<std::vector<float>> MlPerfFileIO::image_data;
fastResizeFifo<mlperf::QuerySample> MlPerfFileIO::m_fifo;
std::mutex MlPerfFileIO::queueMutex;
std::condition_variable MlPerfFileIO::fifoIsPushed;
std::condition_variable MlPerfFileIO::fifoIsPopped;
bool MlPerfFileIO::isStop;

bool MlPerfFileIO::isBackPressure = true;
int MlPerfFileIO::max_queue_size = 1;

using namespace runner_lib;

#ifdef nnpiHostResourceBindToMem_NOT_SUPPORTED_WA
using ml_infer_job_type = NNPIInferJobMlPerf<MlPerfFileIO, EmptyTimeLogger>;
#else
using ml_infer_job_type = NNPIInferJob<MlPerfFileIO, EmptyTimeLogger>;
#endif

class MlPerfResultValidator: public ResultValidatorInterface {
public:
    MlPerfResultValidator(bool is_validation_enabled) : m_is_validation_enabled(is_validation_enabled) { }
    virtual ~MlPerfResultValidator() { };
    virtual bool validate(const NetworkConfig& m_cfg, void* datanc) {
        auto data = reinterpret_cast<ml_infer_job_type::ValidationUserData*>(datanc);
        if (m_is_validation_enabled) {
            auto outdata = data->output_data[0].data();
            float *fptr = (float*) outdata;
            //std::vector<float> probs_vector {fptr, fptr + data->output_data[0].size()/4};
            auto probs_vector = runner_lib::make_buffer_view(fptr, data->output_data[0].size() / sizeof(float));
            std::vector<size_t> top1 = GetTopN(probs_vector, 1);
            uint32_t result = top1.at(0) - 1;
            mlperf::QuerySampleResponse resp { data->input_handle[0].id, (uintptr_t) (void*) &result, 4 };
            mlperf::QuerySamplesComplete(&resp, 1);
        } else {
            mlperf::QuerySampleResponse resp { data->input_handle[0].id, 0, 0 };
            mlperf::QuerySamplesComplete(&resp, 1);
        }
        return true;
    }
private:
    bool m_is_validation_enabled;
};


EmptyTimeLogger empty_mes;
InferenceRunner< ml_infer_job_type, MultiThreadJobScheduler<EmptyTimeLogger>, EmptyTimeLogger> inf_runner(empty_mes);

#ifdef nnpiHostResourceBindToMem_NOT_SUPPORTED_WA
#define COPY_CMD_LIMIT (64*1024 - 2)

template<class T, class U>
std::vector<std::vector<runner_lib::NamedResource>> NNPIInferJobMlPerf<T,U>::m_hostInputResourcesPerSample;
template<class T, class U>
std::mutex NNPIInferJobMlPerf<T,U>::initMutex;

#endif

/////////////////////////////////////////////////

int compare_sample(const void* a, const void* b){
  mlperf::QuerySampleIndex arg1;
  mlperf::QuerySampleIndex arg2;

  arg1 = static_cast<const mlperf::QuerySample*>(a)->index;
  arg2 = static_cast<const mlperf::QuerySample*>(b)->index;

  if(arg1 < arg2) return -1;
  if(arg1 > arg2) return 1;
  return 0;
}

void PrintSamplesIndexes(const mlperf::QuerySample* samples, size_t samples_size) {
    std::cout << "Samples: ";
    for(uint32_t i = 0; i < samples_size; i++) {
        std::cout << samples[i].index << " ";
    }
    std::cout << "\n\n";
}

void PrintSamplesIndexes(const mlperf::QuerySampleIndex* samples, size_t samples_size) {
    std::cout << "Samples: ";
    for(uint32_t i = 0; i < samples_size; i++) {
        std::cout << samples[i] << " ";
    }
    std::cout << "\n\n";
}

void IssueQuery(mlperf::c::ClientData client_data, const mlperf::QuerySample* samples, size_t samples_size) { 
    MlPerfFileIO::QueueData(samples, samples_size);
    return;
}

void ReportLatencyResults(mlperf::c::ClientData client_data, const int64_t*, size_t sample_size) {
  return;
}

void LoadSamplesToRam(mlperf::c::ClientData client_data, const mlperf::QuerySampleIndex* samples, size_t sample_size) {
  std::cout << "LoadSamplesToRam ";
  PrintSamplesIndexes(samples,sample_size);
  MlPerfFileIO::Init();
  MlPerfFileIO::OpenMLPerf(samples, sample_size);

  RuntimeConfig run_config;
  run_config.repeat_count_per_infer = -1;
  run_config.run_validators = true;
  run_config.warm_on = false;
  inf_runner.Execute(run_config);
  while(!inf_runner.m_job_Scheduler.isAllThreadsInitialized());
  std::cout << "Runner lib threads are started and ready to infer\n";

  return;
}

void UnloadSamplesFromRam(mlperf::c::ClientData client_data, const mlperf::QuerySampleIndex* samples, size_t sample_size) {
  std::cout << "UnLoadSamplesFromRam ";
  PrintSamplesIndexes(samples,sample_size);

  if(!isExceptionStorageEmpty())
      printStoredExceptions(std::cout);

  MlPerfFileIO::Stop();
  inf_runner.Complete();
  clearExceptionStorage();

  MlPerfFileIO::CloseMLPerf(samples,sample_size);
  return;
}

void FlushQueries() {
  return;
}

uint64_t constexpr mix(char m, uint64_t s) {
  return ((s<<7) + ~(s>>3)) + ~m;
}
 
uint64_t constexpr hash(const char * m) {
  return (*m) ? mix(*m,hash(m+1)) : 0;
}

void my_handler(int s){
   std::cout << "\nA signal caught, Sending Stop command to all threads\n";

   if(!isExceptionStorageEmpty())
       printStoredExceptions(std::cout);

   MlPerfFileIO::Stop();
   inf_runner.Complete();
   clearExceptionStorage();

   exit(1);
}



int main(int argc, char** argv) {
    /* Uncomment to disable output buffering */
    //setvbuf(stderr, NULL, _IONBF, 0);
    //setvbuf(stdout, NULL, _IONBF, 0);

    /* Add handler for signals like CNTR */
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);

	mlperf::TestSettings mlperf_args;
	size_t performance_samples = 0;
	size_t total_samples = 0;
	size_t card_num = 1;
	size_t dev_nets_per_card = 12;
	size_t parallel_infer_count = 2;

	// move framework related args to new array for initialization  
	std::vector<char*> inferencer_argv;
	int inferencer_argc = 0;
	for (int index = 0; index < argc; index++) {
		std::string arg(argv[index]);
		switch (hash(argv[index])) {
		case hash("--scenario"): {
			std::string scenario(argv[++index]);
			if (scenario.compare("Offline") == 0) {
				mlperf_args.scenario = mlperf::TestScenario::Offline;
				loadrun_args.is_offline = true;
			}
			else if (scenario.compare("Server") == 0) {
				mlperf_args.scenario = mlperf::TestScenario::Server;
			}
			else if (scenario.compare("SingleStream") == 0) {
				mlperf_args.scenario = mlperf::TestScenario::SingleStream;
			}
			else if (scenario.compare("MultiStream") == 0) {
				mlperf_args.scenario = mlperf::TestScenario::MultiStream;
			}
			else {
				std::cout << "Didn't recieve a valid scenario, options are: [SingleStream, MultiStream, Server, Offline]\n";
				std::exit(EXIT_FAILURE);
			}
			break;
		}
		case hash("--mode"): {
			std::string mode(argv[++index]);
			if (mode.compare("PerformanceOnly") == 0) {
				mlperf_args.mode = mlperf::TestMode::PerformanceOnly;
			}
			else if (mode.compare("AccuracyOnly") == 0) {
				mlperf_args.mode = mlperf::TestMode::AccuracyOnly;
			}
			else if (mode.compare("FindPeakPerformance") == 0) {
				mlperf_args.mode = mlperf::TestMode::FindPeakPerformance;
			}
			else {
				std::cout << "Only support PerformanceOnly and FindPeakPerformance mode\n";
				std::exit(EXIT_FAILURE);
			}
			break;
		}
		case hash("--blob"): {
		    blob_name = argv[++index];
		    break;
		}
		case hash("-i"): {
		    input_directory = argv[++index];
		    break;
		}
							 // single stream specific
		case hash("--single_stream_expected_latency_ns"):
			mlperf_args.single_stream_expected_latency_ns = std::stol(argv[++index]);
			break;
		case hash("--single_stream_target_latency_percentile"):
			mlperf_args.single_stream_target_latency_percentile = std::stod(argv[++index]);
			break;
			// multi stream specific
		case hash("--multi_stream_target_qps"):
			mlperf_args.multi_stream_target_qps = std::stod(argv[++index]);
			break;
		case hash("--multi_stream_target_latency_ns"):
			mlperf_args.multi_stream_target_latency_ns = std::stol(argv[++index]);
			break;
		case hash("--multi_stream_target_latency_percentile"):
			mlperf_args.multi_stream_target_latency_percentile = std::stod(argv[++index]);
			break;
		case hash("--multi_stream_samples_per_query"):
			mlperf_args.multi_stream_samples_per_query = std::stoi(argv[++index]);
			break;
		case hash("--multi_stream_max_async_queries"):
			mlperf_args.multi_stream_max_async_queries = std::stoi(argv[++index]);
			break;
			// server specific
		case hash("--server_target_qps"):
			mlperf_args.server_target_qps = std::stod(argv[++index]);
			break;
		case hash("--server_target_latency_ns"):
			mlperf_args.server_target_latency_ns = std::stol(argv[++index]);
			break;
		case hash("--server_target_latency_percentile"):
			mlperf_args.server_target_latency_percentile = std::stod(argv[++index]);
			break;
		case hash("--server_coalesce_queries"): {
			// NOT IMPLEMENTED YET!!!
			assert(0);
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				mlperf_args.server_coalesce_queries = true;
			}
			else {
				mlperf_args.server_coalesce_queries = false;
			}
			break;
		}
		case hash("--server_find_peak_qps_decimals_of_precision"):
			mlperf_args.server_find_peak_qps_decimals_of_precision = std::stoi(argv[++index]);
			break;
		case hash("--server_find_peak_qps_boundary_step_size"):
			mlperf_args.server_find_peak_qps_boundary_step_size = std::stod(argv[++index]);
			break;
			// offline specific
		case hash("--offline_expected_qps"):
			mlperf_args.offline_expected_qps = std::stod(argv[++index]);
			break;
			// test duration
		case hash("--min_query_count"):
			mlperf_args.min_query_count = std::stol(argv[++index]);
			break;
		case hash("--max_query_count"):
			mlperf_args.max_query_count = std::stol(argv[++index]);
			break;
		case hash("--min_duration_ms"):
			mlperf_args.min_duration_ms = std::stol(argv[++index]);
			break;
		case hash("--max_duration_ms"):
			mlperf_args.max_duration_ms = std::stol(argv[++index]);
			break;
			// random number generation
		case hash("--qsl_rng_seed"):
			mlperf_args.qsl_rng_seed = std::stol(argv[++index]);
			break;
		case hash("--sample_index_rng_seed"):
			mlperf_args.sample_index_rng_seed = std::stol(argv[++index]);
			break;
		case hash("--schedule_rng_seed"):
			mlperf_args.schedule_rng_seed = std::stol(argv[++index]);
			break;
		case hash("--accuracy_log_rng_seed"):
			mlperf_args.accuracy_log_rng_seed = std::stol(argv[++index]);
			break;
		case hash("--accuracy_log_probability"):
			mlperf_args.accuracy_log_probability = std::stod(argv[++index]);
			break;
			// performance sample modifiers
		case hash("--performance_issue_unique"): {
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				mlperf_args.performance_issue_unique = true;
			}
			else {
				mlperf_args.performance_issue_unique = false;
			}
			break;
		}
		case hash("--performance_issue_same"): {
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				mlperf_args.performance_issue_same = true;
			}
			else {
				mlperf_args.performance_issue_same = false;
			}
			break;
		}
		case hash("--performance_issue_same_index"):
			mlperf_args.performance_issue_same_index = std::stol(argv[++index]);
			break;
		case hash("--performance_sample_count_override"):
			mlperf_args.performance_sample_count_override = std::stol(argv[++index]);
			break;
			// loadrun parameters
		case hash("--performance_samples"):
			performance_samples = std::stoi(argv[++index]);
			break;
		case hash("--total_samples"):
			total_samples = std::stoi(argv[++index]);
			break;
		case hash("--loadrun_queue_size"):
			loadrun_args.loadrun_queue_size = std::stoi(argv[++index]);
			break;
		case hash("--interactive"): {
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				interactive = true;
			}
			else {
				interactive = false;
			}
			break;
		}
		case hash("--card_num"): {
		    card_num = std::stoi(argv[++index]);
		    break;
		}
		case hash("--dev_nets_per_card"): {
			dev_nets_per_card = std::stoi(argv[++index]);
		    break;
		}
		case hash("--parallel_infer_count"): {
			parallel_infer_count = std::stoi(argv[++index]);
		    break;
		}
		case hash("--offline_single_response"): {
			// when true, put all samples in one query's response
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				offline_single_response = true;
			}
			else {
				offline_single_response = false;
			}
			break;
		}
		case hash("--loadrun_sleep_after_each_response_check"): {
			// when true, put all samples in one query's response
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				loadrun_sleep_after_each_response_check = true;
			}
			else {
				loadrun_sleep_after_each_response_check = false;
			}
			break;
		}
		case hash("--flush_queries"): {
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				handle_flush_queries = true;
			}
			else {
				handle_flush_queries = false;
			}
			break;
		}
		case hash("--sort_samples"): {
			std::string temp(argv[++index]);
			if (temp.compare("true") == 0) {
				sort_samples = true;
			}
			else {
				sort_samples = false;
			}
			break;
		}
		default:
			// parameters for the backend
			inferencer_argv.push_back(argv[index]);
			inferencer_argc++;
			break;
		}
	};

	loadrun_args.batch_size = loadrun_args.loadrun_queue_size;

#ifdef nnpiHostResourceBindToMem_NOT_SUPPORTED_WA
	int net_input_count = 1; // TODO: use real network number of inputs
	int net_output_count = 1; // TODO: use real network number of outputs
	int input_cpy_cmd_per_sample = card_num * dev_nets_per_card * parallel_infer_count * net_input_count;
	int output_cpy_cmd_per = card_num * dev_nets_per_card * parallel_infer_count * net_output_count;
	int max_hostres_samples = (COPY_CMD_LIMIT - output_cpy_cmd_per) /  input_cpy_cmd_per_sample;
	if (performance_samples > max_hostres_samples) {
		std::cout << "[WARNING] Performance samples reduced from "<< performance_samples << " to "<< max_hostres_samples << " due to limit on copy command count \n";
		performance_samples = max_hostres_samples;
	}
#endif


	MlPerfFileIO::Init(); // Init IO interface for runner lib
        InferenceConfig inf_cfg;
        NetworkConfig network;
        MlPerfResultValidator validator(mlperf_args.mode == mlperf::TestMode::AccuracyOnly);
        network.batch_sizes = loadrun_args.batch_size;
        network.infer_types = "image";
        network.inputFiles.resize(1); // TODO: change InitInfer() to be more generic
        network.outputFiles.resize(1); // TODO: change InitInfer() to be more generic
        switch(mlperf_args.scenario) {
            case mlperf::TestScenario::SingleStream:
                network.parallel_infer_count = 1;///mlperf_args.multi_stream_max_async_queries;
                for(uint32_t i = 0; i < card_num; i++)
                    network.dev_ids.resize(1 * (i+1),i);
                MlPerfFileIO::max_queue_size = 1;
                MlPerfFileIO::isBackPressure = false;
                break;
            default:
                network.parallel_infer_count = parallel_infer_count;
                for(uint32_t i = 0; i < card_num; i++)
                    network.dev_ids.resize(dev_nets_per_card * (i+1),i);
                MlPerfFileIO::max_queue_size = dev_nets_per_card * card_num * network.parallel_infer_count;
                MlPerfFileIO::isBackPressure = true;
                //MlPerfFileIO::isBackPressure = false;
        }
        network.workloadFile = blob_name;
        network.validator = &validator;
        inf_cfg.networks.push_back(network);
        inf_runner.Setup(inf_cfg);

//        {
//            /* Warm up */
//            std::cout << "Warming up\n";
//            validator.is_validation_enabled = false;
//            size_t thread_num = network.parallel_infer_count * network.dev_ids.size();
//
//            std::vector<mlperf::QuerySample> QuerySampleSet(thread_num, {0,0});
//            std::vector<mlperf::QuerySampleIndex> QuerySampleIndexSet(thread_num, 0);
//
//            MlPerfFileIO::OpenMLPerf(QuerySampleIndexSet.data(), QuerySampleIndexSet.size());
//            MlPerfFileIO::QueueData(QuerySampleSet.data(), QuerySampleSet.size());
//            MlPerfFileIO::WaitQueueEmpty();

//            using namespace std::chrono_literals;
//            std::this_thread::sleep_for(2s);
//            MlPerfFileIO::CloseMLPerf(QuerySampleIndexSet.data(), QuerySampleIndexSet.size());
//            validator.is_validation_enabled = true;
//        }




	mlperf::SystemUnderTest* sut = (mlperf::SystemUnderTest*)mlperf::c::ConstructSUT(0, "SPH_SUT", 4,
		IssueQuery, FlushQueries, ReportLatencyResults);
	mlperf::QuerySampleLibrary* qsl = (mlperf::QuerySampleLibrary*)mlperf::c::ConstructQSL(0, "SPH_SUT", 4,
		total_samples, performance_samples, LoadSamplesToRam,
		UnloadSamplesFromRam);

	std::cout << "====== SPH arguments ====== " << "\n";
	std::cout << "number of cards in use is " << card_num << "\n";
	std::cout << "number of ices used per card is " << dev_nets_per_card << "\n";
	std::cout << "number of parallel infers per ice is " << parallel_infer_count << "\n";
	std::cout << "number of total infers used is " << card_num * dev_nets_per_card* parallel_infer_count << "\n";
	std::cout << "====== MLPerf arguments ====== " << "\n";
	std::cout << "mlperf performance_samples is " << performance_samples << "\n";
	std::cout << "mlperf total_samples is " << total_samples << "\n";
	std::cout << "mlperf min_query_count is " << mlperf_args.min_query_count << "\n";
	std::cout << "mlperf min_duration_ms is " << mlperf_args.min_duration_ms << "\n";
	switch (mlperf_args.scenario) {
	case(mlperf::TestScenario::SingleStream): {
		std::cout << "mlperf single_stream_expected_latency_ns  is " << mlperf_args.single_stream_expected_latency_ns << "\n";
		std::cout << "mlperf single_stream_target_latency_percentile  is " << mlperf_args.single_stream_target_latency_percentile << "\n";
		break;
	}
	case(mlperf::TestScenario::MultiStream): {
		std::cout << "mlperf multi_stream_target_qps   is " << mlperf_args.multi_stream_target_qps << "\n";
		std::cout << "mlperf multi_stream_target_latency_ns   is " << mlperf_args.multi_stream_target_latency_ns << "\n";
		std::cout << "mlperf multi_stream_target_latency_percentile  is " << mlperf_args.multi_stream_target_latency_percentile << "\n";
		std::cout << "mlperf multi_stream_samples_per_query   is " << mlperf_args.multi_stream_samples_per_query << "\n";
		std::cout << "mlperf multi_stream_max_async_queries   is " << mlperf_args.multi_stream_max_async_queries << "\n";
		break;
	}
	case(mlperf::TestScenario::Server): {
		std::cout << "mlperf server_target_qps is " << mlperf_args.server_target_qps << "\n";
		std::cout << "mlperf server_target_latency_ns   is " << mlperf_args.server_target_latency_ns << "\n";
		std::cout << "mlperf server_target_latency_percentile  is " << mlperf_args.server_target_latency_percentile << "\n";
		std::cout << "mlperf server_coalesce_queries    is " << mlperf_args.server_coalesce_queries << "\n";
		std::cout << "mlperf server_find_peak_qps_decimals_of_precision  is " << mlperf_args.server_find_peak_qps_decimals_of_precision << "\n";
		std::cout << "mlperf server_find_peak_qps_boundary_step_size    is " << mlperf_args.server_find_peak_qps_boundary_step_size << "\n";
		break;
	}
	case(mlperf::TestScenario::Offline): {
		std::cout << "mlperf offline_expected_qps is " << mlperf_args.offline_expected_qps << "\n";
		break;
	}
	}

	std::chrono::time_point<std::chrono::high_resolution_clock> ben_start;
	std::chrono::time_point<std::chrono::high_resolution_clock> ben_end;
	std::chrono::duration<double, std::milli> ben_time(0);
	ben_start = std::chrono::high_resolution_clock::now();

	mlperf::c::StartTest(sut, qsl, mlperf_args);

	ben_end = std::chrono::high_resolution_clock::now();
	ben_time = ben_end - ben_start;
	double ben_seconds = ben_time.count() / 1000;
	std::cout << "\nLoadgen time is " << ben_seconds << " seconds\n";

	double hd_seconds = 0;
	double run_seconds = 0;
	float top1 = 0;
	float top5 = 0;
	return 0;

}

