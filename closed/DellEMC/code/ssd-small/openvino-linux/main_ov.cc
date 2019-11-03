#include <string>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <regex>

#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"
#include "dataset_ov.h"
#include "sut_ov.h"

#define NANO_SEC 1e9
#define MILLI_SEC 1000
#define MILLI_TO_NANO 1000000

std::unique_ptr<Dataset> ds;
std::unique_ptr<Sut> sut;
mlperf::TestSettings settings;
mlperf::LogSettings log_settings;

void issueQueries(mlperf::c::ClientData client,
        const mlperf::QuerySample* samples, size_t size) {
    sut->issueQueries(samples, size,
            ((settings.mode == mlperf::TestMode::AccuracyOnly) ? 1 : 0));
}

void processLatencies(mlperf::c::ClientData client, const int64_t* latencies,
        size_t size) {
    sut->processLatencies(latencies, size);
}

void flushQueries(void) {
    sut->flushQueries();
}

void loadQuerySamples(mlperf::c::ClientData client,
        const mlperf::QuerySampleIndex* samples, size_t num_samples) {
    ds->loadQuerySamples(samples, num_samples);
}

void unloadQuerySamples(mlperf::c::ClientData client,
        const mlperf::QuerySampleIndex* samples, size_t num_samples) {
    ds->unloadQuerySamples(samples, num_samples);
}

int main(int argc, char **argv) {
    settings.mode = mlperf::TestMode::PerformanceOnly;
    settings.scenario = mlperf::TestScenario::SingleStream;
    int image_width = 224, image_height = 224, num_channels = 3;
    std::string image_format = "NCHW";
    size_t total_sample_count = 0, perf_sample_count = 200, nstreams = 1,
            batch_size = 1, nireq = 1, nthreads = 56, nwarmup_iters = 0;
    std::string scenario, mode, datapath, model, workload, dataset,
            mlperf_conf_filename = "mlperf.conf", user_conf_filename =
                    "user.conf", device = "cpu";
    std::map<std::string, int> mp;

    // MLPerf/LoadGen settings
    mp.insert(std::pair<std::string, int>("--scenario", 1)); // Allowed values: "SingleStream", "Offline", "Server", "MultiStream"
    mp.insert(std::pair<std::string, int>("--mode", 2)); // Allowed values: "Accuracy", "Performance", "Submission"
    mp.insert(std::pair<std::string, int>("--mlperf_conf_filename", 3));
    mp.insert(std::pair<std::string, int>("--user_conf_filename", 4));
    mp.insert(std::pair<std::string, int>("--total_sample_count", 5)); // ?
    mp.insert(std::pair<std::string, int>("--perf_sample_count", 6)); // ?

    // WOrkload settings
    mp.insert(std::pair<std::string, int>("--data_path", 7)); // Path to the dataset
    mp.insert(std::pair<std::string, int>("--dataset", 8)); // Allowed values: "imagenet", "coco"
    mp.insert(std::pair<std::string, int>("--model_path", 9)); // Path to the model file
    mp.insert(std::pair<std::string, int>("--model_name", 10)); // Allowed values: resnet50, mobilenet, ssd-mobilenet, ssd-resnet34

    // OpenVino settings
    mp.insert(std::pair<std::string, int>("--nstreams", 11)); // # of inference streams
    mp.insert(std::pair<std::string, int>("--nireq", 12)); // # of inference requests
    mp.insert(std::pair<std::string, int>("--nthreads", 13)); // # of CPU threads
    mp.insert(std::pair<std::string, int>("--batch_size", 14)); // per stream
    mp.insert(std::pair<std::string, int>("--device", 15)); // Allowed values: "cpu", "gpu"

    // Other general settings
    mp.insert(std::pair<std::string, int>("--nwarmup_iters", 16));

    for (int i = 1; i < argc; ++i) {
        std::string arg = std::string(argv[i]);
        switch (mp[arg]) {
        case 1: {
            scenario = std::string(argv[++i]);
            if (scenario.compare("SingleStream") == 0) {
                settings.scenario = mlperf::TestScenario::SingleStream;
            }
            if (scenario.compare("Server") == 0) {
                settings.scenario = mlperf::TestScenario::Server;
            }
            if (scenario.compare("Offline") == 0) {
                settings.scenario = mlperf::TestScenario::Offline;
            }
            if (scenario.compare("MultiStream") == 0) {
                settings.scenario = mlperf::TestScenario::MultiStream;
            }
        }
            break;
        case 2: {
            mode = std::string(argv[++i]);
            if (mode.compare("Accuracy") == 0) {
                settings.mode = mlperf::TestMode::AccuracyOnly;
            }
            if (mode.compare("Performance") == 0) {
                settings.mode = mlperf::TestMode::PerformanceOnly;
            }
            if (mode.compare("Submission") == 0) {
                settings.mode = mlperf::TestMode::SubmissionRun;
            }
            if (mode.compare("FindPeakPerformance") == 0) {
                settings.mode = mlperf::TestMode::FindPeakPerformance;
            }
        }
            break;
        case 3: {
            mlperf_conf_filename = argv[++i];
        }
            break;
        case 4: {
            user_conf_filename = argv[++i];
        }
            break;
        case 5: {
            total_sample_count = std::stoi(argv[++i]);
        }
            break;
        case 6: {
            perf_sample_count = std::stoi(argv[++i]);
        }
            break;
        case 7: {
            datapath = argv[++i];
        }
            break;
        case 8: {
            dataset = argv[++i];
        }
            break;
        case 9: {
            model = argv[++i];
        }
            break;
        case 10: {
            workload = argv[++i];
        }
            break;
        case 11: {
            nstreams = std::stoi(argv[++i]);
        }
            break;
        case 12: {
            nireq = std::stoi(argv[++i]);
        }
            break;
        case 13: {
            nthreads = std::stoi(argv[++i]);
        }
            break;
        case 14: {
            batch_size = std::stoi(argv[++i]);
        }
            break;
        case 15: {
            device = argv[++i];
        }
            break;
        case 16: {
            nwarmup_iters = std::stoi(argv[++i]);
        }
            break;
        default:
            std::cout << "Unknown option: " << arg << "\n";
            break;
        }
    }

    if (!(mlperf_conf_filename.compare("") == 0)) {
        settings.FromConfig(mlperf_conf_filename, workload, scenario);
    }
    if (!(user_conf_filename.compare("") == 0)) {
        settings.FromConfig(user_conf_filename, workload, scenario);
    }

    log_settings.enable_trace = false;
    perf_sample_count = settings.performance_sample_count_override;

    if (workload.compare("resnet50") == 0) {
        image_format = "NCHW";
        image_height = 224;
        image_width = 224;
        num_channels = 3;
        total_sample_count = 50000;
    } else if (workload.compare("mobilenet") == 0) {
        image_format = "NCHW";
        image_height = 224;
        image_width = 224;
        num_channels = 3;
        total_sample_count = 50000;
    } else if (workload.compare("ssd-mobilenet") == 0) {
        image_format = "NCHW";
        image_height = 300;
        image_width = 300;
        num_channels = 3;
        total_sample_count = 5000;
    } else if (workload.compare("ssd-resnet34") == 0) {
        image_format = "NCHW";
        image_height = 1200;
        image_width = 1200;
        num_channels = 3;
        total_sample_count = 5000;
    }

    // Init SUT
    sut = std::unique_ptr < Sut
            > (new Sut(settings, nireq, nstreams, nthreads, model, batch_size,
                    dataset, workload));
    // Init Dataset
    if (dataset.compare("imagenet") == 0) {
        ds = std::unique_ptr < Imagenet
                > (new Imagenet(settings, image_width, image_height,
                        num_channels, datapath, image_format,
                        total_sample_count, perf_sample_count, workload,
                        dataset));
    } else if (dataset.compare("coco") == 0) {
        ds = std::unique_ptr < Coco
                > (new Coco(settings, image_width, image_height, num_channels,
                        datapath, image_format, total_sample_count,
                        perf_sample_count, workload, dataset));
    }

    if (nwarmup_iters > 0) {
        sut->warmUp(nwarmup_iters, batch_size, workload);
    }

    void* sut = mlperf::c::ConstructSUT(0, "SUT", 4, issueQueries, flushQueries,
            processLatencies);

    void* qsl = mlperf::c::ConstructQSL(0, "QSL", 4, total_sample_count,
            perf_sample_count, loadQuerySamples, unloadQuerySamples);

    mlperf::c::StartTest(sut, qsl, settings, log_settings);

    mlperf::c::DestroyQSL(qsl);
    mlperf::c::DestroySUT(sut);

    return 0;
}
