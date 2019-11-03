#ifndef SUT_H__
#define SUT_H__

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include "backend_ov.h"
#include "item_ov.h"

extern std::unique_ptr<Dataset> ds;
using namespace InferenceEngine;

class Sut {

public:
    Sut(mlperf::TestSettings settings, unsigned nireq, int nstreams,
            int nthreads, const std::string model, int batch_size,
            std::string dataset, std::string workload) :
            settings_(settings), batch_size_(batch_size), nstreams_(nstreams), nireq_(
                    nireq), workload_(workload), backend_ov_(
                    std::unique_ptr < BackendOV
                            > (new BackendOV(settings, nireq, nstreams,
                                    batch_size, nthreads, dataset, workload))) {
        // load model to backend
        backend_ov_->load(model);
        qs_.setBackend(backend_ov_);
        qs_offline_.setBackend(backend_ov_);
        qs_server_.setBackend(backend_ov_);
        qs_ms_.setBackend(backend_ov_);
    }

    void warmUp(size_t nwarmup_iters, size_t batch_size, std::string workload) {
        std::vector < mlperf::QuerySampleIndex > samples;

        std::vector < mlperf::ResponseId > queryIds;

        for (size_t i = 0; i < batch_size; ++i) {
            samples.push_back(0);
            queryIds.push_back(1);
        }

        ds->loadQuerySamples(samples.data(), batch_size);

        for (size_t i = 0; i < nwarmup_iters; ++i) {
            if (settings_.scenario == mlperf::TestScenario::SingleStream) {
                Blob::Ptr data;
                int label = 0;
                unsigned res = 0;
                ds->getSamples(samples.data(), &data, &label);
                if ((workload.compare("ssd-mobilenet") == 0)
                        || (workload.compare("ssd-resnet34") == 0)) {
                    std::vector<float> result;
                    std::vector<unsigned> counts;
                    backend_ov_->predict(data, result, 0, 1, counts);
                } else {
                    backend_ov_->predict(data, label, res);
                }
            } else if (settings_.scenario == mlperf::TestScenario::Offline) {
                std::vector < mlperf::ResponseId > results_ids;
                std::vector < mlperf::QuerySampleIndex > sample_idxs;

                std::vector<Item> items;
                ds->getSamplesBatched(samples, queryIds, batch_size, 1, items);

                if ((workload.compare("ssd-mobilenet") == 0)
                        || (workload.compare("ssd-resnet34") == 0)) {
                    std::vector<float> results;
                    std::vector<unsigned> counts;
                    backend_ov_->predictOffline(items, results, results_ids,
                            sample_idxs, counts, 1);

                } else {
                    std::vector<unsigned> results;
                    backend_ov_->predictOffline(items, results, results_ids, 1);
                }

            } else if (settings_.scenario == mlperf::TestScenario::Server) {
                Blob::Ptr data;
                int label = 0;
                if (batch_size == 1) {
                    ds->getSamples(samples.data(), &data, &label);
                    Item item = Item(data, queryIds, samples);
                    backend_ov_->predictServer(item, true);
                } else {
                    int num_batches = 1;
                    std::vector<Item> items;
                    ds->getSamplesBatchedServer(samples, queryIds, batch_size,
                            num_batches, items);
                    backend_ov_->predictServer(items[0], true);
                }
            } else if (settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                std::map<Blob::Ptr, std::vector<mlperf::ResponseId>> data;
                std::vector < mlperf::ResponseId > results_ids;
                std::vector < mlperf::QuerySampleIndex > sample_idxs;

                std::vector<Item> items;
                ds->getSamplesBatchedMultiStream(samples, queryIds, batch_size,
                        1, items);
                if ((workload.compare("ssd-mobilenet") == 0)
                        || (workload.compare("ssd-resnet34") == 0)) {
                    std::vector<float> results;
                    std::vector<unsigned> counts;
                    backend_ov_->predictMultiStream(items, results, results_ids,
                            sample_idxs, counts, 1);

                } else {
                    std::vector<unsigned> results;
                    backend_ov_->predictMultiStream(items, results, results_ids,
                            1);
                }
            }
        }

        ds->unloadQuerySamples(samples.data(), 0);

        // end warmup
        if (settings_.scenario == mlperf::TestScenario::Offline) {
            backend_ov_->reset();
        } else if (settings_.scenario == mlperf::TestScenario::MultiStream) {
            backend_ov_->reset();
        }
    }

    ~Sut() {
    }

    class QueryScheduler {
    public:
        QueryScheduler() {

        }

        ~QueryScheduler() {
        }

        void setBackend(std::unique_ptr<BackendOV> &backend_ov) {
            backend_ov_ = (backend_ov).get();
        }

        void runOneItem(Blob::Ptr data, int label,
                mlperf::ResponseId query_id) {
            unsigned int result = 0;

            backend_ov_->predict(data, label, result);

            std::vector < mlperf::QuerySampleResponse > responses;
            mlperf::QuerySampleResponse response { query_id,
                    reinterpret_cast<std::uintptr_t>(&result), sizeof(result) };
            responses.push_back(response);
            mlperf::QuerySamplesComplete(&responses[0], 1);

            return;
        }

        void runOneItem(Blob::Ptr data, mlperf::ResponseId query_id,
                mlperf::QuerySampleIndex sample_id) {
            std::vector<float> result;
            std::vector<unsigned> counts;

            backend_ov_->predict(data, result, sample_id, query_id, counts);

            std::vector < mlperf::QuerySampleResponse > responses;
            mlperf::QuerySampleResponse response { query_id,
                    reinterpret_cast<std::uintptr_t>(&result[0]), (sizeof(float)
                            * counts[0]) };
            responses.push_back(response);
            mlperf::QuerySamplesComplete(responses.data(), responses.size());

            return;
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size) {
            std::vector < mlperf::QuerySampleIndex > idxs;
            idxs.push_back(samples[0].index);

            mlperf::ResponseId query_id = samples[0].id;
            Blob::Ptr data;
            int label = 0;
            ds->getSamples(&idxs[0], &data, &label);

            if (ds->dataset_.compare("imagenet") == 0) {
                runOneItem(data, label, query_id);
            } else {
                runOneItem(data, query_id, samples[0].index);
            }
        }
    private:
        BackendOV * backend_ov_;
    };

    class QuerySchedulerOffline {
    public:
        QuerySchedulerOffline() {

        }

        ~QuerySchedulerOffline() {
        }

        void setBackend(std::unique_ptr<BackendOV> &backend_ov) {
            backend_ov_ = backend_ov.get();
        }

        void runOneItem(std::vector<Item> items, std::vector<unsigned> &results,
                std::vector<mlperf::ResponseId> &response_ids,
                int num_batches) {

            backend_ov_->predictOffline(items, results, response_ids,
                    num_batches);

            return;
        }

        void runOneItem(std::vector<Item> items, std::vector<float> &results,
                std::vector<mlperf::ResponseId> &response_ids,
                std::vector<mlperf::QuerySampleIndex> &sample_idxs,
                std::vector<unsigned> &counts, int num_batches) {

            backend_ov_->predictOffline(items, results, response_ids,
                    sample_idxs, counts, num_batches);

            return;
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size,
                unsigned nireq, int bs, int nstreams) {
            std::vector < mlperf::QuerySampleResponse > responses;
            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;
            std::vector<int> labels;

            for (size_t i = 0; i < size; ++i) {
                auto sample = (*samples);
                samples++;
                sample_idxs.push_back(sample.index);
                response_ids.push_back(sample.id);
            }

            int num_batches = size / bs;
            std::vector<Item> items;
            ds->getSamplesBatched(sample_idxs, response_ids, bs, num_batches,
                    items);

            std::vector < mlperf::ResponseId > ids;
            std::vector < mlperf::QuerySampleIndex > idxs;

            if (ds->dataset_.compare("imagenet") == 0) {
                std::vector<unsigned> results;

                runOneItem(items, results, ids, num_batches);

                for (size_t i = 0; i < size; ++i) {
                    mlperf::QuerySampleResponse response { ids[i],
                            reinterpret_cast<std::uintptr_t>(&results[i]),
                            sizeof(results[i]) };

                    responses.push_back(response);
                }

                mlperf::QuerySamplesComplete(responses.data(),
                        responses.size());
            } else {
                std::vector<float> results;
                std::vector<unsigned> counts;

                runOneItem(items, results, ids, idxs, counts, num_batches);

                size_t idx = 0;
                for (size_t i = 0; i < size; ++i) {
                    mlperf::QuerySampleResponse response { ids[i],
                            reinterpret_cast<std::uintptr_t>(&results[idx]),
                            (sizeof(float) * counts[i]) };
                    responses.push_back(response);
                    idx = idx + counts[i];
                }

                mlperf::QuerySamplesComplete(responses.data(),
                        responses.size());
            }

            backend_ov_->reset();
        }

    private:
        BackendOV *backend_ov_;
    };

    class QuerySchedulerServer {
    public:
        QuerySchedulerServer() {

        }

        ~QuerySchedulerServer() {
        }

        void setBackend(std::unique_ptr<BackendOV> &backend_ov) {
            backend_ov_ = backend_ov.get();
        }

        void runOneItem(Item item) {
            backend_ov_->predictServer(item);

            return;
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size, int bs) {
            if (bs == 1) {
                enqueueOne(samples, size);
            } else {
                enqueueMany(samples, size, bs);
            }
        }

        void enqueueOne(const mlperf::QuerySample* samples, size_t size) {
            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;

            auto sample = (*samples);
            sample_idxs.push_back(sample.index);
            response_ids.push_back(sample.id);

            Blob::Ptr data;
            int label = 0;
            ds->getSamples(&sample_idxs[0], &data, &label);
            Item item = Item(data, response_ids, sample_idxs);

            runOneItem(item);
        }

        void enqueueMany(const mlperf::QuerySample* samples, size_t size,
                int bs) {
            static std::vector<mlperf::QuerySampleIndex> global_sample_idxs;
            static std::vector<mlperf::ResponseId> global_response_ids;

            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;

            auto sample = (*samples);
            global_sample_idxs.push_back(sample.index);
            global_response_ids.push_back(sample.id);

            if (global_sample_idxs.size() >= (unsigned) bs) {
                sample_idxs = global_sample_idxs;
                response_ids = global_response_ids;
                global_sample_idxs.clear();
                global_response_ids.clear();

                int num_batches = 1;
                std::vector<Item> items;
                ds->getSamplesBatchedServer(sample_idxs, response_ids, bs,
                        num_batches, items);
                runOneItem(items[0]);
            }
        }

    private:
        BackendOV *backend_ov_;
    };

    class QuerySchedulerMultiStream {
    public:
        QuerySchedulerMultiStream() {

        }

        ~QuerySchedulerMultiStream() {
        }

        void setBackend(std::unique_ptr<BackendOV> &backend_ov) {
            backend_ov_ = backend_ov.get();
        }

        void runOneItem(std::vector<Item> items, std::vector<unsigned> &results,
                std::vector<mlperf::ResponseId> &response_ids,
                std::vector<mlperf::QuerySampleIndex> &sample_idxs,
                int num_batches) {

            backend_ov_->predictMultiStream(items, results, response_ids,
                    num_batches);

            return;
        }

        void runOneItem(std::vector<Item> items, std::vector<float> &results,
                std::vector<mlperf::ResponseId> &response_ids,
                std::vector<mlperf::QuerySampleIndex> &sample_idxs,
                std::vector<unsigned> &counts, int num_batches) {

            backend_ov_->predictMultiStream(items, results, response_ids,
                    sample_idxs, counts, num_batches);

            return;
        }

        void enqueue(const mlperf::QuerySample* samples, size_t size,
                unsigned nireq, int bs, int nstreams) {
            std::vector < mlperf::QuerySampleResponse > responses;
            std::vector < mlperf::QuerySampleIndex > sample_idxs;
            std::vector < mlperf::ResponseId > response_ids;

            for (size_t i = 0; i < size; ++i) {
                auto sample = (*samples);
                samples++;
                sample_idxs.push_back(sample.index);
                response_ids.push_back(sample.id);
            }

            int num_batches = size / bs;

            std::vector<Item> items;
            ds->getSamplesBatchedMultiStream(sample_idxs, response_ids, bs,
                    num_batches, items);

            /////////
            std::vector < mlperf::ResponseId > ids;
            std::vector < mlperf::QuerySampleIndex > idxs;

            if (ds->dataset_.compare("imagenet") == 0) {
                std::vector<unsigned> results;
                runOneItem(items, results, ids, idxs, num_batches);

                for (size_t i = 0; i < size; ++i) {
                    mlperf::QuerySampleResponse response { ids[i],
                            reinterpret_cast<std::uintptr_t>(&results[i]),
                            sizeof(results[i]) };
                    responses.push_back(response);
                }

                mlperf::QuerySamplesComplete(&responses[0], size);
            } else {
                std::vector<float> results;
                std::vector<unsigned> counts;

                runOneItem(items, results, ids, idxs, counts, num_batches);
                size_t idx = 0;
                for (size_t i = 0; i < size; ++i) {
                    mlperf::QuerySampleResponse response { ids[i],
                            reinterpret_cast<std::uintptr_t>(&results[idx]),
                            (sizeof(float) * counts[i]) };
                    responses.push_back(response);
                    idx = idx + counts[i];
                }

                mlperf::QuerySamplesComplete(responses.data(),
                        responses.size());
            }

            backend_ov_->reset();
        }
    private:
        BackendOV *backend_ov_;
    };

    void issueQueries(const mlperf::QuerySample* samples, size_t size,
            bool accuracy) {
        if (settings_.scenario == mlperf::TestScenario::Offline) {
            qs_offline_.enqueue(samples, size, nireq_, batch_size_, nstreams_);
        } else if (settings_.scenario == mlperf::TestScenario::SingleStream) {
            qs_.enqueue(samples, size);
        } else if (settings_.scenario == mlperf::TestScenario::Server) {
            qs_server_.enqueue(samples, size, batch_size_);
        } else if (settings_.scenario == mlperf::TestScenario::MultiStream) {
            qs_ms_.enqueue(samples, size, nireq_, batch_size_, nstreams_);
        }
    }

    void processLatencies(const int64_t* latencies, size_t size) {
        return;
    }

    void flushQueries(void) {
        return;
    }

public:

    mlperf::TestSettings settings_;
    int batch_size_ = 1;
    int nstreams_ = 1;
    unsigned nireq_ = 1;
    std::string workload_;
    std::unique_ptr<BackendOV> backend_ov_;
    //std::string dataset_;
    QueryScheduler qs_;
    QuerySchedulerOffline qs_offline_;
    QuerySchedulerServer qs_server_;
    QuerySchedulerMultiStream qs_ms_;

};

#endif
