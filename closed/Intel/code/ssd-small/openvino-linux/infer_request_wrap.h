#pragma once

#include <queue>
#include <condition_variable>
#include <mutex>

#include <inference_engine.hpp>
#include <ie_blob.h>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include "item_ov.h"

extern std::unique_ptr<Dataset> ds;

typedef std::function<
        void(size_t id, InferenceEngine::InferRequest req, Item input)> QueueCallbackFunction;

/// @brief Wrapper class for InferenceEngine::InferRequest. Handles asynchronous callbacks .
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    explicit InferReqWrap(InferenceEngine::ExecutableNetwork& net, size_t id,
            std::string outName, mlperf::TestSettings settings,
            std::string workload, QueueCallbackFunction callback_queue) :
            request_(net.CreateInferRequest()), id_(id), output_name_(outName), settings_(
                    settings), workload_(workload), callback_queue_(
                    callback_queue) {
        request_.SetCompletionCallback([&]() {

            callback();
        });
    }

    void callback() {
        if ((settings_.scenario == mlperf::TestScenario::Offline)
                || (settings_.scenario == mlperf::TestScenario::MultiStream)) {
            callback_queue_(id_, request_, input_);
        } else if (settings_.scenario == mlperf::TestScenario::Server) { //TODO: rps: Fix this logic
            if (!is_warm_up) {
                std::vector < mlperf::QuerySampleResponse > responses;
                unsigned bs = input_.sample_idxs_.size();

                if (workload_.compare("ssd-resnet34") == 0) { // TODO: rps: ssd-resnet34 only supports bs=1
                    std::vector<Item> outputs;
                    std::vector<float> result;
                    std::vector<unsigned> counts;
                    std::vector < mlperf::ResponseId > response_ids;

                    input_.blob_ = getBlob("Unsqueeze_bboxes777");
                    input_.blob1_ = getBlob("Unsqueeze_scores835");
                    input_.blob2_ = getBlob("Add_labels");

                    outputs.push_back(input_);

                    postProcessSSDResnet(outputs, result, counts, response_ids,
                            1);

                    mlperf::QuerySampleResponse response { response_ids[0],
                            reinterpret_cast<std::uintptr_t>(&result[0]),
                            (result.size() * sizeof(float)) };
                    responses.push_back(response);

                    mlperf::QuerySamplesComplete(&responses[0], 1);
                } else if (workload_.compare("ssd-mobilenet") == 0) {
                    std::vector<Item> outputs;
                    std::vector<unsigned> counts;
                    std::vector<float> results;

                    std::vector < mlperf::ResponseId > response_ids;
                    input_.blob_ = getBlob(output_name_);
                    outputs.push_back(input_);
                    postProcessSSDMobilenet(outputs, results, counts,
                            response_ids, 1, bs);

                    size_t idx = 0;
                    for (size_t i = 0; i < bs; ++i) {
                        mlperf::QuerySampleResponse response { response_ids[i],
                                reinterpret_cast<std::uintptr_t>(&results[idx]),
                                (sizeof(float) * counts[i]) };
                        responses.push_back(response);
                        idx = idx + counts[i];
                    }

                    mlperf::QuerySamplesComplete(responses.data(),
                            responses.size());
                } else {
                    std::vector<unsigned> results;
                    std::vector<Item> outputs;
                    std::vector < mlperf::ResponseId > response_ids;

                    input_.blob_ = getBlob(output_name_);
                    outputs.push_back(input_);

                    postProcessImagenet(outputs, results, response_ids);

                    for (size_t i = 0; i < bs; ++i) {
                        mlperf::QuerySampleResponse response { response_ids[i],
                                reinterpret_cast<std::uintptr_t>(&results[i]),
                                sizeof(results[i]) };
                        responses.push_back(response);
                    }

                    mlperf::QuerySamplesComplete(responses.data(),
                            responses.size());
                }
            }

            callback_queue_(id_, request_, input_);
        }
    }

    void postProcessImagenet(std::vector<Item> &blob,
            std::vector<unsigned> &results,
            std::vector<mlperf::ResponseId> &response_ids) {

        for (size_t i = 0; i < blob.size(); ++i) {
            Item b = blob[i];
            std::vector<unsigned> res;
            TopResults(1, *(b.blob_), res);

            for (size_t j = 0; j < res.size(); ++j) {
                results.push_back(res[j] - 1);
                response_ids.push_back(b.response_ids_[j]);
            }
        }
    }

    void postProcessSSDMobilenet(std::vector<Item> outputs,
            std::vector<float> &results, std::vector<unsigned> &counts,
            std::vector<mlperf::ResponseId> &response_ids, unsigned num_batches,
            unsigned bs) {
        unsigned count = 0;
        int image_id = 0, prev_image_id = 0;
        size_t j = 0;
        for (size_t i = 0; i < num_batches; ++i) {
            Blob::Ptr out = outputs[i].blob_;
            std::vector < mlperf::QuerySampleIndex > sample_idxs =
                    outputs[i].sample_idxs_;
            std::vector < mlperf::QuerySampleIndex > resp_ids =
                    outputs[i].response_ids_;

            const float* detection =
                    static_cast<PrecisionTrait<Precision::FP32>::value_type*>(out->buffer());

            int objectSize = 7;
            int max_proposal_count_ = 100;
            count = 0;
            image_id = 0;
            prev_image_id = 0;
            j = 0;
            for (int curProposal = 0; curProposal < 100; curProposal++) {
                image_id = static_cast<int>(detection[curProposal * objectSize
                        + 0]);

                if (image_id != prev_image_id) {
                    counts.push_back(count * 7);
                    response_ids.push_back(resp_ids[j]);
                    ++j;
                    count = 0;
                    prev_image_id = prev_image_id + 1;
                    if (image_id > 0) {
                        while (image_id != prev_image_id) {
                            counts.push_back(count * 7);
                            response_ids.push_back(resp_ids[j]);
                            ++j;
                            count = 0;
                            prev_image_id = prev_image_id + 1;
                        }
                    } else {
                        while (prev_image_id < (int) bs) {
                            counts.push_back(count * 7);
                            response_ids.push_back(resp_ids[j]);
                            ++j;
                            count = 0;
                            prev_image_id = prev_image_id + 1;
                        }
                    }
                }
                if (image_id < 0) {
                    break;
                }

                float confidence = detection[curProposal * objectSize + 2];
                float label = static_cast<float>(detection[curProposal
                        * objectSize + 1]);
                float xmin = static_cast<float>(detection[curProposal
                        * objectSize + 3]);
                float ymin = static_cast<float>(detection[curProposal
                        * objectSize + 4]);
                float xmax = static_cast<float>(detection[curProposal
                        * objectSize + 5]);
                float ymax = static_cast<float>(detection[curProposal
                        * objectSize + 6]);

                if (confidence > 0.05) {
                    /** Add only objects with >95% probability **/
                    results.push_back(float(sample_idxs[j]));
                    results.push_back(ymin);
                    results.push_back(xmin);
                    results.push_back(ymax);
                    results.push_back(xmax);
                    results.push_back(confidence);
                    results.push_back(label);

                    ++count;
                }

                if (curProposal == (max_proposal_count_ - 1)) {
                    counts.push_back(count * 7);
                    response_ids.push_back(resp_ids[j]);
                    ++j;
                    count = 0;
                    prev_image_id = prev_image_id + 1;
                    while (prev_image_id < (int) bs) {
                        counts.push_back(count * 7);
                        response_ids.push_back(resp_ids[j]);
                        ++j;
                        count = 0;
                        prev_image_id = prev_image_id + 1;
                    }
                }
            }
        }
    }

    void postProcessSSDResnet(std::vector<Item> outputs,
            std::vector<float> &result, std::vector<unsigned> &counts,
            std::vector<mlperf::ResponseId> &response_ids,
            unsigned num_batches) {
        int objectSize = 4;
        int max_proposal_count_ = 200;
        size_t batch_size_ = 1;
        unsigned count = 0;

        for (size_t i = 0; i < num_batches; ++i) {
            Blob::Ptr bbox_blob = outputs[i].blob_;
            Blob::Ptr scores_blob = outputs[i].blob1_;
            Blob::Ptr labels_blob = outputs[i].blob2_;
            std::vector < mlperf::QuerySampleIndex > sample_idxs =
                    outputs[i].sample_idxs_;

            const float* BoundingBoxes =
                    static_cast<float*>(bbox_blob->buffer());
            const float* Confidence = static_cast<float*>(scores_blob->buffer());
            const float* Labels = static_cast<float*>(labels_blob->buffer());

            for (size_t j = 0; j < batch_size_; ++j) {
                auto cur_item = (j * max_proposal_count_);
                auto cur_bbox = (j * max_proposal_count_ * objectSize);

                count = 0;
                for (int curProposal = 0; curProposal < max_proposal_count_;
                        curProposal++) {
                    float confidence = Confidence[cur_item + curProposal];
                    float label =
                            static_cast<int>(Labels[cur_item + curProposal]);
                    float xmin = static_cast<float>(BoundingBoxes[cur_bbox
                            + curProposal * objectSize + 0]);
                    float ymin = static_cast<float>(BoundingBoxes[cur_bbox
                            + curProposal * objectSize + 1]);
                    float xmax = static_cast<float>(BoundingBoxes[cur_bbox
                            + curProposal * objectSize + 2]);
                    float ymax = static_cast<float>(BoundingBoxes[cur_bbox
                            + curProposal * objectSize + 3]);

                    if (confidence > 0.05) {
                        /** Add only objects with > 0.05 probability **/
                        result.push_back(float(sample_idxs[j]));
                        result.push_back(ymin);
                        result.push_back(xmin);
                        result.push_back(ymax);
                        result.push_back(xmax);
                        result.push_back(confidence);
                        result.push_back(label);

                        ++count;
                    }
                }

                counts.push_back(count * 7);
                response_ids.push_back(outputs[i].response_ids_[j]);
            }
        }
    }

    void startAsync() {
        request_.StartAsync();
    }

    void infer() {
        request_.Infer();
    }

    InferenceEngine::Blob::Ptr getBlob(const std::string &name) {
        return request_.GetBlob(name);
    }

    void setInputs(Item input, std::string name) {
        input_ = input;
        request_.SetBlob(name, input_.blob_);
    }

    void setIsWarmup(bool warmup) {
        is_warm_up = warmup;
    }

private:
    InferenceEngine::InferRequest request_;
    size_t id_;
    std::string output_name_;
    mlperf::TestSettings settings_;
    std::string workload_;
    QueueCallbackFunction callback_queue_;
    Item input_;
    bool is_warm_up = false;
};

using namespace InferenceEngine;

class InferRequestsQueue final {
public:
    InferRequestsQueue(InferenceEngine::ExecutableNetwork& net, size_t nireq,
            std::string outName, mlperf::TestSettings settings,
            std::string workload) {
        for (size_t id = 0; id < nireq; id++) {
            requests.push_back(
                    std::make_shared < InferReqWrap
                            > (net, id, outName, settings, workload, std::bind(
                                    &InferRequestsQueue::putIdleRequest, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3)));
            idle_ids_.push(id);
        }

        settings_ = settings;
        out_name_ = outName;
        workload_ = workload;
    }

    ~InferRequestsQueue() = default;

    void putIdleRequest(size_t id, InferenceEngine::InferRequest req,
            Item input) {
        std::unique_lock < std::mutex > lock(mutex_);
        idle_ids_.push(id);
        if ((settings_.scenario == mlperf::TestScenario::Offline)
                || (settings_.scenario == mlperf::TestScenario::MultiStream)) {
            if (workload_.compare("ssd-resnet34") == 0) {
                TBlob < PrecisionTrait < Precision::FP32 > ::value_type > &tblob =
                        dynamic_cast<TBlob<
                                PrecisionTrait<Precision::FP32>::value_type> &>(*(req.GetBlob(
                                "Unsqueeze_bboxes777")));

                InferenceEngine::Blob::Ptr out = make_shared_blob(tblob);

                TBlob < PrecisionTrait < Precision::FP32 > ::value_type
                        > &tblob1 =
                        dynamic_cast<TBlob<
                                PrecisionTrait<Precision::FP32>::value_type> &>(*(req.GetBlob(
                                "Unsqueeze_scores835")));

                InferenceEngine::Blob::Ptr out1 = make_shared_blob(tblob1);

                TBlob < PrecisionTrait < Precision::FP32 > ::value_type
                        > &tblob2 =
                        dynamic_cast<TBlob<
                                PrecisionTrait<Precision::FP32>::value_type> &>(*(req.GetBlob(
                                "Add_labels")));

                InferenceEngine::Blob::Ptr out2 = make_shared_blob(tblob2);

                input.blob_ = out;
                input.blob1_ = out1;
                input.blob2_ = out2;
                outputs_.push_back(input);
            } else {
                TBlob < PrecisionTrait < Precision::FP32 > ::value_type > &tblob =
                        dynamic_cast<TBlob<
                                PrecisionTrait<Precision::FP32>::value_type> &>(*(req.GetBlob(
                                out_name_)));
                InferenceEngine::Blob::Ptr out = make_shared_blob(tblob);
                input.blob_ = out;
                outputs_.push_back(input);

            }
        } /*else {
            TBlob < PrecisionTrait < Precision::FP32 > ::value_type > &tblob =
                    dynamic_cast<TBlob<
                            PrecisionTrait<Precision::FP32>::value_type> &>(*(req.GetBlob(
                            out_name_)));
            InferenceEngine::Blob::Ptr out = make_shared_blob(tblob);

            input.blob_ = out;
            outputs_.push_back(input);
        }*/

        cv_.notify_one();
    }

    InferReqWrap::Ptr getIdleRequest() {
        std::unique_lock < std::mutex > lock(mutex_);
        cv_.wait(lock, [this] {return idle_ids_.size() > 0;});
        auto request = requests.at(idle_ids_.front());
        idle_ids_.pop();
        return request;
    }

    void waitAll() {
        std::unique_lock < std::mutex > lock(mutex_);
        cv_.wait(lock, [this] {return idle_ids_.size() == requests.size();});
    }

    std::vector<Item> getOutputs() {
        return outputs_;
    }

    void reset() {
        outputs_.clear();
    }

    std::vector<InferReqWrap::Ptr> requests;

private:
    std::queue<size_t> idle_ids_;
    std::mutex mutex_;
    std::condition_variable cv_;
    mlperf::TestSettings settings_;
    std::string out_name_;
    std::string workload_;
    std::vector<Item> outputs_;
};
