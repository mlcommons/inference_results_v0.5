#ifndef ITEM_H__
#define ITEM_H__

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include <inference_engine.hpp>
#include <ie_blob.h>

using namespace InferenceEngine;

class Item {
public:
    Item(Blob::Ptr blob, std::vector<mlperf::ResponseId> response_ids,
            std::vector<mlperf::QuerySampleIndex> sample_idxs) :
            blob_(blob), response_ids_(response_ids), sample_idxs_(sample_idxs) {
    }
    Item(Blob::Ptr blob, std::vector<mlperf::QuerySampleIndex> sample_idxs) :
            blob_(blob),
            sample_idxs_(sample_idxs) {
    }
    Item() {
    }
public:
    Blob::Ptr blob_;
    std::vector<mlperf::ResponseId> response_ids_;
    std::vector<mlperf::QuerySampleIndex> sample_idxs_;
    Blob::Ptr blob1_;
    Blob::Ptr blob2_;

};

#endif
