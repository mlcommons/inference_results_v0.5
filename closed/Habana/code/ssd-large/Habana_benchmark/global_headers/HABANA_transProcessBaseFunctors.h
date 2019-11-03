#ifndef _TRANS_OPERATORS_BASE_FUNCTORS_
#define _TRANS_OPERATORS_BASE_FUNCTORS_
#include <vector>
#include <query_sample.h>
#include <synapse_types.h>






class baseFunctor
{
    public:
    baseFunctor() {}
    virtual ~baseFunctor(){}
    virtual bool operator()(std::vector<uint32_t>               &sampleResCnt,
                            uint32_t                            numOfSamples,
                            EnqueueTensorInfo                   *resOutput,
                            float                               *transResOutputIn[],
                            std::vector<mlperf::QuerySample>    &querySamples) =0;
};
#endif