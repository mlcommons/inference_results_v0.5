#ifndef _RUNNER_TASK_INFO_
#define _RUNNER_TASK_INFO_
#include <query_sample.h>
#include <query_sample_library.h>
#include <synapse_types.h>
#include <HABANA_transProcessBaseFunctors.h>

#include <HABANA_global_definitions.h>

struct Habana_taskInfo
    {
        synWaitHandle                                   m_enqueueHandle;                                    // handle to an enqueue task in the Habana HW, this is used to wait for task to end
        std::vector<mlperf::QuerySample>                m_querySamples;                                     // vector contains the query sample descriptors of this task
        std::vector<uint32_t>                           m_resPartitions;                                    // vector contains number of elements in the result of each sample
        std::vector<mlperf::QuerySampleResponse>        m_querySamplesResponseBuff;                         // query response buffer that will be returned to the loadgen
        EnqueueTensorInfo                               m_enqueueInputTensor;
        EnqueueTensorInfo                               m_enqueueOutputTensors[MAX_NUM_OUTPUT_TENSORS];
        float                                           *m_floatTranslationbuffer[MAX_NUM_OUTPUT_TENSORS];  // output buffer containing results translated into floats.
                                                                                                            // loadgen expects that each element in the result will be translated into floats
        uint64_t                                        m_floatOutputsResultBufferSize[MAX_NUM_OUTPUT_TENSORS]; 
        std::unique_ptr<baseFunctor>                    m_postProcFactorPtr;
        uint32_t                                        m_deviceID;
        uint32_t                                        m_numOfOutputs;
        uint64_t                                        m_inputSize;
        uint64_t                                        m_rawOutputSize[MAX_NUM_OUTPUT_TENSORS];
        uint64_t                                        m_floatOutputSize[MAX_NUM_OUTPUT_TENSORS];
        Habana_taskInfo(bool                           isEnforcedResBuf,
                        uint64_t                       enforcedResBufByteSize,
                        uint32_t                       deviceId,
                        uint64_t                       batchSize,
                        uint32_t                       numOfOutputs,
                        TensorMetadataInfo             outputTensorInfo[],
                        std::unique_ptr<baseFunctor>   postProcessFunctor,
                        const char *                   inputName,
                        uint64_t                       inputSizeInBytes);
        Habana_taskInfo(Habana_taskInfo &&);
        Habana_taskInfo& operator= (Habana_taskInfo &&);
        ~Habana_taskInfo();
        void updateSizeIO(uint32_t newBatchSize, char *ptr);
    };    



#endif