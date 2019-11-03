#include <HABANA_runner_task.h>
#include <map>
#include <HABANA_exception.h>
#include <synapse.h>
Habana_taskInfo::Habana_taskInfo(bool                           isEnforcedResBuf,
                                 uint64_t                       enforcedResBufByteSize,
                                 uint32_t                       deviceId,
                                 uint64_t                       batchSize,
                                 uint32_t                       numOfOutputs,
                                 TensorMetadataInfo             outputTensorInfo[],
                                 std::unique_ptr<baseFunctor>   postProcessFunctor,
                                 const char *                   inputName,
                                 uint64_t                       inputSizeInBytes)
{
        bool outputAllocStatus=true;
        synStatus status;
        uint64_t outputTensorByteSize, floatOutputResBuffSize;
         void *synAllocPtr;
        std::map<synDataType, uint64_t> dataTypeToNumBytes 
        {
            {syn_type_fixed,  1},
            {syn_type_uint8,  1},
            {syn_type_int16,  2},
            {syn_type_int32,  4},
            {syn_type_single, 4}
        };
        m_deviceID      = deviceId;
        m_numOfOutputs  = numOfOutputs;

        m_enqueueInputTensor.tensorName = inputName;
        m_enqueueInputTensor.tensorSize = batchSize*inputSizeInBytes;
        m_inputSize = inputSizeInBytes;
      
        // calculate all output tensor byte sizes and update all tasl info structures
        for(int i = 0; i < numOfOutputs;i++)
        {
            outputTensorByteSize = 1;
            floatOutputResBuffSize = 0;
            for(int j = 0; j < outputTensorInfo[i].dimensions-1;j++)
            {
              outputTensorByteSize *= outputTensorInfo[i].dimensionsSize[j];
            }
            m_rawOutputSize[i] = outputTensorByteSize*dataTypeToNumBytes[(synDataType)outputTensorInfo[i].elementType];
            m_floatOutputSize[i] = outputTensorByteSize*sizeof(float);
            floatOutputResBuffSize += outputTensorByteSize*batchSize*sizeof(float);
            outputTensorByteSize *= (batchSize * dataTypeToNumBytes[(synDataType)outputTensorInfo[i].elementType]);
            m_enqueueOutputTensors[i].tensorName    = outputTensorInfo[i].tensorName;
            m_enqueueOutputTensors[i].tensorSize    = outputTensorByteSize;
            m_enqueueOutputTensors[i].pTensorData   = nullptr; //mark this output as not allocated
            m_floatOutputsResultBufferSize[i]       = floatOutputResBuffSize;
        }

      
      
        try
        {
            //reserve all task vectors
            m_querySamplesResponseBuff.reserve(batchSize);
            m_resPartitions.reserve(batchSize);
            m_querySamples.reserve(batchSize);
        }
      

        catch (const std::exception &e) 
        {
            throw HABANA_benchException("Habana_taskInfo: can't allocate Habana_taskInfo vectors");
        }
      
        for(int i = 0; i < numOfOutputs ;i++)
        {
            status = synMalloc(deviceId, m_enqueueOutputTensors[i].tensorSize, synMemHost, &synAllocPtr);
            m_enqueueOutputTensors[i].pTensorData = static_cast<char *>(synAllocPtr);
            if (status != synSuccess) 
            {
                m_enqueueOutputTensors[i].pTensorData = nullptr;
                outputAllocStatus = false;
                break;
            }
            
            if(outputAllocStatus)
            {
                if(isEnforcedResBuf)
                {
                    status = synMalloc(deviceId, batchSize*enforcedResBufByteSize, synMemHost,&synAllocPtr);
                    m_floatTranslationbuffer[i] = static_cast<float*>(synAllocPtr);
                    if (status != synSuccess) 
                    {
                        m_floatTranslationbuffer[i] = nullptr;
                        outputAllocStatus = false;
                        break;
                    }                    
                }
                else
                {
                    status = synMalloc(deviceId, m_floatOutputsResultBufferSize[i], synMemHost,&synAllocPtr);
                    m_floatTranslationbuffer[i] = static_cast<float*>(synAllocPtr);
                    if (status != synSuccess) 
                    {
                        m_floatTranslationbuffer[i] = nullptr;
                        outputAllocStatus = false;
                        break;
                    }                    
                }
            }
        }
        if (outputAllocStatus == false) 
        {
               for(int i = 0; i < numOfOutputs;i++)
                {
                    if(m_enqueueOutputTensors[i].pTensorData != nullptr)
                    {
                        synFree(deviceId,m_enqueueOutputTensors[i].pTensorData,synMemHost);
                        m_enqueueOutputTensors[i].pTensorData = nullptr;
                    }
                    if(m_floatTranslationbuffer[i] != nullptr)
                    {
                        synFree(deviceId,m_floatTranslationbuffer[i],synMemHost);
                        m_floatTranslationbuffer[i] = nullptr;
                    }
                } 
                throw HABANA_benchException("Habana_taskInfo: can't create output buffer in taskinfo structure");
        }
        m_postProcFactorPtr = std::move(postProcessFunctor);
}
Habana_taskInfo::Habana_taskInfo(Habana_taskInfo &&other) : m_postProcFactorPtr(std::move(other.m_postProcFactorPtr)),
                                                            m_querySamples(std::move(other.m_querySamples)),
                                                            m_querySamplesResponseBuff(std::move(other.m_querySamplesResponseBuff)),
                                                            m_resPartitions(std::move(other.m_resPartitions))

{
    m_deviceID      = other.m_deviceID;
    m_numOfOutputs  = other.m_numOfOutputs;
    m_enqueueHandle             = other.m_enqueueHandle;
    m_enqueueInputTensor        = other.m_enqueueInputTensor;
    m_inputSize                 = other.m_inputSize;
    for(int32_t i=0;i<MAX_NUM_OUTPUT_TENSORS;i++)
    {
        m_enqueueOutputTensors[i]           = other.m_enqueueOutputTensors[i];
        m_floatOutputsResultBufferSize[i]   = other.m_floatOutputsResultBufferSize[i];
        m_floatTranslationbuffer[i]         = other.m_floatTranslationbuffer[i];
        m_rawOutputSize[i]                  = other.m_rawOutputSize[i];
        m_floatOutputSize[i]                = other.m_floatOutputSize[i];   

        other.m_enqueueOutputTensors[i].pTensorData = nullptr;
        other.m_floatTranslationbuffer[i] = nullptr;
    }
}
Habana_taskInfo& Habana_taskInfo::operator=(Habana_taskInfo &&other)
{

    m_postProcFactorPtr         = std::move(other.m_postProcFactorPtr);
    m_querySamples              = std::move(other.m_querySamples);
    m_querySamplesResponseBuff  = std::move(other.m_querySamplesResponseBuff);
    m_resPartitions             = std::move(other.m_resPartitions);
    m_deviceID                  = other.m_deviceID;
    m_numOfOutputs              = other.m_numOfOutputs;
    m_enqueueHandle             = other.m_enqueueHandle;
    m_enqueueInputTensor        = other.m_enqueueInputTensor;
    for(int32_t i=0;i<MAX_NUM_OUTPUT_TENSORS;i++)
    {
        m_enqueueOutputTensors[i]           = other.m_enqueueOutputTensors[i];
        m_floatOutputsResultBufferSize[i]   = other.m_floatOutputsResultBufferSize[i];
        m_floatTranslationbuffer[i]         = other.m_floatTranslationbuffer[i];
        m_rawOutputSize[i]                  = other.m_rawOutputSize[i];
        m_floatOutputSize[i]                = other.m_floatOutputSize[i];    
        other.m_enqueueOutputTensors[i].pTensorData = nullptr;
        other.m_floatTranslationbuffer[i] = nullptr;
    }
    return *this;
}
Habana_taskInfo::~Habana_taskInfo()
{
    if(m_postProcFactorPtr.get() != nullptr)
    {
        m_postProcFactorPtr.reset(nullptr); //first release the function which will close all threads using the allocated memory
    }
    for(int i = 0; i <m_numOfOutputs;i++)
    {
        if(m_enqueueOutputTensors[i].pTensorData != nullptr)
           synFree(m_deviceID, m_enqueueOutputTensors[i].pTensorData, synMemHost);
        if(m_floatTranslationbuffer[i] != nullptr)
           synFree(m_deviceID, m_floatTranslationbuffer[i], synMemHost);
    }
    //all other automatic variables will be destructed by their own destructors.
}

    void Habana_taskInfo::updateSizeIO(uint32_t newBatchSize, char *ptr)
    {
        m_enqueueInputTensor.tensorSize = newBatchSize*m_inputSize;
        if(m_enqueueInputTensor.tensorSize == 0)
            newBatchSize = newBatchSize;
        m_enqueueInputTensor.pTensorData = ptr;
        for(int i = 0; i < m_numOfOutputs;i++)
        {
            m_enqueueOutputTensors[i].tensorSize    =  m_rawOutputSize[i]*newBatchSize;
            m_floatOutputsResultBufferSize[i]       =  m_floatOutputSize[i]*newBatchSize;
        }
    }
