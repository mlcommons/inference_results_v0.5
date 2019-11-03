#ifndef _TRANS_OPERATORS_FUNCTORS_
#define _TRANS_OPERATORS_FUNCTORS_
#include <HABANA_transProcessBaseFunctors.h>
#include <vector>
#include <query_sample.h>
#include <HABANA_nms_class.h>

#define BOXES_BUF_LEN                   (15130UL)
#define NUM_OF_CATEGORIES               (81UL)
#define MAX_OUTPUTS                     (200UL)
#define MAX_NUM                         (200UL)
#define MAX_INT16_IN_FLOAT              32768.0f
#define NUM_OF_ELEMENTS_IN_SSD_RESULT   (7UL)


uint16_t inv_map[81] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                        44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
                        75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,90};

class resnetTransOperator : public baseFunctor
{
    public:
        resnetTransOperator() {}
        virtual ~resnetTransOperator() = default;
        virtual bool operator()(std::vector<uint32_t> &sampleResCnt, uint32_t numOfSamples,EnqueueTensorInfo *resOutput, float *transResOutputIn[], std::vector<mlperf::QuerySample> &querySamples) override
        {
             int i;
        float *transResOutput = transResOutputIn[0];
        uint8_t *resRawdata = reinterpret_cast<uint8_t*>(resOutput->pTensorData);
        for(i = 0; i < numOfSamples;i++)
        {
            // the results is casted to int since resnet argmax returns int type
            *transResOutput = static_cast<float> ((*((int*)resRawdata))-1);
            sampleResCnt.push_back(1);
            transResOutput++;
            resRawdata+=4;
        }
        return true;
        }
};


class ssdTransOperatorSSDOmp : public baseFunctor
{
    public:
        ssdTransOperatorSSDOmp() = delete;
        ssdTransOperatorSSDOmp(size_t numOfFinalDet , 
                                 size_t numOfThreads,
                                float   *xywh_ptr,
                                float   scale_xy,
                                float   scale_wh,
                                float   criteria,
                                int16_t score_threshold) :m_nsmThreadRunner(numOfThreads,xywh_ptr,scale_xy,scale_wh,criteria,score_threshold)
        {
            m_resVec.reserve(numOfFinalDet);
        }
        virtual ~ssdTransOperatorSSDOmp()=default;
        virtual bool operator()(std::vector<uint32_t> &sampleResCnt, uint32_t numOfSamples,EnqueueTensorInfo *resOutput, float *transResOutputIn[], std::vector<mlperf::QuerySample> &querySamples) override
        {
            std::vector<fastNMS::DetectionBox>::iterator it;
            float *transResOutput = transResOutputIn[0];
            int16_t *scorePtr = reinterpret_cast<int16_t*>(resOutput[1].pTensorData);
            float   *boxesPtr = reinterpret_cast<float*>(resOutput[0].pTensorData);
            for(int i = 0; i < numOfSamples;i++)
            {
                
                
                m_resVec.clear();
                m_nsmThreadRunner.multiThreadedNmsExecution(scorePtr,
                                                            boxesPtr,
                                                            m_resVec);
                sampleResCnt.push_back(m_resVec.size()*NUM_OF_ELEMENTS_IN_SSD_RESULT);
                scorePtr += (MAX_BOX_NUM*NUM_OF_CATEGORIES);
                boxesPtr += (MAX_BOX_NUM*4);

                for(it = m_resVec.begin(); it != m_resVec.end(); it++)
                {
                    *transResOutput++ = static_cast<float>(querySamples[i].index);
                    *transResOutput++ = it->m_y1;
                    *transResOutput++ = it->m_x1;
                    *transResOutput++ = it->m_y2;
                    *transResOutput++ = it->m_x2;
                    *transResOutput++ = (static_cast<float>(it->m_score)/MAX_INT16_IN_FLOAT);
                    *transResOutput++ = static_cast<float>(inv_map[it->m_classId]);
                    
                }
            }
        
            return true;  
        }
    private:
       
       std::vector<fastNMS::DetectionBox>            m_resVec;
       fastNMS::nmsThreadRunnerOmp                   m_nsmThreadRunner;
};

class ssdTransOperatorSSDTsk : public baseFunctor
{
    public:
        ssdTransOperatorSSDTsk() = delete;
        ssdTransOperatorSSDTsk(size_t numOfFinalDet , 
                                 size_t numOfThreads,
                                float   *xywh_ptr,
                                float   scale_xy,
                                float   scale_wh,
                                float   criteria,
                                int16_t score_threshold) :m_nsmThreadRunner(numOfThreads,xywh_ptr,scale_xy,scale_wh,criteria,score_threshold)
        {
            m_resVec.reserve(numOfFinalDet);
        }
        virtual ~ssdTransOperatorSSDTsk()=default;
        virtual bool operator()(std::vector<uint32_t> &sampleResCnt, uint32_t numOfSamples,EnqueueTensorInfo *resOutput, float *transResOutputIn[], std::vector<mlperf::QuerySample> &querySamples) override
        {
            std::vector<fastNMS::DetectionBox>::iterator it;
            float *transResOutput = transResOutputIn[0];
            int16_t *scorePtr = reinterpret_cast<int16_t*>(resOutput[1].pTensorData);
            float   *boxesPtr = reinterpret_cast<float*>(resOutput[0].pTensorData);
            for(int i = 0; i < numOfSamples;i++)
            {
                
                
                m_resVec.clear();
                m_nsmThreadRunner.multiThreadedNmsExecution(scorePtr,
                                                            boxesPtr,
                                                            m_resVec);
                sampleResCnt.push_back(m_resVec.size()*NUM_OF_ELEMENTS_IN_SSD_RESULT);
                scorePtr += (MAX_BOX_NUM*NUM_OF_CATEGORIES);
                boxesPtr += (MAX_BOX_NUM*4);

                for(it = m_resVec.begin(); it != m_resVec.end(); it++)
                {
                    *transResOutput++ = static_cast<float>(querySamples[i].index);
                    *transResOutput++ = it->m_y1;
                    *transResOutput++ = it->m_x1;
                    *transResOutput++ = it->m_y2;
                    *transResOutput++ = it->m_x2;
                    *transResOutput++ = (static_cast<float>(it->m_score)/MAX_INT16_IN_FLOAT);
                    *transResOutput++ = static_cast<float>(inv_map[it->m_classId]);
                    
                }
            }
        
            return true;  
        }
    private:
       
       std::vector<fastNMS::DetectionBox>            m_resVec;
       fastNMS::nmsThreadRunnerTsk                   m_nsmThreadRunner;
};



#endif