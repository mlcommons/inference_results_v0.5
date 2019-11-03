#ifndef NMS_F_H
#define NMS_F_H

#include <iostream>
#include <vector>
#include <math.h>
#include <synapse_types.h>
#include <algorithm>
#include <atomic>
#include <HABANA_threadedTask.h>
#include <functional>
#include <omp.h>
#define MAX_BOX_NUM                     (15130UL)
#define MAX_NUM_OF_FINAL_DET            (200U)
#define NUM_OF_CLASSES_MINUS1           (80UL)

#define USE_OPENMP



namespace fastNMS
{
  
    struct sortingPair
    {
        int16_t     m_score;
        uint16_t    m_boxIndex;
    };
    
  
  
    struct DetectionBox 
    {
        float   m_x1;
        float   m_y1;
        float   m_x2;
        float   m_y2;
        int16_t m_score;
        int16_t m_classId;
        float   m_area;  
        DetectionBox() = default;
        //...LG_REVIEW disbale the default ctor
        DetectionBox(sortingPair pairIn, int16_t classIdIn, float *boxPtr, float *xywh_ptr, float scale_xy, float scale_wh) : m_score(pairIn.m_score), m_classId(classIdIn), m_area(0.0)
        {
            float x,y,w,h;
            size_t boxIdx = pairIn.m_boxIndex;
            m_x1 = boxPtr[boxIdx + 0*MAX_BOX_NUM];
            m_y1 = boxPtr[boxIdx + 1*MAX_BOX_NUM];
            m_x2 = boxPtr[boxIdx + 2*MAX_BOX_NUM];
            m_y2 = boxPtr[boxIdx + 3*MAX_BOX_NUM];
            m_x1*=scale_xy;   m_y1*=scale_xy;  m_x2*=scale_wh;  m_y2*=scale_wh;
            x = xywh_ptr[boxIdx*4+0];
            y = xywh_ptr[boxIdx*4+1];
            w = xywh_ptr[boxIdx*4+2];
            h = xywh_ptr[boxIdx*4+3];
            m_x1 = (m_x1*w) + x;
            m_y1 = (m_y1*h) + y;
            m_x2 = expf(m_x2) * w;
            m_y2 = expf(m_y2) * h;

            float left  = m_x1 - 0.5 * m_x2;
            float top   = m_y1 - 0.5 * m_y2;
            float right = m_x1 + 0.5 * m_x2;
            float bott  = m_y1 + 0.5 * m_y2;

            m_x1 = left;
            m_y1 = top;
            m_x2 = right;
            m_y2 = bott;
            m_area = (m_x2-m_x1)*(m_y2-m_y1);
        }

 
        ~DetectionBox() {}
        
    
      
        float intersection_area(const DetectionBox & rhs) const
        {
            float n1 = std::max(rhs.m_x1, m_x1);
            float n2 = std::min(rhs.m_x2, m_x2);
            float m1 = std::max(rhs.m_y1, m_y1);
            float m2 = std::min(rhs.m_y2, m_y2);
            return (std::max(0.0f, (n2 - n1)) * std::max(0.0f, (m2 - m1)));
        }
      
        bool IOU(const  DetectionBox & rhs, float threshold) const
        {

            float inter_area =  intersection_area(rhs);
            float threshold_inter = 1.0f+threshold;
            if (threshold_inter * inter_area < threshold*(m_area + rhs.m_area))
                return false;
            return true;
        }
    };  
    
        
    
        class executeNms
        {
            private:
                sortingPair                                m_classIdSortingVect[MAX_BOX_NUM];
                std::vector<DetectionBox>                  m_detectionPlaceHolder;
                std::vector<DetectionBox *>                m_sortingVector;
                std::vector<DetectionBox*>::iterator       m_lastDet;
                float                                      *m_xywh_ptr;
                float                                      m_scale_xy;
                float                                      m_scale_wh;
                float                                      m_criteria;
                int16_t                                    m_score_threshold;
                uint32_t                                   m_startClassId;
                uint32_t                                   m_endClassId;
                int16_t                                    *m_score_ptr;
                float                                      *m_box_ptr;
                static bool cmpPair(sortingPair l, sortingPair r)
                {
                    return (l.m_score > r.m_score);
                }
                static bool cmpPointer(DetectionBox *l, DetectionBox *r)
                {
                    return (l->m_score > r->m_score);
                }
                uint32_t addnewPairs(int16_t *score_ptr)
                {
                    uint16_t boxNum;
                    uint32_t count=0;
                    int16_t  score_threshold = m_score_threshold;
                    sortingPair *ptr = m_classIdSortingVect;
                    for(boxNum = 0; boxNum < MAX_BOX_NUM;boxNum++)
                    { 
                        if(score_ptr[boxNum] > score_threshold)
                        {
                            ptr->m_score = score_ptr[boxNum];
                            ptr->m_boxIndex = boxNum;
                            ptr++;
                        }
                    }
                    count = std::distance(m_classIdSortingVect,ptr);
                    if(count < 2)
                        return count;
                    std::sort(m_classIdSortingVect,ptr,cmpPair);
                    return std::min(count,MAX_NUM_OF_FINAL_DET);
                }
                void sortAndFixup(int16_t                             classIdIn,
                                float                                 *boxPtr,
                                uint32_t                              numOfSortedElems,
                                std::vector<DetectionBox>::iterator   &end,
                                std::vector<DetectionBox*>::iterator  &end_ptr)
                {
                    std::vector<sortingPair>::iterator   sortedIt;
                    uint32_t                             sortedIndex;
                    float                                 *xywh_ptr = m_xywh_ptr;
                    float                                 scale_xy = m_scale_xy;
                    float                                 scale_wh = m_scale_wh;

                    for(sortedIndex =0; sortedIndex < numOfSortedElems;sortedIndex++)
                    {
                        *end = DetectionBox(m_classIdSortingVect[sortedIndex],classIdIn,boxPtr,xywh_ptr,scale_xy,scale_wh);
                        *end_ptr = &*end;
                        end++;
                        end_ptr++;
                    }
                }

                void classNMS(std::vector<DetectionBox*>::iterator  &start_ptr,
                              std::vector<DetectionBox*>::iterator  &end_ptr)
                {
                    std::vector<DetectionBox*>::iterator check;
                    std::vector<DetectionBox*>::iterator copy;
                    float criteria = m_criteria;
                    while(start_ptr != end_ptr)
                    {
                        check = start_ptr+1;
                        if(check == end_ptr)
                            break;
                        copy =  start_ptr+1; 
                        do
                        {
                            if((*start_ptr)->IOU(**check,criteria) == false)
                            {
                                *copy = *check;
                                copy++;
                            }
                            check++;
                        } while (check != end_ptr);
                        *copy = nullptr;  //debug marker
                        end_ptr = copy;
                        start_ptr++;
                    }  
                    start_ptr = end_ptr;
                }
            public:
                executeNms(size_t   numOfClasses, 
                          float     *xywh_ptr,
                          float     scale_xy,
                          float     scale_wh,
                          float     criteria,
                          int16_t   score_threshold,
                          uint32_t  startClassId,
                          uint32_t  endClassId) 
                {
                    m_detectionPlaceHolder.resize((MAX_NUM_OF_FINAL_DET)*numOfClasses);
                    m_sortingVector.resize((MAX_NUM_OF_FINAL_DET)*numOfClasses);
                    m_lastDet           = m_sortingVector.begin();
                    m_xywh_ptr          = xywh_ptr;
                    m_scale_xy          = scale_xy;
                    m_scale_wh          = scale_wh;
                    m_criteria          = criteria;
                    m_startClassId      = startClassId;
                    m_endClassId        = endClassId;
                    m_score_threshold   = score_threshold;
                    m_box_ptr           = nullptr;
                    m_score_ptr         = nullptr;
                }

                executeNms(executeNms &&other) : m_detectionPlaceHolder(std::move(other.m_detectionPlaceHolder)),
                                                 m_sortingVector(std::move(other.m_sortingVector))
                                                 
                {
                    memcpy(m_classIdSortingVect,other.m_classIdSortingVect, MAX_BOX_NUM*sizeof(sortingPair));
                    m_xywh_ptr          = other.m_xywh_ptr;
                    m_box_ptr           = other.m_box_ptr;
                    m_criteria          = other.m_criteria;
                    m_endClassId        = other.m_endClassId;
                    m_lastDet           = other.m_lastDet;
                    m_scale_wh          = other.m_scale_wh;
                    m_scale_xy          = other.m_scale_xy;
                    m_score_ptr         = other.m_score_ptr;
                    m_score_threshold   = other.m_score_threshold;
                    m_startClassId      = other.m_startClassId;
                }
                executeNms(const executeNms& other)             = delete;
                executeNms& operator=(const executeNms & other) = delete;
                
                void setParamsForProcess(int16_t    *score_ptr,
                                         float      *box_ptr)
                {
                    m_score_ptr     =   score_ptr;
                    m_box_ptr       =   box_ptr;
                }

                void processClasses()
                {
                    size_t classId;
                    size_t boxNum;
                    std::vector<DetectionBox>::iterator  end         = m_detectionPlaceHolder.begin();
                    std::vector<DetectionBox*>::iterator start_ptr  = m_sortingVector.begin();
                    std::vector<DetectionBox*>::iterator end_ptr    = m_sortingVector.begin();
                    uint32_t                             numOfElem;
                    m_lastDet = m_sortingVector.begin();
                    for(classId = m_startClassId; classId < m_endClassId; classId++)
                    {
                            numOfElem = addnewPairs(&m_score_ptr[classId*MAX_BOX_NUM]);
                            if(numOfElem > 0)
                            {
                                sortAndFixup(classId,m_box_ptr, numOfElem,end, end_ptr);
                                //*end_ptr = nullptr; //debug marker
                                classNMS(start_ptr, end_ptr);
                            }
                    }
                    m_lastDet = end_ptr;
                    
                }
                bool selfSort(std::vector<DetectionBox>  &result)
                {
                    int32_t i;
                    if(m_lastDet == m_sortingVector.begin())
                        return false;
                    std::vector<DetectionBox*>::iterator end_ptr    = m_lastDet, start_ptr;
                    m_lastDet = m_sortingVector.begin();   
                    std::stable_sort(m_sortingVector.begin(),end_ptr,cmpPointer);
                    for(i=0,start_ptr = m_sortingVector.begin();(i < MAX_NUM_OF_FINAL_DET) && (start_ptr != end_ptr);start_ptr++,i++)
                    {
                        result.push_back(**start_ptr);
                    }
                    return true;
                }

                void extMerge(std::vector<DetectionBox*> &mergeVector)
                {
                    if(m_lastDet == m_sortingVector.begin())
                        return;
                    std::vector<DetectionBox*>::iterator copyIt, endPtr = m_lastDet;
                    m_lastDet = m_sortingVector.begin();
                    for(copyIt = m_sortingVector.begin(); copyIt != endPtr;copyIt++)
                    {
                        mergeVector.push_back(*copyIt);
                    }
                }
                uint32_t getSortingBufSize()
                {
                    return std::distance(m_sortingVector.begin(),m_lastDet);
                }
        };

        class nmsThreadRunnerOmp
        {
            public:
                nmsThreadRunnerOmp(uint32_t  maxThreadNum,
                                float   *xywh_ptr,
                                float   scale_xy,
                                float   scale_wh,
                                float   criteria,
                                int16_t score_threshold)
                {
                    uint32_t step = NUM_OF_CLASSES_MINUS1/maxThreadNum;
                    m_partialClassNmsDecoders.reserve(maxThreadNum);

                    for(uint32_t i = 0; i < maxThreadNum;i++) 
                    {
                        m_partialClassNmsDecoders.emplace_back(step,xywh_ptr,scale_xy,scale_wh,criteria,score_threshold,i * step + 1,(i+1) *step + 1);
                    }                
                    m_mergedSortingVect.reserve(MAX_NUM_OF_FINAL_DET*NUM_OF_CLASSES_MINUS1);
                }
                nmsThreadRunnerOmp(const nmsThreadRunnerOmp&)              = delete;
                nmsThreadRunnerOmp& operator=(const nmsThreadRunnerOmp &)  = delete;
                static bool cmp(DetectionBox *l, DetectionBox *r)
                {
                    return (l->m_score > r->m_score);
                }
                void multiThreadedNmsExecution(int16_t                    *score_ptr,
                                               float                      *box_ptr,
                                               std::vector<DetectionBox>  &finalRes)   
                {   
        
                    m_mergedSortingVect.clear();                    
                    size_t numOfThreads = m_partialClassNmsDecoders.size();
                    if(numOfThreads > 1)
                    {
                        
                        for(int i=0; i < numOfThreads;i++)
                            m_partialClassNmsDecoders[i].setParamsForProcess(score_ptr,box_ptr);
                        #pragma omp parallel
                        {
                            size_t tid = omp_get_thread_num(); 
                            m_partialClassNmsDecoders[tid].processClasses();   
                        }
                        std::vector<DetectionBox*>::iterator sortingVecIt = m_mergedSortingVect.begin();                        
                        for(int i=0;i < numOfThreads;i++)
                        {
                            if(m_partialClassNmsDecoders[i].getSortingBufSize() > 0)
                                m_partialClassNmsDecoders[i].extMerge(m_mergedSortingVect);       
                        }
                        if(m_mergedSortingVect.size() > 0)
                        {
                            uint32_t  i;
                            std::stable_sort(m_mergedSortingVect.begin(),m_mergedSortingVect.end(),cmp);
                            for(i=0,sortingVecIt = m_mergedSortingVect.begin();((i < MAX_NUM_OF_FINAL_DET) && (sortingVecIt != m_mergedSortingVect.end()));sortingVecIt++,i++)
                            {
                                finalRes.push_back(**sortingVecIt);
                            }
                        }
                    }
                    else
                    {
                        m_partialClassNmsDecoders[0].setParamsForProcess(score_ptr,box_ptr);
                        m_partialClassNmsDecoders[0].processClasses();
                        m_partialClassNmsDecoders[0].selfSort(finalRes);
                    }
                }
                    
           private:
                    
        
                    std::vector<executeNms>                                             m_partialClassNmsDecoders;
                    std::vector<DetectionBox *>                                         m_mergedSortingVect;
        };


        class nmsThreadRunnerTsk
        {
            public:
                nmsThreadRunnerTsk(uint32_t  maxThreadNum,
                                float   *xywh_ptr,
                                float   scale_xy,
                                float   scale_wh,
                                float   criteria,
                                int16_t score_threshold)
                {
                    uint32_t step = NUM_OF_CLASSES_MINUS1/maxThreadNum;
                    m_partialClassNmsDecoders.reserve(maxThreadNum);

                    for(uint32_t i = 0; i < maxThreadNum;i++) 
                    {
                        m_partialClassNmsDecoders.emplace_back(step,xywh_ptr,scale_xy,scale_wh,criteria,score_threshold,i * step + 1,(i+1) *step + 1);
                    }                
                    m_mergedSortingVect.reserve(MAX_NUM_OF_FINAL_DET*NUM_OF_CLASSES_MINUS1);
                    if(maxThreadNum > 1)
                    {
                        m_threadPool.reserve(maxThreadNum);
                        for(uint32_t i = 0; i < maxThreadNum;i++) 
                        {

                            m_processFunctors.push_back([i,this](){this->m_partialClassNmsDecoders[i].processClasses();});
                            m_postProcFunctors.push_back([this](){this->decDoneCnt();});
                        }
                        for(uint32_t i = 0; i < maxThreadNum;i++) 
                        {

                            m_threadPool.emplace_back(m_processFunctors[i],m_postProcFunctors[i]);
                        }
                    
                   }
                }
                
                ~nmsThreadRunnerTsk()
                {
                    while (m_threadPool.size() > 0)
                    {
                        m_threadPool.pop_back();
                    } 
                    while (m_processFunctors.size() > 0)
                    {
                        m_processFunctors.pop_back();
                    } 
                    while (m_postProcFunctors.size() > 0)
                    {
                        m_postProcFunctors.pop_back();
                    } 
                }
                nmsThreadRunnerTsk(nmsThreadRunnerTsk &&other)              =   delete;
                nmsThreadRunnerTsk& operator=(nmsThreadRunnerTsk &&other)   =   delete;
                nmsThreadRunnerTsk()                                        =   delete;
                nmsThreadRunnerTsk(const nmsThreadRunnerTsk&)               =   delete;
                nmsThreadRunnerTsk& operator=(const nmsThreadRunnerTsk &)   =   delete;
                static bool cmp(DetectionBox *l, DetectionBox *r)
                {
                    return (l->m_score > r->m_score);
                }
                void multiThreadedNmsExecution(int16_t                    *score_ptr,
                                               float                      *box_ptr,
                                               std::vector<DetectionBox>  &finalRes)   
                {   
                    m_mergedSortingVect.clear();
                    size_t numOfThreads = m_threadPool.size();
                    
                    if(numOfThreads > 1)
                    {
                        for(int32_t i = 0; i < numOfThreads;i++)
                        {
                            
                            m_partialClassNmsDecoders[i].setParamsForProcess(score_ptr,box_ptr);
                        }
                        m_doneCnt = numOfThreads;
                        for(int32_t tid = 0; tid < numOfThreads;tid++)
                        {
                            
                            m_threadPool[tid].sendTask();
                        }
                        while(m_doneCnt > 0);
                       
                        
                        std::vector<DetectionBox*>::iterator sortingVecIt = m_mergedSortingVect.begin();
                        
                        for(int i=0;i < numOfThreads;i++)
                        {
                            if(m_partialClassNmsDecoders[i].getSortingBufSize() > 0)
                                m_partialClassNmsDecoders[i].extMerge(m_mergedSortingVect);       
                        }
                        if(m_mergedSortingVect.size() > 0)
                        {
                            std::stable_sort(m_mergedSortingVect.begin(),m_mergedSortingVect.end(),cmp);
                            uint32_t  i;
                            for(i=0,sortingVecIt = m_mergedSortingVect.begin();((i < MAX_NUM_OF_FINAL_DET) && (sortingVecIt != m_mergedSortingVect.end()));sortingVecIt++,i++)
                            {
                                    finalRes.push_back(**sortingVecIt);
                            }
                        }
                    }
                    else
                    {
                        m_partialClassNmsDecoders[0].setParamsForProcess(score_ptr,box_ptr);
                        m_partialClassNmsDecoders[0].processClasses();
                        m_partialClassNmsDecoders[0].selfSort(finalRes);
                    }
                }
                    
           private:
                    void decDoneCnt() 
                    {
                         m_doneCnt--;
                    }
                    std::atomic<uint32_t>                                                                       m_doneCnt;
                    std::vector<std::function<void(void)>>                                                      m_processFunctors;
                    std::vector<std::function<void(void)>>                                                      m_postProcFunctors;
                    std::vector<ThrTask::ThreadedTask<std::function<void(void)>,std::function<void(void)>>>     m_threadPool; 
                    std::vector<executeNms>                                                                     m_partialClassNmsDecoders;
                    std::vector<DetectionBox *>                                                                 m_mergedSortingVect;
        };
  
};  
#endif
