#ifndef __BATCHIFIER_DATA__
#define __BATCHIFIER_DATA__
#include <stdint.h>
#include <cstring>
#include <chrono>
#include <mutex>
#include <memory>
#include <HABANA_global_definitions.h>

struct Hababan_batchifierData
{
    char        *m_cpyStartBuffer;
    char        *m_cpyCurrPos;
    uint32_t    m_imageSizeInBytes;
    uint32_t    m_numOfImagesInBuffer;
    uint32_t    m_numOfBytesInBuffer;
    uint32_t    m_maxImagesInBuffer;
    std::unique_ptr<std::mutex>                         m_lockingMutex;
    std::chrono::time_point<std::chrono::steady_clock>   m_batchStartTime;
    std::chrono::duration<int,std::micro>                m_imageExpectedProcTime;
    std::chrono::duration<int,std::micro>                m_processSlotDuration;
    Hababan_batchifierData(char         *allocPtr,
                           uint32_t     imageSizeInBytes,
                           uint32_t     maxNumOfImages,
                           std::chrono::duration<int,std::micro> expectedImageProcTime,
                           std::chrono::duration<int,std::micro> procSlotTimeDuration);
    void lock();
    void unlock();
    bool isBatchDone();
    bool isBatchOverdue();
    bool copyImage(char *imagePtr);
    void clear();
    char * get_ptr();
    uint32_t getNumOfImages();
};
#endif