#include <HABANA_batchifierDataNode.h>

    Hababan_batchifierData::Hababan_batchifierData(char         *allocPtr,
                                                    uint32_t     imageSizeInBytes,
                                                    uint32_t     maxNumOfImages,
                                                    std::chrono::duration<int,std::micro> expectedImageProcTime,
                                                    std::chrono::duration<int,std::micro> procSlotTimeDuration) : m_lockingMutex(new std::mutex)
    {
        m_maxImagesInBuffer     = maxNumOfImages;
        m_cpyStartBuffer        = allocPtr;
        m_cpyCurrPos            = m_cpyStartBuffer;
        m_imageSizeInBytes      = imageSizeInBytes;
        m_numOfBytesInBuffer    = 0;
        m_numOfImagesInBuffer   = 0;
        m_imageExpectedProcTime = expectedImageProcTime;
        m_processSlotDuration   = procSlotTimeDuration;
         
    }
    void Hababan_batchifierData::lock()
    {
        m_lockingMutex->lock();
    }
    void Hababan_batchifierData::unlock()
    {
        m_lockingMutex->unlock();
    }
    bool Hababan_batchifierData::isBatchDone()
    {
            std::chrono::time_point<std::chrono::steady_clock> currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<int,std::micro> copyDuration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - m_batchStartTime);
            return ((m_numOfBytesInBuffer == m_maxImagesInBuffer) || ((m_processSlotDuration - copyDuration) <= (m_imageExpectedProcTime*(m_numOfImagesInBuffer))));
    }
    bool Hababan_batchifierData::isBatchOverdue()
    {
            std::chrono::time_point<std::chrono::steady_clock> currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<int,std::micro> copyDuration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - m_batchStartTime);
            return ((m_processSlotDuration <= copyDuration) && (m_numOfImagesInBuffer > 0));
    }
    bool Hababan_batchifierData::copyImage(char *imagePtr)
    {
        if(m_numOfImagesInBuffer < m_maxImagesInBuffer)
        {
            if(m_numOfImagesInBuffer == 0)
                m_batchStartTime = std::chrono::steady_clock::now();
            memcpy(m_cpyCurrPos,imagePtr,m_imageSizeInBytes);
            m_cpyCurrPos += m_imageSizeInBytes;
            m_numOfBytesInBuffer += m_imageSizeInBytes;
            m_numOfImagesInBuffer++;
        }
        return isBatchDone();
    }
    void Hababan_batchifierData::clear()
    {
        m_cpyCurrPos            = m_cpyStartBuffer;
        m_numOfImagesInBuffer   = 0;
        m_numOfBytesInBuffer    = 0;
    }
    char * Hababan_batchifierData::get_ptr()
    {
        return m_cpyStartBuffer;
    }
    uint32_t Hababan_batchifierData::getNumOfImages()
    {
        return m_numOfImagesInBuffer;
    }
