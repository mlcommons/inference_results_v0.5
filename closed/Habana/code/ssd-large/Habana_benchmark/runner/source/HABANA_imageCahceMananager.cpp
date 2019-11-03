#include <HABANA_imageCacheManager.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace HABANA_bench
{
    //#define DEBUG_IMAGE_LOAD

    //******************************************************************************************************
    //******************************************************************************************************
    // Image Manager
    //******************************************************************************************************
    //******************************************************************************************************


    HABANA_imageCacheManager::HABANA_imageCacheManager(const std::string &imageCachePath,
                                                    const std::string &imageNameList,
                                                    uint64_t           maxQuerySize,
                                                    uint64_t           numOfImagesToload,
                                                    uint64_t           imageHeight,
                                                    uint64_t           imageWidth,
                                                    uint64_t           imageDepth,
                                                    uint8_t            *queryBuf) : m_imageCacheDirPath(imageCachePath),
                                                                                    m_queryBuffer(queryBuf),
                                                                                    m_numOfImagesToload(numOfImagesToload),
                                                                                    m_maxquerySize(maxQuerySize),
                                                                                    m_imageSizeInBytes(imageHeight * imageWidth * imageDepth)
    {

        std::ifstream nameListIO;
        std::string fullListName = imageCachePath + "/" + imageNameList; // full path to image list file
        nameListIO.open(fullListName.c_str());
        if (!nameListIO.is_open()) // check that image names file exists
        {
            std::string error("imageCacheManager: can't open name list file: ");
            error = error + imageNameList;
            throw HABANA_benchException(error.c_str());
        }
        std::string image_file_name;
        unsigned int label;
                                                                                                                // map
        m_indexImageMap.reserve(numOfImagesToload); // the image file name mape is pre allocated to hold up to 0x10000 entries
        int32_t count = 0;
        while ((!nameListIO.eof()) && (count < numOfImagesToload)) 
        {
            nameListIO >> image_file_name;
            nameListIO >> label;
            m_indexImageMap.push_back(std::pair<int, std::string>(label, image_file_name));
            count++;
        }
        if (m_indexImageMap.empty()) // check if no entries were read (atleast on of the files were empty)
        {
            nameListIO.close();
            throw HABANA_benchException("imageCacheManager: no entries were read into the conversion map seems one of the imnput file were empty");
        }
        nameListIO.close();
        if (m_queryBuffer == nullptr)
        {
            // a null pointer to the loaded  images buffer - this buffer is allocated by
            // the runner class
            throw HABANA_benchException("imageCacheManager: couldn't allocate queryBuffer using the synapse callbacks");
        }
        try 
        {
            m_listOfloadedIndex.resize(numOfImagesToload); // uploaded image vector, this vector contains pointer
                                                        // to each image loaded
        } 
        catch (const std::exception &e) 
        {
            throw HABANA_benchException("imageCacheManager: couldn't allocate m_lisOfLoadedIndez vector");
        }
        std::vector<uint8_t *>::iterator it;
        for (it = m_listOfloadedIndex.begin(); it != m_listOfloadedIndex.end(); it++)
            *it = nullptr; // place nullptr so during upload of image we can test
                            // whether the image was loaded in the past loadgen some times
    }
    //*********************************************************************************************************************************
    //*********************************************************************************************************************************
    HABANA_imageCacheManager::~HABANA_imageCacheManager() {}


    
    void HABANA_imageCacheManager::load_samples(const     mlperf::QuerySampleIndex *queryIndexBuf,
                                                              size_t    numberOfquerySamples)
    {
        uint8_t *load_ptr = m_queryBuffer;
        std::string basePath = m_imageCacheDirPath + '/';
        std::ofstream dbg;
        #ifdef DEBUG_IMAGE_LOAD
        dbg.open("debug.txt");
        #endif
        for (int i = 0; i < numberOfquerySamples; i++)
        {
                size_t qindex = queryIndexBuf[i];
                if (qindex < m_indexImageMap.size())
                {
                    
                        std::string fullFileName;
                        std::ifstream imageRead;
                        fullFileName = basePath + m_indexImageMap[qindex].second;
                        imageRead.open(fullFileName.c_str(), std::ios::binary);

                        if (imageRead.is_open())
                        {
                            uint64_t length;
                            // calculate image size on disk
                            imageRead.seekg(0, std::ios::end);
                            length = imageRead.tellg();
                            imageRead.seekg(0, std::ios::beg);
                            if (length == m_imageSizeInBytes) // check that image size is as expected
                            {
                                #ifdef DEBUG_IMAGE_LOAD
                                std::stringstream a, b;
                                a << std::hex << (uint64_t)load_ptr;
                                b << std::hex << (((uint64_t)(m_queryBuffer)) + (uint64_t)numberOfquerySamples * 224ull * 224ull * 3ull);
                                dbg << std::setw(6) << i << std::setw(6) << qindex<<std::setw(6)<<m_indexImageMap[qindex].first<<std::setw(30)<<m_indexImageMap[queryIndexBuf[i]].second<< std::setw(13) << b.str() << std::setw(13) << a.str()<< std::endl;
                                #endif
                                imageRead.read(reinterpret_cast<char *>(load_ptr),m_imageSizeInBytes); // read the image
                                if (m_listOfloadedIndex[qindex] == nullptr)
                                {
                                    m_listOfloadedIndex[qindex] = load_ptr;
                                }                                
                                load_ptr += m_imageSizeInBytes; // move to next position in the buffer

                            } 
                            else
                            {
                                imageRead.close();  // close file and return status - the image size
                                                    // is not as expeted
                                throw HABANA_benchException("imageCacheManager: bad image size encountered during load function");
                            }     

                        } 
                        else
                        {
                            // return status can't open requested image file
                            throw HABANA_benchException("imageCacheManager: couldn't load image list file");

                        }
                        imageRead.close();
                    
                    
                    
                } 
                else
                {
                    throw HABANA_benchException("imageCacheManager: image map vector is empty");
                }
            }
            #ifdef DEBUG_IMAGE_LOAD
            dbg.close();
            #endif
    }

    void HABANA_imageCacheManager::unload_samples(size_t                         numberOfquerySamples)
    {
        for (int i = 0; i < numberOfquerySamples; i++)
        {
                m_listOfloadedIndex[i] = nullptr;
        }
    }

    void HABANA_imageCacheManager::getNextBatch(Habana_taskInfo &taskDescriptor)
    {
    uint32_t firstSampleIdx = taskDescriptor.m_querySamples[0].index;
    taskDescriptor.m_enqueueInputTensor.pTensorData =  reinterpret_cast<char*>(m_listOfloadedIndex[firstSampleIdx]);
    }
    char*    HABANA_imageCacheManager::getSample(const mlperf::QuerySample* querySamples)
    {
        uint32_t firstSampleIdx = querySamples->index;
    
        return reinterpret_cast<char*>(m_listOfloadedIndex[firstSampleIdx]);
    }

    std::string   HABANA_imageCacheManager::getImageName(uint64_t imageIndex)
    {
        if(imageIndex >= m_indexImageMap.size())
            return "";
        else
        {
            return m_indexImageMap[imageIndex].second;
        }
    }
}