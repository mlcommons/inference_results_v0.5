#ifndef _IMAGE_CACHE_MANAGER_
#define _IMAGE_CACHE_MANAGER_

#include   <HABANA_runner_task.h>
#include   <string>
#include   <HABANA_exception.h>
#include   <unordered_map>
#include   <vector>

namespace HABANA_bench
{
        //*********************************************************************************************************************************
    //*********************************************************************************************************************************
    // HABANA_imageCacheManager is the image cache responsible to upload images from disk into the host allocated buffer.
    // the upload happens each time a new upload request received from the loadgen. 
    // the host buffer is been allocated using synAlloc (that use pinning) 
    //*********************************************************************************************************************************
    //*********************************************************************************************************************************
    class HABANA_imageCacheManager
    {
        public:
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            // ImageDB constructor
            // allocates a query buffer size of image_size*max_num_of_query_samples the buffer is allocated using synAlloc() on the host memory
            // the constructor read the image name for a predefined image list files and fill the names and labels into an ordered map to be used later
            // during the upload function.
            //
            // IMPORTANT: the constructor may throw HABANA_benchException if the following occures
            // 1. allocation problem 
            // 2. nullptr in the query buffer
            //
            // INPUT:
            // 1) imageCachePath - path containing all preprocessed images
            // 2) name of a text file containing a list of all images in the image cache directory
            // 3) name of a text file containing all image tags (must be aligned with the imageNameList)
            // 4) maxQuerySize - maximal sample in a query
            // 5) batchSize - optimal batchSize (as defined by Habana platform - typically 10 images per batch)
            // 6) imageHeight - imageWidth, imageDepth - image dimensions
            // 7) synAllocFunc - functions for allocating Host memory by Synapse API
            // 
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            HABANA_imageCacheManager(const std::string  &imageCachePath,
                                     const std::string  &imageNameList,
                                     uint64_t           maxQuerySize, 
                                     uint64_t           numOfImagesToLoad,
                                     uint64_t           imageHeight,
                                     uint64_t           imageWidth,
                                     uint64_t           imageDepth,
                                     uint8_t            *queryBuf);
            // empty constructor
            virtual ~HABANA_imageCacheManager();

            
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            // load_samples()
            // the function upload "numberOfQuery" samples (images) from disk to the internally allocated cache buffer.
            // the function uses the m_indexImageMap to identify the images by the given indexes in the queryIndexBuf.
            // it also fills in the index value to m_listOfLoadedIndex this buffer is used for easier indexing and query partitioning into batch sizes.
            // INPUT:
            // queryIndexBuf - a buffer containing all query indexes to be loaded
            // numberOfQuerySamples - number of samples in the query.
            // 
            // OUTPUT
            // execution status
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            void load_samples(const mlperf::QuerySampleIndex* queryIndexBuf, size_t numberOfquerySamples);
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            // unload_samples()
            // the function marks samples as removed from the query cache by placing a nullptr in the loaded sample entry
            // 
            // OUTPUT
            // execution status
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            void unload_samples(size_t numberOfquerySamples);
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            // getNextBatch
            // the function returns a vector of image descriptor to be enqueued (processed by the Habana HW)
            // the assumption is that the caller allocates the vector of size of batchsize (minimal processing unit of the GOYA for specific run
            // single stream, multistream,server,offline).
            // ass all images placed in consecutive buffer in pinned memory, the first descriptor contains the pointer for the full chunk of images
            // the size of the vector is the amount of images returned for processing.
            //*********************************************************************************************************************************
            //*********************************************************************************************************************************
            void                    getNextBatch(Habana_taskInfo&taskDesc);
            char*                   getSample(const mlperf::QuerySample* querySamples);
            std::string             getImageName(uint64_t imageIndex);
        private:
            uint8_t                                                  *m_queryBuffer;             // query cache buffer (containing the image data of the query)
            std::vector<std::pair<uint32_t,std::string>>             m_indexImageMap;           // translation map between sample query index and image file name
            std::vector<uint8_t *>                                   m_listOfloadedIndex;       // loaded index in the query image cache
            uint64_t                                                 m_numOfImagesToload;               // batch size to be used by the Habana API during inference
            uint64_t                                                 m_maxquerySize;            // maximal size of a query
            uint64_t                                                 m_imageSizeInBytes;        // image size in bytes
            std::string                                              m_imageCacheDirPath;       // path to the image directory path
    };

}
#endif