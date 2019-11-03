#ifndef BATCH_STREAM_PPM_H
#define BATCH_STREAM_PPM_H
#include <vector>
#include <iostream>
#include <string>
#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include "NvInfer.h"
#include "logger.h"
#include "common.h"

static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 300;
static constexpr int INPUT_W = 300;

//const char* INPUT_BLOB_NAME = "Input";

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
    {
        mDims = nvinfer1::DimsNCHW{batchSize, 3, 300, 300};
        mImageSize = mDims.c() * mDims.h() * mDims.w();
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.n() * mImageSize, 0);
        mFileLabels.resize(mDims.n(), 0);
        reset(0);
    }

    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.n();
        skip(firstBatch);
    }

    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
//            if (mFileBatchPos == mDims.n() && !update())
  //              return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        }
        mBatchCount++;
        return true;
    }

    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.n() && mBatchSize % mDims.n() == 0 && mFileBatchPos == mDims.n())
        {
            mFileCount += skipCount * mBatchSize / mDims.n();
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return mBatch.data(); }
    float* getLabels() { return mLabels.data(); }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    nvinfer1::DimsNCHW getDims() const { return mDims; }

private:
    float* getFileBatch() { return mFileBatch.data(); }
    float* getFileLabels() { return mFileLabels.data(); }

/*    bool update()
      {
          std::vector<std::string> fNames;

          std::ifstream file(locateFile("/data/dataset-imagenet-ilsvrc2012-val/val_map.txt"), std::ios::binary);
          if (file)
          {
              gLogInfo << "Batch #" << mFileCount << std::endl;
              file.seekg(((mBatchCount * mBatchSize)) * 7);
          }
          for (int i = 1; i <= mBatchSize; i++)
          {
              std::string sName;
              std::getline(file, sName);
              sName = sName + ".ppm";

              gLogInfo << "Calibrating with file " << sName << std::endl;
              fNames.emplace_back(sName);
          }
          mFileCount++;

          std::vector<samplesCommon::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(fNames.size());
          for (uint32_t i = 0; i < fNames.size(); ++i)
          {
              readPPMFile(locateFile(fNames[i]), ppms[i]);
          }
          std::vector<float> data(samplesCommon::volume(mDims));

          long int volChl = mDims.h() * mDims.w();

          for (int i = 0, volImg = mDims.c() * mDims.h() * mDims.w(); i < mBatchSize; ++i)
          {
              for (int c = 0; c < mDims.c(); ++c)
              {
                  for (int j = 0; j < volChl; ++j)
                  {
                      data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * mDims.c() + c]) - 1.0;
                  }
              }
          }

          std::copy_n(data.data(), mDims.n() * mImageSize, getFileBatch());

          mFileBatchPos = 0;
          return true;
      }*/

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};

    int mFileCount{0}, mFileBatchPos{0};
    int mImageSize{0};

    nvinfer1::DimsNCHW mDims;
    std::vector<float> mBatch;
    std::vector<float> mLabels;
    std::vector<float> mFileBatch;
    std::vector<float> mFileLabels;
};

#endif
