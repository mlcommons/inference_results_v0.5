/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudnn.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"

typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} ssdStatus_t;

// STRUCT BBOX {{{
template <typename T>
struct Bbox
{
    T xmin, ymin, xmax, ymax;
    Bbox(T xmin, T ymin, T xmax, T ymax)
        : xmin(xmin)
        , ymin(ymin)
        , xmax(xmax)
        , ymax(ymax)
    {
    }
    Bbox() = default;
};

template <typename T>
struct BboxInfo
{
    T conf_score;
    int label;
    int bbox_idx;
    bool kept;
    BboxInfo(T conf_score, int label, int bbox_idx, bool kept)
        : conf_score(conf_score)
        , label(label)
        , bbox_idx(bbox_idx)
        , kept(kept)
    {
    }
    BboxInfo() = default;
};

template <typename TFloat>
bool operator<(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
    return lhs.x1 < rhs.x1;
}

template <typename TFloat>
bool operator==(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
    return lhs.x1 == rhs.x1 && lhs.y1 == rhs.y1 && lhs.x2 == rhs.x2 && lhs.y2 == rhs.y2;
}
// }}}

template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}


template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

using nvinfer1::plugin::CodeTypeSSD;
typedef nvinfer1::DataType DType_t;

namespace nvinfer1
{
namespace plugin
{
ssdStatus_t detectionInferenceOpt(
    cudaStream_t stream,
    const int N,
    const int C1,
    const int C2,
    const bool shareLocation,
    const bool varianceEncodedInTarget,
    const int backgroundLabelId,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const float confidenceThreshold,
    const float nmsThreshold,
    const CodeTypeSSD codeType,
    const DType_t DT_BBOX,
    const void* const* locData,
    const void* priorData,
    const DType_t DT_SCORE,
    const void* const* confData,
    void* topDetections,
    void* workspace,
    bool isNormalized,
    bool confSigmoid,
    bool confSoftmax,
    const int numLayers,
    const int* featureSize,
    const int* numAnchors,
    const bool packed32NCHW,
    cudnnHandle_t cudnnHandle,
    cudnnTensorDescriptor_t inScoreDesc,
    cudnnTensorDescriptor_t outScoreDesc);

ssdStatus_t decodeBBoxesOpt(
    cudaStream_t stream,
    const int nthreads,
    const CodeTypeSSD code_type,
    const bool variance_encoded_in_target,
    const int num_priors,
    const bool share_location,
    const int num_loc_classes,
    const int background_label_id,
    const bool clip_bbox,
    const DType_t DT_BBOX,
    const void* const* loc_data,
    const void* prior_data,
    void* bbox_data,
    const int num_layers,
    const int * feature_size,
    const int * num_anchors,
    const bool packed32NCHW,
    const bool reshape_before_permute);

ssdStatus_t permuteConfData(cudaStream_t stream,
                        const int nthreads,
                        const int num_classes,
                        const int num_priors,
                        const int num_dim,
                        const DType_t DT_DATA,
                        bool confSigmoid,
                        const void* const* conf_data,
                        void* new_data,
                        void *active_count,
                        const int num_layers,
                        const int * feature_size,
                        const int * num_anchors,
                        const bool packed32NCHW);

ssdStatus_t allClassNMSOpt(cudaStream_t stream,
                        const int num,
                        const int num_classes,
                        const int num_preds_per_class,
                        const int top_k,
                        const float nms_threshold,
                        const bool share_location,
                        const bool isNormalized,
                        const DType_t DT_SCORE,
                        const DType_t DT_BBOX,
                        void* bbox_data,
                        void* beforeNMS_scores,
                        void* beforeNMS_index_array,
                        void* afterNMS_scores,
                        void* afterNMS_index_array,
                        void *active_count,
                        void *active_count_per_class,
                        bool flipXY);

ssdStatus_t topKScoresPerClass(
    cudaStream_t stream,
    int num,
    int num_classes,
    int num_preds_per_batch,
    int num_top_k,
    int background_label_id,
    float confidence_threshold,
    DType_t DT_SCORE,
    void* conf_scores_gpu,
    void* index_array_gpu,
    void *active_count,
    void *active_count_per_class,
    void* workspace);

size_t topKScoresPerClassWorkspaceSize(
    int num,
    int num_classes,
    int num_preds_per_class,
    int num_top_k,
    DType_t DT_CONF);

ssdStatus_t topKScoresPerClassFusedPermute(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int num_top_k,
    const int background_label_id,
    const float confidence_threshold,
    const DType_t DT_SCORE,
    void* conf_scores_gpu,
    void* index_array_gpu,
    void* active_count,
    void* active_count_per_batch,
    void* workspace,
    const int num_priors,
    const int num_dim,
    bool confSigmoid,
    const void* const* conf_data,
    const int num_layers,
    const int* feature_size,
    const int* num_anchors,
    const bool packed32NCHW
);

ssdStatus_t topKScoresPerImage(
    cudaStream_t stream,
    int num_images,
    int num_items_per_image,
    int num_top_k,
    DType_t DT_SCORE,
    void* unsorted_scores,
    void* unsorted_bbox_indices,
    void* sorted_scores,
    void* sorted_bbox_indices,
    void* active_count_per_batch,
    void* workspace);

size_t topKScoresPerImageWorkspaceSize(
    int num_images,
    int num_items_per_image,
    int num_top_k,
    DType_t DT_SCORE);

ssdStatus_t gatherTopDetectionsOpt(
    cudaStream_t stream,
    bool shareLocation,
    int numImages,
    int numPredsPerClass,
    int numClasses,
    int topK,
    int keepTopK,
    DType_t DT_BBOX,
    DType_t DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* topDetections);

ssdStatus_t softmaxScore(cudaStream_t stream,
                        const int N,
                        const int num_classes,
                        const int num_priors,
                        const int num_dims,
                        const DType_t DT_DATA,
                        const void * in_scores,
                        void * out_scores,
                        cudnnHandle_t handle,
                        cudnnTensorDescriptor_t inScoreDesc,
                        cudnnTensorDescriptor_t outScoreDesc);


size_t detectionForwardBBoxDataSize(int N,
                                    int C1,
                                    DType_t DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       DType_t DT_BBOX);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N,
                                   int numClasses,
                                   int topK);

size_t detectionInferenceWorkspaceSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       int C2,
                                       int numClasses,
                                       int numPredsPerClass,
                                       int topK,
                                       DType_t DT_BBOX,
                                       DType_t DT_SCORE);



} // namespace plugin
} // namespace nvinfer1
