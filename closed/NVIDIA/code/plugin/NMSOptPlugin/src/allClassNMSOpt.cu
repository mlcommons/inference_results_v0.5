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

#include <vector>
#include <algorithm>
#include <stdint.h>

#include "ssdOpt.h"
#include "ssdOptMacros.h"

namespace nvinfer1
{
namespace plugin
{

namespace {
__device__ __inline__ void swap(float &a, float &b)
{
    float temp = a;
    a = temp;
    b = temp;
}

} // namespace anonymous

template <typename T_BBOX>
__device__ T_BBOX bboxSizeOpt(
    const Bbox<T_BBOX>& bbox,
    const bool normalized)
{
    if (normalized) {
        // If any bbox dimension is negative the result will be zero.
        T_BBOX width = fmaxf(bbox.xmax - bbox.xmin, 0.0f);
        T_BBOX height = fmaxf(bbox.ymax - bbox.ymin, 0.0f);
        return width * height;
    } else {
        T_BBOX width = bbox.xmax - bbox.xmin;
        T_BBOX height = bbox.ymax - bbox.ymin;
        if (width < 0 || height < 0) {
            return 0.0f;
        }
        return (width + 1.0f) * (height + 1.0f);
    }
}

template <typename T_BBOX>
__device__ void intersectBboxOpt(
    const Bbox<T_BBOX>& bbox1,
    const Bbox<T_BBOX>& bbox2,
    Bbox<T_BBOX>* intersect_bbox)
{
    intersect_bbox->xmin = max(bbox1.xmin, bbox2.xmin);
    intersect_bbox->ymin = max(bbox1.ymin, bbox2.ymin);
    intersect_bbox->xmax = min(bbox1.xmax, bbox2.xmax);
    intersect_bbox->ymax = min(bbox1.ymax, bbox2.ymax);
}

template <typename T_BBOX>
__device__ float jaccardOverlapOpt(
   const Bbox<T_BBOX>& bbox1,
   const Bbox<T_BBOX>& bbox2,
    const bool normalized)
{
    Bbox<T_BBOX> intersect_bbox;
    intersectBboxOpt(bbox1, bbox2, &intersect_bbox);

    float intersect_size = bboxSizeOpt(intersect_bbox, normalized);
    float bbox1_size = bboxSizeOpt(bbox1, normalized);
    float bbox2_size = bboxSizeOpt(bbox2, normalized);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}

template <typename T_BBOX>
__device__ void emptyBboxInfoOpt(
    BboxInfo<T_BBOX>* bbox_info)
{
    bbox_info->conf_score = T_BBOX(0);
    bbox_info->label = -2; // -1 is used for all labels when shared_location is ture
    bbox_info->bbox_idx = -1;
    bbox_info->kept = false;
}
/********** new NMS for only score and index array **********/


template <typename T_SCORE, typename T_BBOX, int TSIZE, bool isNormalized>
__global__ void allClassNMSOpt_kernel(
    const int num_no_use,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k_,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized_unused,
    T_BBOX* bbox_data, // bbox_data should be float to preserve location information
    T_SCORE* beforeNMS_scores,
    int* beforeNMS_index_array,
    T_SCORE* afterNMS_scores,
    int* afterNMS_index_array,
    int* active_count, // number of active elemements per class/batch
    int* active_count_per_batch,
    bool flipXY = false)
{
    const int num_smem_elements = TSIZE * blockDim.x;

    // keep a small smem cache for the bboxes. Alignment is guaranteed due to the order of the definitions.
    extern __shared__ int4 dynamic_smem[];
    // number of active elements for the current batch_class combi
    __shared__ int result_active_count;

    Bbox<T_BBOX> *sh_bbox = reinterpret_cast<Bbox<T_BBOX>*>(dynamic_smem);
    bool *kept_bboxinfo_flag = reinterpret_cast<bool*>(sh_bbox + num_smem_elements);

    int active = active_count[blockIdx.y * gridDim.x + blockIdx.x];
    int top_k = (active < top_k_) ? active : top_k_;

    int class_id = blockIdx.x;
    int batch_id = blockIdx.y;

    // Each thread touches only a certain subset of all bboxinfos. Keep the kept_bboxinfo_flag for the thread in a bitmask.
    uint32_t thread_kept_bboxinfo_flag = 0;
    const int offset = batch_id * num_classes * num_preds_per_class + class_id * num_preds_per_class;

    // local thread data
    // TODO loc_bboxIndex is only required during the bbox initialization phase. don't waste registers for it...
    int loc_bboxIndex[TSIZE];
    Bbox<T_BBOX> loc_bbox[TSIZE];

    if (active)
    {
        // we do not have to synchronize after writing active_count_per_batch.
        // T_SIZE is > 0, so there'll be at least one syncthreads before the first usage of this variable.
        if (threadIdx.x == 0) {
            result_active_count = 0;
        }

        const int max_idx = offset + top_k; // put top_k bboxes into NMS calculation
        const int bbox_idx_offset = share_location ? (batch_id * num_preds_per_class) : (batch_id * num_classes * num_preds_per_class);


// {{{ initialize Bbox, Bboxinfo, kept_bboxinfo_flag
#pragma unroll
        for (int t = 0; t < TSIZE; t++)
        {
            bool thread_kept_bboxinfo = false;
            const int cur_idx = threadIdx.x + blockDim.x * t;
            const int item_idx = offset + cur_idx;

            if (item_idx < max_idx)
            {
                loc_bboxIndex[t] = beforeNMS_index_array[item_idx];
                if (loc_bboxIndex[t] != -1)
                {
                    const int bbox_data_idx = share_location ? (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset) : loc_bboxIndex[t];
                    loc_bbox[t] = ((Bbox<T_BBOX>*)bbox_data)[bbox_data_idx];
                    if (flipXY) {
                        swap(loc_bbox[t].xmin, loc_bbox[t].ymin);
                        swap(loc_bbox[t].xmax, loc_bbox[t].ymax);
                    }
                    sh_bbox[cur_idx] = loc_bbox[t];

                    thread_kept_bboxinfo = true;
                    thread_kept_bboxinfo_flag |= (1 << t);
                }
            }
            kept_bboxinfo_flag[cur_idx] = thread_kept_bboxinfo;
        }

        // }}}
        __syncthreads();

        // TODO we can use loc_bboxIndex[t] == -1 to find the maximum index which is -1 and set max_idx to this value. This would reduce
        // the number of iterations for all threads if there are less than top-k bboxes available. How likey is this?

        // {{{ filter out overlapped boxes with lower scores
        {
            const int offset = 0;
            const int max_idx = top_k;
            int ref_item_idx = 0;

            while (ref_item_idx < max_idx)
            {
                Bbox<T_BBOX> ref_bbox;
                //*((int4*)&ref_bbox) = *((int4*)&sh_bbox[ref_item_idx - offset]);
                ref_bbox = sh_bbox[ref_item_idx];

                //uint32_t enabled = ~1;
                for (int t = 0; t < TSIZE; t++)
                {
                    const int cur_idx = threadIdx.x + blockDim.x * t;
                    const int item_idx = offset + cur_idx;

                    if ((item_idx > ref_item_idx) && (thread_kept_bboxinfo_flag & (1 << t)))
                    {
                        if (jaccardOverlapOpt(ref_bbox, loc_bbox[t], isNormalized) > nms_threshold)
                        {
                            thread_kept_bboxinfo_flag &= ~(1 << t);
                            kept_bboxinfo_flag[cur_idx] = false;
                        }
                    }
                }
                __syncthreads();

                do
                {
                    ref_item_idx++;
                } while (ref_item_idx < max_idx && !kept_bboxinfo_flag[ref_item_idx - offset]);
            }
        }
        // }}}

        // {{{ store data
        // Ideally we'd compact the data for the next stage to reduce work on the next stage.
        // As long as there's no TopK algorithm with a dynamic number of elements for the input
        // it doesn't make sense yet to do the compact step.

        // first determine the total amount of active elements after the NMS step
        int thread_active =  __popc(thread_kept_bboxinfo_flag);
        if (thread_active) {
            int write_offset = atomicAdd(&active_count_per_batch[batch_id], thread_active);
            int write_item_idx = (batch_id * num_classes * top_k_) + write_offset;
            for (int t = 0; t < TSIZE; t++) {
                const int cur_idx = threadIdx.x + blockDim.x * t;
                const int read_item_idx = offset + cur_idx;

                bool is_valid_bbox = (thread_kept_bboxinfo_flag & (1 << t));
                if (is_valid_bbox) {
                    afterNMS_scores[write_item_idx] = beforeNMS_scores[read_item_idx];
                    afterNMS_index_array[write_item_idx] = loc_bboxIndex[t];
                    ++write_item_idx;
                }
            }
        }
        // }}}
    }
}

template <typename T_SCORE, typename T_BBOX>
ssdStatus_t allClassNMSOpt_gpu(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int top_k,
    const float nms_threshold,
    const bool share_location,
    const bool isNormalized,
    void* bbox_data,
    void* beforeNMS_scores,
    void* beforeNMS_index_array,
    void* afterNMS_scores,
    void* afterNMS_index_array,
    void* active_count,
    void* active_count_per_batch,
    bool flipXY = false)
{
#define NMS_P(tsize) allClassNMSOpt_kernel<T_SCORE, T_BBOX, (tsize), true>
#define NMS_P_U(tsize) allClassNMSOpt_kernel<T_SCORE, T_BBOX, (tsize), false>

    void (*kernel[2][8])(const int, const int, const int, const int, const float,
                         const bool, const bool, float*, T_SCORE*, int*, T_SCORE*, int*,
                         int*, int*, bool)
        = {
        {NMS_P_U(1), NMS_P_U(2), NMS_P_U(3), NMS_P_U(4), NMS_P_U(5), NMS_P_U(6), NMS_P_U(7), NMS_P_U(8),},
        {NMS_P(1), NMS_P(2), NMS_P(3), NMS_P(4), NMS_P(5), NMS_P(6), NMS_P(7), NMS_P(8),}
        };

    // round up #threads to the minimum cta size possible which holds 1 bbox per thread
    // TODO 1024 is the #threads per CTA limit and should be queried from the GPU.
    // With top_k > max #threads per CTA this heuristic gets inefficient and should be enhanced
    // to reduce the number of idle threads.
    const int BS = std::min(((top_k + 31) / 32) * 32, 1024);
    const dim3 GS(num_classes, num);
    const int t_size = (top_k + BS - 1) / BS;
    assert(t_size < 8);

    // compute smem size for bbox cache and kept boxes
    const int smem_size = BS * t_size * (sizeof(bool) + sizeof(Bbox<T_BBOX>));
    kernel[isNormalized][t_size - 1]<<<GS, BS, smem_size, stream>>>(num, num_classes, num_preds_per_class,
                                                                    top_k, nms_threshold, share_location, isNormalized,
                                                                    (T_BBOX*) bbox_data,
                                                                    (T_SCORE*) beforeNMS_scores,
                                                                    (int*) beforeNMS_index_array,
                                                                    (T_SCORE*) afterNMS_scores,
                                                                    (int*) afterNMS_index_array,
                                                                    (int*) active_count,
                                                                    (int*) active_count_per_batch,
                                                                    flipXY);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// allClassNMSOpt LAUNCH CONFIG {{{
typedef ssdStatus_t (*nmsOptFunc)(cudaStream_t,
                               const int,
                               const int,
                               const int,
                               const int,
                               const float,
                               const bool,
                               const bool,
                               void*,
                               void*,
                               void*,
                               void*,
                               void*,
                                  void*, // activeCount
                                  void*, // activeCountPerClass
                               bool);

struct nmsOptLaunchConfigSSD
{
    DType_t t_score;
    DType_t t_bbox;
    nmsOptFunc function;

    nmsOptLaunchConfigSSD(DType_t t_score, DType_t t_bbox)
        : t_score(t_score)
        , t_bbox(t_bbox)
    {
    }
    nmsOptLaunchConfigSSD(DType_t t_score, DType_t t_bbox, nmsOptFunc function)
        : t_score(t_score)
        , t_bbox(t_bbox)
        , function(function)
    {
    }
    bool operator==(const nmsOptLaunchConfigSSD& other)
    {
        return t_score == other.t_score && t_bbox == other.t_bbox;
    }
};

static std::vector<nmsOptLaunchConfigSSD> nmsOptFuncVec;

bool nmsOptInit()
{
    nmsOptFuncVec.push_back(nmsOptLaunchConfigSSD(DataType::kFLOAT, DataType::kFLOAT,
                                            allClassNMSOpt_gpu<float, float>));
    return true;
}

static bool initialized = nmsOptInit();

//}}}

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
                           void* active_count,
                           void* active_count_per_batch,
                           bool flipXY)
{
    nmsOptLaunchConfigSSD lc = nmsOptLaunchConfigSSD(DT_SCORE, DT_BBOX, allClassNMSOpt_gpu<float, float>);
    for (unsigned i = 0; i < nmsOptFuncVec.size(); ++i)
    {
        if (lc == nmsOptFuncVec[i])
        {
            DEBUG_PRINTF("all class nms kernel %d\n", i);
            return nmsOptFuncVec[i].function(stream,
                                          num,
                                          num_classes,
                                          num_preds_per_class,
                                          top_k,
                                          nms_threshold,
                                          share_location,
                                          isNormalized,
                                          bbox_data,
                                          beforeNMS_scores,
                                          beforeNMS_index_array,
                                          afterNMS_scores,
                                          afterNMS_index_array,
                                          active_count,
                                          active_count_per_batch,
                                          flipXY);
        }
    }
    return STATUS_BAD_PARAM;
}

} // namespace plugin
} // namespace nvinfer1
