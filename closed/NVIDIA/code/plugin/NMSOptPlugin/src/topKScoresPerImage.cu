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

#include <cub/cub.cuh>
#include <vector>

#include "ssdOpt.h"
#include "ssdOptMacros.h"

template <typename KeyT, typename ValueT>
size_t cubSortPairsWorkspaceSize(int num_items, int num_segments)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        (void*) NULL, temp_storage_bytes,
        (const KeyT*) NULL, (KeyT*) NULL,
        (const ValueT*) NULL, (ValueT*) NULL,
        num_items,    // # items
        num_segments, // # segments
        (const int*) NULL, (const int*) NULL);
    return temp_storage_bytes;
}


namespace nvinfer1
{
namespace plugin
{

namespace {
// sort one segment per cta
template<typename T_SCORE, int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void blockSortKernel(const T_SCORE *d_keys_in, T_SCORE *d_keys_out, const int *d_values_in, int *d_values_out, int* active_count_per_batch, int num_items, int stride_items, int num_segments)
{
    // Specialize BlockRadixSort for a 1D block
    typedef cub::BlockRadixSort<T_SCORE, BLOCK_THREADS, ELEMENTS_PER_THREAD, int> BlockRadixSort;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    if (blockIdx.x >= num_segments)
        return;

    int num_active_items = active_count_per_batch[blockIdx.x];

    // Obtain a segment of consecutive items that are blocked across threads
    T_SCORE thread_keys[ELEMENTS_PER_THREAD];
    int thread_values[ELEMENTS_PER_THREAD];

    int block_offset = blockIdx.x * stride_items;
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_in + block_offset, thread_keys, num_active_items, 0);
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_in + block_offset, thread_values, num_active_items, -1);
    __syncthreads();

    // Collectively sort the keys and values among block threads
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items);
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items);
}

/// block sort kernel
template <typename T_SCORE>
void blockSort(const T_SCORE *d_keys_in, T_SCORE *d_keys_out, const int *d_values_in, int *d_values_out, int* active_count_per_batch, int num_items, int stride_items, int num_segments, cudaStream_t stream)
{
    if (num_items == 0)
        return;

    int warps_per_cta = (num_items + 31) / 32;
    assert(warps_per_cta <= 8);

    dim3 block(warps_per_cta * 32);
    dim3 grid(num_segments);

    using kernel_func = void (*)(const T_SCORE *d_keys_in, T_SCORE *d_keys_out, const int *d_values_in, int *d_values_out, int* active_count_per_batch, int num_items, int stride_items, int num_segments);

    static const kernel_func kernel_funcs[] = {
        &blockSortKernel<T_SCORE, 32, 1>,
        &blockSortKernel<T_SCORE, 64, 1>,
        &blockSortKernel<T_SCORE, 96, 1>,
        &blockSortKernel<T_SCORE, 128, 1>,
        &blockSortKernel<T_SCORE, 160, 1>,
        &blockSortKernel<T_SCORE, 192, 1>,
        &blockSortKernel<T_SCORE, 224, 1>,
        &blockSortKernel<T_SCORE, 256, 1>,
    };
    kernel_funcs[warps_per_cta - 1]<<<grid, block, 0, stream>>>(d_keys_in, d_keys_out, d_values_in, d_values_out, active_count_per_batch, num_items, stride_items, num_segments);
}


template <int ITEMS_PER_THREAD>
__global__ void top_k_cuda(int *in, int *in_indices, int *out, int* out_indices, int* active_count_per_batch, int items, unsigned int num_top_k, int segments)
{
  extern __shared__ uint32_t dynamic_memory[];
  uint32_t* selected_items = dynamic_memory;
  int32_t* selected_indices = reinterpret_cast<int32_t*>(selected_items + num_top_k);
  __shared__ unsigned int selected_count;
  unsigned int old_selected_count;

  int batch = blockIdx.x;
  int first_index = batch * items;

  in += first_index;
  in_indices += first_index;

  out += first_index;
  out_indices += first_index;

  items = active_count_per_batch[batch];

  // Feed input
  uint32_t thread_items[ITEMS_PER_THREAD];
  int32_t thread_indices[ITEMS_PER_THREAD];

  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int offset = threadIdx.x + i * blockDim.x;
    if (offset < items) {
      thread_items[i] = in[offset];
      thread_indices[i] = in_indices[offset];
    }
     else {
      thread_items[i] = 0;
      thread_indices[i] = -1;
    }
  }

  if (items <= num_top_k) {
      if (threadIdx.x == 0) {
          active_count_per_batch[batch] = items;
      }

      // we know that the results are compact, so we can bail out early.
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
          int offset = threadIdx.x + i * blockDim.x;
          if (offset < num_top_k) {
              out[offset] = thread_items[i];
              out_indices[offset] = thread_indices[i];
          }
          else {
              return;
          }
      }
  }

  uint32_t select_mask = 0;
  uint32_t save_mask = 0;
  uint32_t save_bit = 0;

  if (threadIdx.x == 0) {
    selected_count = 0;
    old_selected_count = 0;
  }

  #define MTA_D 0

  // iterate over bits
  for (int i = 0; i < 32; ++i) {
    __syncthreads();
    uint32_t bit = select_mask | (1u << (31 - i));
    uint32_t &bit_mask = bit;

    // determine the number of elements for the current selection mask
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      if ((thread_items[item] & bit) == bit) {
        unsigned int offset = atomicAdd(&selected_count,1);
        if (offset < num_top_k) {
            selected_items[offset] = thread_items[item];
            selected_indices[offset] = thread_indices[item];
        }
        else {
            break;
        }
      }
    }

    // remove items from the list
    // TODO this has to be something different!
   __syncthreads();
    int sc = selected_count;
    __syncthreads();

    if (sc < num_top_k) {
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if ((thread_items[i] & bit) == bit) {
          thread_items[i] = 0;
        }
      }
    }

    if (sc == num_top_k || i == 31) {
        break;
    }
    else if (sc > num_top_k)
    {
        // There are too many bits in the current selection
        // Save the current state and go to the next bit
        // If there are not enough items left using the next bit
        // it's necessary to restart here with the current bit not set
        save_mask = bit_mask;
        save_bit = i + 1;
        select_mask |= bit;

        if (threadIdx.x == 0)
        {
            selected_count = old_selected_count;
        }
    }
    else {
        if (save_mask) {
            select_mask = save_mask;
            i = save_bit;

            save_mask = 0;
        }
        if (threadIdx.x == 0) {
            old_selected_count = sc;
        }
    }
  }

  __syncthreads();

  // store data to global memory
  int sc = selected_count;
  for (int i = threadIdx.x; i < num_top_k; i += blockDim.x) {
      out[i] = (i < sc) ? selected_items[i] : 1;
      out_indices[i] = (i < sc && selected_items[0] > 0) ? selected_indices[i] : -1;
  }

  if (threadIdx.x == 0) {
      active_count_per_batch[batch] = num_top_k;
  }

}

}

template <typename T_SCORE>
ssdStatus_t topKScoresPerImage_gpu(
    cudaStream_t stream,
    const int num_images,
    const int num_items_per_image,
    const int num_top_k,
    void* unsorted_scores,
    void* unsorted_bbox_indices,
    void* sorted_scores,
    void* sorted_bbox_indices,
    void* active_count_per_class,
    void* workspace)
{
    void* d_offsets = workspace;
    void* cubWorkspace = nextWorkspacePtr((int8_t*) d_offsets, (num_images + 1) * sizeof(int));

    uint32_t smem_size = num_top_k * (sizeof(int) + sizeof(uint32_t));
    uint32_t num_warps = (num_items_per_image > 1024) ? 32 : (num_items_per_image + 31) / 32;

    dim3 block(num_warps * 32);
    dim3 grid(num_images);

    using top_k_kernel = void (*)(int *in, int *in_indices, int *out, int* out_indices, int* active_count_per_class, int items, unsigned int num_top_k, int segments);
    top_k_kernel top_k_kernels[] = {
        top_k_cuda<1>,
        top_k_cuda<2>,
        top_k_cuda<3>,
        top_k_cuda<4>,
        top_k_cuda<5>,
        top_k_cuda<6>,
        top_k_cuda<7>,
        top_k_cuda<8>,
        top_k_cuda<9>,
        top_k_cuda<10>,
        top_k_cuda<11>,
        top_k_cuda<12>,
        top_k_cuda<13>,
        top_k_cuda<14>,
        top_k_cuda<15>,
        top_k_cuda<16>,
        top_k_cuda<17>,
        top_k_cuda<18>,
        top_k_cuda<19>,
        top_k_cuda<20>,
        top_k_cuda<21>,
        top_k_cuda<22>,
        top_k_cuda<23>,
        top_k_cuda<24>,
        top_k_cuda<25>,
        top_k_cuda<26>,
        top_k_cuda<27>,
        top_k_cuda<28>,
        top_k_cuda<29>,
        top_k_cuda<30>,
        top_k_cuda<31>,
        top_k_cuda<32>,
    };

    int kernel_index = num_items_per_image / block.x;
    while (kernel_index >= 32) {
        kernel_index /= 2;
        num_warps *= 2;
    }

    //printf("kernel index image %d\n", kernel_index);
    assert(kernel_index < 32);

    block.x = num_warps * 32;

    top_k_kernels[kernel_index]<<<grid, block, smem_size, stream>>>((int*) (unsorted_scores), (int*)unsorted_bbox_indices, (int*) (sorted_scores), (int*)sorted_bbox_indices, (int*)active_count_per_class, num_items_per_image, num_top_k, num_images);


    blockSort<T_SCORE>(
                       (const T_SCORE*) (sorted_scores), (T_SCORE*) (sorted_scores),
                       (const int*) (sorted_bbox_indices), (int*) (sorted_bbox_indices), (int*) active_count_per_class,
                       num_top_k, num_items_per_image, num_images, stream
    );

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerImage LAUNCH CONFIG {{{
typedef ssdStatus_t (*tkspiFunc)(cudaStream_t,
                                 const int,
                                 const int,
                                 const int,
                                 void*,
                                 void*,
                                 void*,
                                 void*,
                                 void*,
                                 void*);
struct tkspiLaunchConfig
{
    DType_t t_score;
    tkspiFunc function;

    tkspiLaunchConfig(DType_t t_score)
        : t_score(t_score)
    {
    }
    tkspiLaunchConfig(DType_t t_score, tkspiFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const tkspiLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<tkspiLaunchConfig> tkspiFuncVec;
bool tkspiInit()
{
    tkspiFuncVec.push_back(tkspiLaunchConfig(DataType::kFLOAT,
                                           topKScoresPerImage_gpu<float>));
    return true;
}

static bool initialized = tkspiInit();
//}}}

ssdStatus_t topKScoresPerImage(
    cudaStream_t stream,
    const int num_images,
    const int num_items_per_image,
    const int num_top_k,
    const DType_t DT_SCORE,
    void* unsorted_scores,
    void* unsorted_bbox_indices,
    void* sorted_scores,
    void* sorted_bbox_indices,
    void* active_count_per_gpu,
    void* workspace)
{
    tkspiLaunchConfig lc = tkspiLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < tkspiFuncVec.size(); ++i)
    {
        if (lc == tkspiFuncVec[i])
        {
            DEBUG_PRINTF("topKScoresPerImage kernel %d\n", i);
            return tkspiFuncVec[i].function(stream,
                                            num_images,
                                            num_items_per_image,
                                            num_top_k,
                                            unsorted_scores,
                                            unsorted_bbox_indices,
                                            sorted_scores,
                                            sorted_bbox_indices,
                                            active_count_per_gpu,
                                            workspace);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t topKScoresPerImageWorkspaceSize(
    const int num_images,
    const int num_items_per_image,
    const int num_top_k,
    const DType_t DT_SCORE)
{
    const int arrayLen = num_images * num_items_per_image;
    size_t wss[2];
    wss[0] = (num_images + 1) * sizeof(int); // offsets
    if (DT_SCORE == DataType::kFLOAT)
    {
        wss[1] = cubSortPairsWorkspaceSize<float, int>(arrayLen, num_images); // cub workspace
    }
    else
    {
        printf("SCORE type not supported.\n");
        return (size_t) -1;
    }

    return calculateTotalWorkspaceSize(wss, 2);
}

} // namespace plugin
} // namespace nvinfer1
