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

namespace nvinfer1
{
namespace plugin
{

namespace {
// sort one segment per cta
template<typename T_SCORE, int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void blockSortKernel(const T_SCORE *d_keys_in, T_SCORE *d_keys_out, const int *d_values_in, int *d_values_out, const int* active_counts, int num_items_, int stride_items, int num_segments)
{
    // Specialize BlockRadixSort for a 1D block
    typedef cub::BlockRadixSort<T_SCORE, BLOCK_THREADS, ELEMENTS_PER_THREAD, int> BlockRadixSort;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    if (blockIdx.x >= num_segments)
        return;

    int num_items = active_counts[blockIdx.x] > num_items_ ? num_items_ : active_counts[blockIdx.x];

    if (num_items == 0) {
        return;
    }

    // Obtain a segment of consecutive items that are blocked across threads
    T_SCORE thread_keys[ELEMENTS_PER_THREAD];
    int thread_values[ELEMENTS_PER_THREAD];

    int block_offset = blockIdx.x * stride_items;
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items, 0);
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items, -1);
    __syncthreads();

    // Collectively sort the keys and values among block threads
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items);
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items);
}

/// block sort kernel
template <typename T_SCORE>
void blockSort(const T_SCORE *d_keys_in, T_SCORE *d_keys_out, const int *d_values_in, int *d_values_out, const int* active_counts, int num_items, int stride_items, int num_segments, cudaStream_t stream)
{
    if (num_items == 0)
        return;

    int warps_per_cta = (num_items + 31) / 32;
    assert(warps_per_cta <= 8);

    dim3 block(warps_per_cta * 32);
    dim3 grid(num_segments);

    using kernel_func = void (*)(const T_SCORE *d_keys_in, T_SCORE *d_keys_out, const int *d_values_in, int *d_values_out, const int* active_counts, int num_items, int stride_items, int num_segments);

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
    kernel_funcs[warps_per_cta - 1]<<<grid, block, 0, stream>>>(d_keys_in, d_keys_out, d_values_in, d_values_out, active_counts, num_items, stride_items, num_segments);
}


template <int ITEMS_PER_THREAD>
__global__ void top_k_cuda_fused_prepare(int *in, int *out, int* out_indices, int* active_counts_per_class, int* active_count_per_batch, int items, unsigned int num_top_k, int segments, int background_class_id, float threshold)
{

  extern  __shared__ int2 dynamic_smem[];
  int2* selected_elements = dynamic_smem;
  __shared__ unsigned int selected_count;
  __shared__ unsigned int active_count;
  unsigned int old_selected_count;

  int class_id = blockIdx.x;
  int segment = blockIdx.y * gridDim.x + blockIdx.x;

  if (threadIdx.x == 0) {
      // We have to initialize active_count_per_batch for the following allClassNMS kernel.
      // Do it here to avoid to avoid an extra memset launch.
      if (blockIdx.x == 0) {
          active_count_per_batch[blockIdx.y] = 0;
      }
      active_count = 0;
  }
  __syncthreads();

  int first_index = segment * items;
  in += first_index;
  out += first_index;
  out_indices += first_index;

  int index_limit = items;
  uint32_t thread_items[ITEMS_PER_THREAD];
  int local_filtered = 0;

  // number of items whose score is >0 int he current thread
  int thread_active = 0;
  // in case <= top_k are active, offset where to write the thread items to in the output
  int thread_offset;

  if (background_class_id != class_id) {
      #pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
          int offset = threadIdx.x + i * blockDim.x;
          int index = first_index + offset;
          thread_items[i] = 0;
          if (offset < index_limit) {
              thread_items[i] = in[offset];
          }
      }

      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
          if (__int_as_float(thread_items[i]) < threshold) {
              thread_items[i] = 0;

              // todo a bitmask + popc might be faster here
              int offset = threadIdx.x + i * blockDim.x;
              if (offset < index_limit) {
                  ++local_filtered;
              }
          }
          if (thread_items[i] > 0) {
              thread_active++;
          }
      }
      thread_offset = atomicAdd(&active_count, thread_active);
  }

  uint32_t select_mask = 0;
  uint32_t save_mask = 0;
  uint32_t save_bit = 0;

  if (threadIdx.x == 0) {
    selected_count = 0;
    old_selected_count = 0;
  }

  __syncthreads();

  if (threadIdx.x == 0 ) {
       active_counts_per_class[segment] = active_count;
  } 

  // all elements are filtered, nothing to do
  if (active_count == 0) {
      return;
  }

   // we have at maximum top_k elements. there's no need to filter those, store them directly as result.
  if (active_count <= num_top_k) {
      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
          if (thread_items[i] != 0) {
              out_indices[thread_offset] = threadIdx.x + i * blockDim.x + items * blockIdx.x;
              out[thread_offset] = thread_items[i];
              ++thread_offset;
          }
      }
      return;
  }

  // iterate over bits.
  // skip the first two bits,
  // * bit 31 is the sign bit. all values are positive
  // * bit 30 is only set for values >= 2, but the input consists only of values in the range of [0,1]
  const int skip_bits = 2;
  for (int bit = 31 - skip_bits; true; --bit) {
    __syncthreads();
    uint32_t bit_mask = select_mask | (1u << bit);

    uint32_t enabled = 0;
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        enabled |= (((thread_items[item] ^ bit_mask) & bit_mask) == 0) << item;
    }

    uint32_t selected = __popc(enabled);
    unsigned int offset = atomicAdd(&selected_count,selected);

    __syncthreads();
    int sc = selected_count;
    __syncthreads();

    if ((sc <= num_top_k && sc > 0) || (bit == 0 && sc > 0)) {
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
         if (enabled & (1u << item) && offset < num_top_k) {
           selected_elements[offset] = make_int2(thread_items[item], threadIdx.x + item * blockDim.x + items * blockIdx.x);
           ++offset;
           thread_items[item] = 0;
         }
       }

    }

    if (sc == num_top_k || bit == 0) {
        break;
    }
    else if (sc > num_top_k)
    {
        // There are too many bits in the current selection
        // Save the current state and go to the next bit
        // If there are not enough items left using the next bit
        // it's necessary to restart here with the current bit not set
        save_mask = bit_mask;
        save_bit = bit - 1;
        select_mask |= bit_mask;

        if (threadIdx.x == 0)
        {
            selected_count = old_selected_count;
        }
    }
    else {
        if (save_mask) {
            select_mask = save_mask;
            bit = save_bit;

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
      int2 selected_element = selected_elements[i];
      int out_element = i < sc ? selected_element.x : 0;
      out[i] = out_element;
      out_indices[i] = out_element > 0 ? selected_element.y : -1;
  }
}

}

template <typename T_SCORE>
ssdStatus_t topKScoresPerClass_gpu(
    cudaStream_t stream,
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int num_top_k,
    const int background_label_id,
    const float confidence_threshold,
    void* conf_scores_gpu,
    void* index_array_gpu,
    void *active_counts_gpu,
    void* active_counts_per_batch_gpu,
    void* workspace)
{

    const int num_segments = num * num_classes;

    uint32_t smem_size = num_top_k * (sizeof(int) + sizeof(uint32_t));
    uint32_t num_warps = (num_preds_per_class > 128) ? (128/32) : (num_preds_per_class + 31) / 32;

    dim3 block(num_warps * 32);
    dim3 grid(num_classes, num);

    using top_k_kernel = void (*)(int *in, int *out, int* out_indices, int *active_counts_gpu, int* active_counts_per_batch_gpu, int items, unsigned int num_top_k, int segments, int background_class_id, float threshold) ;
    top_k_kernel top_k_kernels[] = {
        top_k_cuda_fused_prepare<1>,
        top_k_cuda_fused_prepare<2>,
        top_k_cuda_fused_prepare<3>,
        top_k_cuda_fused_prepare<4>,
        top_k_cuda_fused_prepare<5>,
        top_k_cuda_fused_prepare<6>,
        top_k_cuda_fused_prepare<7>,
        top_k_cuda_fused_prepare<8>,
        top_k_cuda_fused_prepare<9>,
        top_k_cuda_fused_prepare<10>,
        top_k_cuda_fused_prepare<11>,
        top_k_cuda_fused_prepare<12>,
        top_k_cuda_fused_prepare<13>,
        top_k_cuda_fused_prepare<14>,
        top_k_cuda_fused_prepare<15>,
        top_k_cuda_fused_prepare<16>,
        top_k_cuda_fused_prepare<17>,
        top_k_cuda_fused_prepare<18>,
        top_k_cuda_fused_prepare<19>,
        top_k_cuda_fused_prepare<20>,
        top_k_cuda_fused_prepare<21>,
        top_k_cuda_fused_prepare<22>,
        top_k_cuda_fused_prepare<23>,
        top_k_cuda_fused_prepare<24>,
        top_k_cuda_fused_prepare<25>,
        top_k_cuda_fused_prepare<26>,
        top_k_cuda_fused_prepare<27>,
        top_k_cuda_fused_prepare<28>,
        top_k_cuda_fused_prepare<29>,
        top_k_cuda_fused_prepare<30>,
        top_k_cuda_fused_prepare<31>,
        top_k_cuda_fused_prepare<32>,
    };

    int kernel_index = num_preds_per_class / block.x;
    while (kernel_index >= 32) {
        kernel_index /= 2;
        num_warps *= 2;
    }
    assert(kernel_index < 32);

    assert(num_warps * 32 <= 1024);

    block.x = num_warps * 32;

    top_k_kernels[kernel_index]<<<grid, block, smem_size, stream>>>((int*) (conf_scores_gpu), (int*) (conf_scores_gpu), (int*)index_array_gpu, (int*)active_counts_gpu, (int*)active_counts_per_batch_gpu, num_preds_per_class, num_top_k, num_segments, background_label_id, confidence_threshold);

    blockSort<T_SCORE>(
        (const T_SCORE*) (conf_scores_gpu), (T_SCORE*) (conf_scores_gpu),
        (const int*) (index_array_gpu), (int*) (index_array_gpu), (int*)active_counts_gpu,
        num_top_k, num_preds_per_class, num_segments, stream
    );

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerClass LAUNCH CONFIG {{{
typedef ssdStatus_t (*tkspcFunc)(cudaStream_t,
                                const int,
                                const int,
                                const int,
                                const int,
                                const int,
                                const float,
                                void*,
                                void*,
                                void*,
                                void*,
                                void*);
struct tkspcLaunchConfig
{
    DType_t t_score;
    tkspcFunc function;

    tkspcLaunchConfig(DType_t t_score)
        : t_score(t_score)
    {
    }
    tkspcLaunchConfig(DType_t t_score, tkspcFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const tkspcLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

static std::vector<tkspcLaunchConfig> tkspcFuncVec;
bool tkspcInit()
{
    tkspcFuncVec.push_back(tkspcLaunchConfig(DataType::kFLOAT,
                                           topKScoresPerClass_gpu<float>));
    return true;
}

static bool initialized = tkspcInit();
//}}}

ssdStatus_t topKScoresPerClass(
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
    void *active_count_per_class,
    void *active_count_per_batch,
    void* workspace)
{
    tkspcLaunchConfig lc = tkspcLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < tkspcFuncVec.size(); ++i)
    {
        if (lc == tkspcFuncVec[i])
        {
            DEBUG_PRINTF("sortScoresPerClass kernel %d\n", i);
            return tkspcFuncVec[i].function(stream,
                                           num,
                                           num_classes,
                                           num_preds_per_class,
                                           num_top_k,
                                           background_label_id,
                                           confidence_threshold,
                                           conf_scores_gpu,
                                           index_array_gpu,
                                           active_count_per_class,
                                           active_count_per_batch,
                                           workspace);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t topKScoresPerClassWorkspaceSize(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int num_top_k,
    const DType_t DT_CONF)
{
    return 0;
}

} // namespace plugin
} // namespace nvinfer1
