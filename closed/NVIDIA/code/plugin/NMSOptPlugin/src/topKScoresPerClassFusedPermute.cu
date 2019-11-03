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
#include <cstring>

#include "ssdOpt.h"
#include "ssdOptMacros.h"
#include "fast_divmod.h"

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


template <typename Dtype, int NUM_LAYERS>
struct PermuteConfData {
    const Dtype * conf_data[NUM_LAYERS];
    int feature_size[NUM_LAYERS];
    int num_anchors[NUM_LAYERS];
    int end_layer_prior[NUM_LAYERS];
    bool packed32_nchw;
};

template <typename Dtype, int NUM_LAYERS>
__device__ __inline__ Dtype permuteConfData(
        int index,
        const int num_classes, int num_classes_mul, int num_classes_shr,
        const int num_priors, int num_priors_mul, int num_priors_shr,
        const int num_dim, int num_dim_mul, int num_dim_shr,
        int fast_divmod3_mul, int fast_divmod3_shr,
        int fast_divmod6_mul, int fast_divmod6_shr,
        int fast_divmod4_mul, int fast_divmod4_shr,
        bool confSigmoid,
        const PermuteConfData<Dtype, NUM_LAYERS> &permute_conf_data)
{
    //int feature_size_pow2[NUM_LAYERS];
    int feature_size[NUM_LAYERS];
    int all_num_anchors[NUM_LAYERS];
    const Dtype *conf_data[NUM_LAYERS];
    const bool packed32_nchw = permute_conf_data.packed32_nchw;

    #pragma unroll
    for (int layer = 0;layer < NUM_LAYERS;++layer) {
        feature_size[layer] = permute_conf_data.feature_size[layer];
        all_num_anchors[layer] = permute_conf_data.num_anchors[layer];
        conf_data[layer] = permute_conf_data.conf_data[layer];
    }
    
    {
        int i, i_div, d, d_div, c, n;

        fast_divmod(i_div, i, index, num_dim, num_dim_mul, num_dim_shr);
        fast_divmod(d_div, d, i_div, num_priors, num_priors_mul, num_priors_shr);
        fast_divmod(n, c, d_div, num_classes, num_classes_mul, num_classes_shr);

        //find layer_id
        int start_layer_prior = 0, end_layer_prior = 0;
        int prior_in_layer = 0;
        const Dtype *conf_data_layer;

        int num_hw;
        int layer;
        int num_anchors;
        #pragma unroll
        for(layer = 0; layer < NUM_LAYERS; layer++) {
            end_layer_prior = permute_conf_data.end_layer_prior[layer];

            if(d < end_layer_prior) {
                conf_data_layer = conf_data[layer];
                num_hw = feature_size[layer];

                num_anchors = all_num_anchors[layer];

                prior_in_layer = d - start_layer_prior;

                d = INT_MAX;
            }
            start_layer_prior = end_layer_prior;
        }

        int mappedIndex;
        int anchor, hw;
        if (num_anchors == 3) {
            fast_divmod(hw, anchor, prior_in_layer, 3, fast_divmod3_mul, fast_divmod3_shr);
        } else if(num_anchors == 6) {
            fast_divmod(hw, anchor, prior_in_layer, 6, fast_divmod6_mul, fast_divmod6_shr);
        } else if(num_anchors == 4) {
            fast_divmod(hw, anchor, prior_in_layer, 4, fast_divmod4_mul, fast_divmod4_shr);
        } else {
            assert(0);
        }

        int num_ch = num_anchors * num_classes * num_dim;
        int ch = (anchor*num_classes+c)*num_dim + i;

        if(packed32_nchw) {
            int packed_num_ch = (num_ch+31)/32;
            
            int packed_ch = ch >> 5; // ch/32;
            int packed_ch_offset = ch & 31; // ch%32;

            mappedIndex = ((n * packed_num_ch + packed_ch)*num_hw + hw)*32 + packed_ch_offset;
        }
        else {
            mappedIndex = (n * num_ch + ch)*num_hw + hw;
        }


        float result = conf_data_layer[mappedIndex];

        if (confSigmoid)
            result = __expf(result) / (1 + __expf(result));

        return result;
    }
}

template <int ITEMS_PER_THREAD, typename Dtype = float, int NUM_LAYERS = 6>
__global__ void top_k_cuda_fused_prepare_permute(int *in, int *out, int* out_indices, int* active_counts, int* active_count_per_batch, int items, unsigned int num_top_k, int segments, int background_class_id, float threshold,
                                                 // parameters for permuteConfData
                                                 const int num_classes, int num_classes_mul, int num_classes_shr,
                                                 const int num_priors, int num_priors_mul, int num_priors_shr,
                                                 const int num_dim, int num_dim_mul, int num_dim_shr,
                                                 int fast_divmod3_mul, int fast_divmod3_shr,
                                                 int fast_divmod6_mul, int fast_divmod6_shr,
                                                 int fast_divmod4_mul, int fast_divmod4_shr,
                                                 bool confSigmoid,
                                                 const PermuteConfData<Dtype, NUM_LAYERS> permute_conf_data
)
{
  extern  __shared__ int2 dynamic_smem[];
  int2* selected_elements = dynamic_smem;
  __shared__ unsigned int selected_count;
  // stores the number of elements which are above the threshold
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
              thread_items[i] = __float_as_int(permuteConfData<float, 6>(
                                                                         index,
                                                                         num_classes,num_classes_mul, num_classes_shr,
                                                                         num_priors, num_priors_mul, num_priors_shr,
                                                                         num_dim, num_dim_mul, num_dim_shr,
                                                                         fast_divmod3_mul, fast_divmod3_shr,
                                                                         fast_divmod6_mul, fast_divmod6_shr,
                                                                         fast_divmod4_mul, fast_divmod4_shr,
                                                                         confSigmoid,
                                                                         permute_conf_data));
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
      active_counts[segment] = active_count;
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
    void *active_count_per_batch_gpu,
    void* workspace,
    const int num_priors,
    const int num_dim,
    bool confSigmoid,
    const void* const* conf_data,
    const int num_layers,
    const int* feature_size,
    const int * num_anchors,
    const bool packed32_nchw)
{

    const int num_segments = num * num_classes;

    uint32_t smem_size = num_top_k * (sizeof(int) + sizeof(uint32_t));
    uint32_t num_warps = (num_preds_per_class > 128) ? (128/32) : (num_preds_per_class + 31) / 32;

    dim3 block(num_warps * 32);
    dim3 grid(num_classes, num);


    using Dtype = float;
    constexpr int NUM_LAYERS = 6;
    using top_k_kernel = void (*)(int *in, int *out, int* out_indices, int* active_acount, int* active_count_per_batch,
                                  int items, unsigned int num_top_k, int segments, int background_class_id, float threshold,
                                  const int num_classes, int num_classes_mul, int num_classes_shr,
                                  const int num_priors, int num_priors_mul, int num_priors_shr,
                                  const int num_dim, int num_dim_mul, int num_dim_shr,
                                  int fast_divmod3_mul, int fast_divmod3_shr,
                                  int fast_divmod6_mul, int fast_divmod6_shr,
                                  int fast_divmod4_mul, int fast_divmod4_shr,
                                  bool confSigmoid,
                                  const PermuteConfData<Dtype, NUM_LAYERS> permute_conf_data);
    top_k_kernel top_k_kernels[] = {
        top_k_cuda_fused_prepare_permute<1>,
        top_k_cuda_fused_prepare_permute<2>,
        top_k_cuda_fused_prepare_permute<3>,
        top_k_cuda_fused_prepare_permute<4>,
        top_k_cuda_fused_prepare_permute<5>,
        top_k_cuda_fused_prepare_permute<6>,
        top_k_cuda_fused_prepare_permute<7>,
        top_k_cuda_fused_prepare_permute<8>,
        top_k_cuda_fused_prepare_permute<9>,
        top_k_cuda_fused_prepare_permute<10>,
        top_k_cuda_fused_prepare_permute<11>,
        top_k_cuda_fused_prepare_permute<12>,
        top_k_cuda_fused_prepare_permute<13>,
        top_k_cuda_fused_prepare_permute<14>,
        top_k_cuda_fused_prepare_permute<15>,
        top_k_cuda_fused_prepare_permute<16>,
        top_k_cuda_fused_prepare_permute<17>,
        top_k_cuda_fused_prepare_permute<18>,
        top_k_cuda_fused_prepare_permute<19>,
        top_k_cuda_fused_prepare_permute<20>,
        top_k_cuda_fused_prepare_permute<21>,
        top_k_cuda_fused_prepare_permute<22>,
        top_k_cuda_fused_prepare_permute<23>,
        top_k_cuda_fused_prepare_permute<24>,
        top_k_cuda_fused_prepare_permute<25>,
        top_k_cuda_fused_prepare_permute<26>,
        top_k_cuda_fused_prepare_permute<27>,
        top_k_cuda_fused_prepare_permute<28>,
        top_k_cuda_fused_prepare_permute<29>,
        top_k_cuda_fused_prepare_permute<30>,
        top_k_cuda_fused_prepare_permute<31>,
        top_k_cuda_fused_prepare_permute<32>,
    };

    // determine constants for efficient integer division
    //printf("num_classes: %d num_priors: %d num_dim: %d\n", num_classes, num_priors, num_dim);
    uint32_t num_classes_mul, num_classes_shr;
    uint32_t num_priors_mul, num_priors_shr;
    uint32_t num_dim_mul, num_dim_shr;
    find_divisor(num_classes_mul, num_classes_shr, num_classes);
    find_divisor(num_priors_mul, num_priors_shr, num_priors);
    find_divisor(num_dim_mul, num_dim_shr, num_dim);

    uint32_t fast_divmod_3_mul, fast_divmod_3_shr;
    uint32_t fast_divmod_6_mul, fast_divmod_6_shr;
    uint32_t fast_divmod_4_mul, fast_divmod_4_shr;
    find_divisor(fast_divmod_3_mul, fast_divmod_3_shr, 3);
    find_divisor(fast_divmod_6_mul, fast_divmod_6_shr, 6);
    find_divisor(fast_divmod_4_mul, fast_divmod_4_shr, 4);

    int kernel_index = num_preds_per_class / block.x;
    // we have kernels with up to 16 registers per thread
    // increase the number of threads if necessary
    while (kernel_index >= 32) {
        kernel_index /= 2;
        num_warps *= 2;
    }

    assert(kernel_index < 32);
    // a maximum of 1024 threads is supported
    assert(num_warps * 32 <= 1024);

    PermuteConfData<Dtype, 6> permute_conf_data;

    // precompute pow2(feature_size) and end_prior_layer for each loop iteration.
    int start_layer_prior = 0;
    for (int i = 0;i < num_layers;++i) {
        permute_conf_data.feature_size[i] = feature_size[i] * feature_size[i];
        permute_conf_data.num_anchors[i] = num_anchors[i];

        int layer_prior_size = num_anchors[i] * permute_conf_data.feature_size[i];
        int end_layer_prior = start_layer_prior + layer_prior_size;

        permute_conf_data.end_layer_prior[i] = end_layer_prior;
        start_layer_prior = end_layer_prior;
    }

    permute_conf_data.packed32_nchw = packed32_nchw;
    std::memcpy(permute_conf_data.conf_data, conf_data, 6 * sizeof(void*));

    block.x = num_warps * 32;

    top_k_kernels[kernel_index]<<<grid, block, smem_size, stream>>>((int*) (conf_scores_gpu), (int*) (conf_scores_gpu), (int*)index_array_gpu,
                                                                    (int*)active_counts_gpu, (int*)active_count_per_batch_gpu,
                                                                    num_preds_per_class, num_top_k, num_segments, background_label_id, confidence_threshold,
                                                                    // permuteConfData params
                                                                    num_classes, num_classes_mul, num_classes_shr,
                                                                    num_priors, num_priors_mul, num_priors_shr,
                                                                    num_dim, num_dim_mul, num_dim_shr,
                                                                    fast_divmod_3_mul, fast_divmod_3_shr,
                                                                    fast_divmod_6_mul, fast_divmod_6_shr,
                                                                    fast_divmod_4_mul, fast_divmod_4_shr,
                                                                    confSigmoid,
                                                                    permute_conf_data

                                                                    );

    blockSort<T_SCORE>(
        (const T_SCORE*) (conf_scores_gpu), (T_SCORE*) (conf_scores_gpu),
        (const int*) (index_array_gpu), (int*) (index_array_gpu), (int*)active_counts_gpu,
        num_top_k, num_preds_per_class, num_segments, stream
    );

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerClass LAUNCH CONFIG {{{
typedef ssdStatus_t (*tkspcpFunc)(cudaStream_t,
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
                                void*,
                                const int,
                                const int,
                                bool,
                                const void* const*,
                                const int,
                                const int*,
                                const int *,
                                const bool
                                 );
struct tkspcpLaunchConfig
{
    DType_t t_score;
    tkspcpFunc function;

    tkspcpLaunchConfig(DType_t t_score)
        : t_score(t_score)
    {
    }
    tkspcpLaunchConfig(DType_t t_score, tkspcpFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const tkspcpLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

static std::vector<tkspcpLaunchConfig> tkspcpFuncVec;
bool tkspcpInit()
{
    tkspcpFuncVec.push_back(tkspcpLaunchConfig(DataType::kFLOAT,
                                           topKScoresPerClass_gpu<float>));
    return true;
}

static bool initialized = tkspcpInit();
//}}}

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
    void* active_counts_gpu,
    void* active_count_per_batch_gpu,
    void* workspace,
    const int num_priors,
    const int num_dim,
    bool confSigmoid,
    const void* const* conf_data,
    const int num_layers,
    const int* feature_size,
    const int* num_anchors,
    const bool packed32_nchw)
{
    tkspcpLaunchConfig lc = tkspcpLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < tkspcpFuncVec.size(); ++i)
    {
        if (lc == tkspcpFuncVec[i])
        {
            DEBUG_PRINTF("sortScoresPerClassPermute kernel %d\n", i);
            return tkspcpFuncVec[i].function(stream,
                                             num,
                                             num_classes,
                                             num_preds_per_class,
                                             num_top_k,
                                             background_label_id,
                                             confidence_threshold,
                                             conf_scores_gpu,
                                             index_array_gpu,
                                             active_counts_gpu,
                                             active_count_per_batch_gpu,
                                             workspace,
                                             num_priors,
                                             num_dim,
                                             confSigmoid,
                                             conf_data,
                                             num_layers,
                                             feature_size,
                                             num_anchors,
                                             packed32_nchw);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t topKScoresPerClassFusedPermuteWorkspaceSize(
    const int num,
    const int num_classes,
    const int num_preds_per_class,
    const int num_top_k,
    const DType_t DT_CONF)
{
    return num * 0;
}

} // namespace plugin
} // namespace nvinfer1
