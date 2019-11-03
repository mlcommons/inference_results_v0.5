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

#include <cassert>
#include <cstdio>
#include <sstream>
#include <cuda_runtime.h>
#include "ssdOpt.h"

void reportAssertion(const char* msg, const char* file, int line);


#define ASSERT(assertion)                                              \
    {                                                                  \
        if (!(assertion))                                              \
        {                                                              \
            reportAssertion(#assertion, __FILE__, __LINE__); \
        }                                                              \
    }


#ifndef DEBUG

#define SSD_ASSERT_PARAM(exp)        \
    do                               \
    {                                \
        if (!(exp))                  \
            return STATUS_BAD_PARAM; \
    } while (0)

#define SSD_ASSERT_FAILURE(exp)    \
    do                             \
    {                              \
        if (!(exp))                \
            return STATUS_FAILURE; \
    } while (0)

#define CSC(call, err)                 \
    do                                 \
    {                                  \
        cudaError_t cudaStatus = call; \
        if (cudaStatus != cudaSuccess) \
        {                              \
            return err;                \
        }                              \
    } while (0)

#define DEBUG_PRINTF(...) \
    do                    \
    {                     \
    } while (0)

#else

#define SSD_ASSERT_PARAM(exp)                                                     \
    do                                                                            \
    {                                                                             \
        if (!(exp))                                                               \
        {                                                                         \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_BAD_PARAM;                                              \
        }                                                                         \
    } while (0)

#define SSD_ASSERT_FAILURE(exp)                                                 \
    do                                                                          \
    {                                                                           \
        if (!(exp))                                                             \
        {                                                                       \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_FAILURE;                                              \
        }                                                                       \
    } while (0)

#define CSC(call, err)                                                                          \
    do                                                                                          \
    {                                                                                           \
        cudaError_t cudaStatus = call;                                                          \
        if (cudaStatus != cudaSuccess)                                                          \
        {                                                                                       \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
            return err;                                                                         \
        }                                                                                       \
    } while (0)

#define DEBUG_PRINTF(...)    \
    do                       \
    {                        \
        printf(__VA_ARGS__); \
    } while (0)

#endif

