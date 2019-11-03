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
#include <common.h>

size_t get_stringLength(std::string input)
{
    std::istringstream sentence(input);

    size_t length = 0;
    while (!sentence.eof())
    {
        std::string token;
        sentence >> token;

        // If sentence ends with empty string, we should stop inserting indices
        if (token == "" && (sentence.tellg() == -1))
        {
            break;
        }

        length++;
    }
    return length;
}
