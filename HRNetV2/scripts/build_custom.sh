#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
if [ $# -ne 0 ]
then
    echo "Usage:
    bash scripts/build_custom.sh
    "
exit 1
fi

mkdir -p src/models/hrnetv2/layers/bin
nvcc --shared -Xcompiler -fPIC -o src/models/hrnetv2/layers/bin/roi_align_gpu.so src/models/hrnetv2/layers/cuda/roi_align_cuda_kernel.cu
g++ --shared -fPIC -o src/models/hrnetv2/layers/bin/roi_align_cpu.so src/models/hrnetv2/layers/cpu/roi_align.cpp
