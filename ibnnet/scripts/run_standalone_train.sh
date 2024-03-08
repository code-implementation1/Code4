#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH EVAL_PATH CKPT_PATH"
echo "For example: bash run.sh /path/dataset /path/evalset /path/ckpt device_id"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
EXE_PATH=$(pwd)
DATA_PATH=$1
EVAL_PATH=$2
CKPT_PATH=$3
DEVICEID=$4

rm -rf "train$4"
mkdir "train$4"
cp -r ./src/ ./"train$4"
cp train.py ./"train$4"
cd ./"train$4"
export DEVICE_ID=$4
export RANK_ID=$4
echo "start training for device $4"
env > env$4.log
python train.py  \
    --epochs 100 \
    --train_url "$EXE_PATH" \
    --data_url "$DATA_PATH" \
    --eval_url "$EVAL_PATH" \
    --ckpt_url "$CKPT_PATH" \
    --pretrained \
    --device_id $DEVICEID \
    > train.log 2>&1 &
echo "start training"
cd ../
