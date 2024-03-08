#!/bin/bash
# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
if [ $# -ne 4 ] && [ $# -ne 5 ]
then
    echo "Usage:
    bash scripts/run_standalone_training_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] (OPTIONAL)[PRETRAINED_PATH]
    "
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_PATH=$(get_real_path $1)
TRAIN_DATA=$(get_real_path $2)
VAL_DATA=$(get_real_path $3)
TRAIN_OUT=$(get_real_path $4)

echo "config_file: $CONFIG_PATH"
echo "train_dataset: $TRAIN_DATA"
echo "val_dataset: $VAL_DATA"
echo "train_outputs: $TRAIN_OUT"

if [ $# -eq 5 ]
then
  PRETRAINED_PATH=$5
  echo "pre_trained: $PRETRAINED_PATH"
fi

if [ $# -eq 5 ]
then
  python train.py --config_path=$CONFIG_PATH --train_dataset=$TRAIN_DATA --val_dataset=$VAL_DATA \
      --pre_trained=$PRETRAINED_PATH --device_target="GPU" --train_outputs=$TRAIN_OUT
fi
if [ $# -eq 4 ]
then
  python train.py --config_path=$CONFIG_PATH --train_dataset=$TRAIN_DATA --val_dataset=$VAL_DATA \
      --device_target="GPU" --train_outputs=$TRAIN_OUT
fi