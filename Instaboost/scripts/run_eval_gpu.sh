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

if [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: 
    bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (Optional)[PREDICTION_PATH]
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
VAL_DATA=$(get_real_path $2)
CHECKPOINT_PATH=$(get_real_path $3)

echo "config_path: $CONFIG_PATH"
echo "val_dataset: $VAL_DATA"
echo "checkpoint_path: $CHECKPOINT_PATH"

if [ $# -eq 4 ]
then
  PREDICTION_PATH=$(get_real_path $4)
  echo "prediction_path: $PREDICTION_PATH"
fi


if [ $# -eq 4 ]
then
  python eval.py --config_path $CONFIG_PATH --checkpoint_path $CHECKPOINT_PATH \
  --val_dataset $VAL_DATA --prediction_path $PREDICTION_PATH \
  --device_target GPU &> eval_gpu.log &
fi
if [ $# -eq 3 ]
then
  python eval.py --config_path $CONFIG_PATH --checkpoint_path $CHECKPOINT_PATH \
  --val_dataset $VAL_DATA --device_target GPU &> eval_gpu.log &
fi