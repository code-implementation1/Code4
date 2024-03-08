#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 3 ]; then
  echo "Usage: bash run_eval_blufr_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]"
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

export DEVICE_ID=$3

if [ -d "eval_blufr" ];
then
    rm -rf ./eval_blufr
fi

mkdir ./eval_blufr
cp ../*.py ./eval_blufr
cp *.sh ./eval_blufr
cp -r ../src ./eval_blufr
cd ./eval_blufr || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"

python eval_blufr.py --dataset_path=$PATH1 --resume=$PATH2 \
                   --device_id=$DEVICE_ID --device_target="GPU" > log 2>&1 &
cd ..
