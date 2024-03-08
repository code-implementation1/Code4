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

if [ $# != 2 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_distribute_train8p.sh [RANK_TABLE_FILE] [PROJECT_PATH]"
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_train8p.sh [RANK_TABLE_FILE] [PROJECT_PATH]"
    echo "for example: bash scripts/run_distribute_train8p.sh /absolute/path/to/RANK_TABLE_FILE /root/ICNet/"
    echo "=============================================================================================================="
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH2=$(get_real_path $2)


if [ ! -d $PATH2 ]
then
    echo "error: PROJECT_PATH=$PATH2 is not a directory"
exit 1
fi

export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=8

for((i=0;i<$RANK_SIZE;i++))
do
    rm -rf LOG$i
    mkdir ./LOG$i
    cp ./*.py ./LOG$i
    cp -r ./src ./LOG$i
    cd ./LOG$i || exit
    export RANK_TABLE_FILE=$1
    export RANK_SIZE=8
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log

    python3 train.py --project_path=$PATH2 > log.txt 2>&1 &

    cd ../
done
