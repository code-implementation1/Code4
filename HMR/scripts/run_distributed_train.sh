#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "bash run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH]  "
    echo "bash run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] "
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
export RANK_TABLE_FILE=$PATH1
echo "RANK_TABLE_FILE=${PATH1}"

if [ $# == 3 ]
then 
    PATH3=$(get_real_path $3)
fi

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))


cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ./*.py ./train_parallel$i
    cp -r ./*.yaml ./train_parallel$i
    cp -r ./src ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log
    if [ $# == 2 ]
    then
    taskset -c $cmdopt python ./trainer_hmr.py --run_distribute=True --data_path=$PATH2 > train.log 2>&1 &
    fi

    if [ $# == 3 ]
    then
    taskset -c $cmdopt python ./trainer_hmr.py --run_distribute=True --data_path=$PATH2  --checkpoint_file_path=$PATH3  > train.log 2>&1 &
    fi
    cd ..
done
