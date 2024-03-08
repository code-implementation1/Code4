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

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
scripts_path=$PWD

device_id=0
if [ $# == 3 ]; then
    device_id=$3
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "device id: "$device_id

function preprocess_data()
{
    cd ../data/lfw/ || exit
    if [ -d bin_data ]; then
        rm -rf ./bin_data
    fi
    mkdir bin_data
    cd $scripts_path
    python ../preprocess.py --data_path=$data_path &> preprocess.log
}

function compile_app()
{   
    cd ../ascend310_infer/src/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> build.log
    cd $scripts_path
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    cd $scripts_path
    ../ascend310_infer/src/main --mindir_path=$model --dataset_path=$data_path"/bin_data" --device_id=$device_id &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --mat_files_path=../mat_files &> acc.log
}

preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess data failed"
    exit 1
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
