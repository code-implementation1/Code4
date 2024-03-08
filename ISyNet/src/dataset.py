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
"""
create train or eval dataset.
"""
import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.dataset.vision import Inter
from mindspore.communication.management import init, get_rank, get_group_size
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_num, get_rank_id
from src.autoaugment import ImageNetPolicy
from src.transform import RandAugment


class ToNumpy:
    def __init__(self):
        pass
    def __call__(self, img):
        return np.asarray(img)

def create_dataset1(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                    enable_cache=False, cache_session_id=None, autoaugment=False):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
    ds.config.set_prefetch_size(64)
    if device_num == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0)
    ]
    if autoaugment:
        trans += [
            C.ToPIL(),
            ImageNetPolicy(),
            ToNumpy(),
        ]
    trans += [
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8, cache=eval_cache)
    else:
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                    enable_cache=False, cache_session_id=None, autoaugment=False, randaugment=False):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1

    ds.config.set_prefetch_size(64)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        if randaugment:
            trans = [
                C.Decode(),
                C.RandomResizedCrop(size=(image_size, image_size),
                                    scale=(0.08, 1.0),
                                    ratio=(3. / 4., 4. / 3.),
                                    interpolation=Inter.BICUBIC),
                C.RandomHorizontalFlip(prob=0.5),
            ]
        else:
            trans = [
                C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                C.RandomHorizontalFlip(prob=0.5)
                ]
            if autoaugment:
                trans += [
                    C.ToPIL(),
                    ImageNetPolicy(),
                    ToNumpy(),
                    ]
            trans += [
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
    else:
        randaugment = False
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12,
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)

    # apply batch operations
    if randaugment:
        efficient_rand_augment = RandAugment()
        data_set = data_set.batch(batch_size,
                                  per_batch_map=efficient_rand_augment,
                                  input_columns=['image', 'label'],
                                  num_parallel_workers=2,
                                  drop_remainder=True)
    else:
        data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set

def create_dataset_pynative(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                            enable_cache=False, cache_session_id=None, autoaugment=False):
    """
    create a train or eval imagenet2012 dataset for resnet50 benchmark

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    ds.config.set_numa_enable(True)
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=2, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5)
            ]
        if autoaugment:
            trans += [
                C.ToPIL(),
                ImageNetPolicy(),
                ToNumpy(),
                ]
        trans += [
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
            ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=4)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=2,
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=2)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set

def create_dataset3(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                    enable_cache=False, cache_session_id=None, autoaugment=False):
    """
    create a train or eval imagenet2012 dataset for resnet101
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
            rank_id = 1
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)
    image_size = 224
    mean = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std = [0.275 * 255, 0.267 * 255, 0.278 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(rank_id / (rank_id + 1))
            ]
        if autoaugment:
            trans += [
                C.ToPIL(),
                ImageNetPolicy(),
                ToNumpy(),
                ]

        trans += [
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
            ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8,
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset4(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                    enable_cache=False, cache_session_id=None, autoaugment=False):
    """
    create a train or eval imagenet2012 dataset for se-resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
    ds.config.set_prefetch_size(64)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)
    image_size = 224
    mean = [123.68, 116.78, 103.94]
    std = [1.0, 1.0, 1.0]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5)
                ]
        if autoaugment:
            trans += [
                C.ToPIL(),
                ImageNetPolicy(),
                ToNumpy(),
                ]

        trans += [
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
                ]
    else:
        trans = [
            C.Decode(),
            C.Resize(292),
            C.CenterCrop(256),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12,
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if config.device_target == "Ascend":
        if rank_size > 1:
            rank_size = get_device_num()
            rank_id = get_rank_id()
        else:
            rank_size = 1
            rank_id = 0
    else:
        if rank_size > 1:
            rank_size = get_group_size()
            rank_id = get_rank()
        else:
            rank_size = 1
            rank_id = 0

    return rank_size, rank_id
