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
"""dataset"""
from __future__ import division

from collections import defaultdict

import json
import os
import copy
import random
import logging

from functools import partial
from PIL import Image

import numpy as np
import cv2
import mindspore as ms
import mindspore.dataset as de
from pycocotools import mask as mask_utils


def preprocess_coco(
        image, label, bboxes, segm, iscrowd, img_id, training, config
):
    """Preprocess function for COCO dataset."""
    pad_max_number = config.num_gts

    if pad_max_number < bboxes.shape[0]:
        bboxes = bboxes[:pad_max_number]
        label = label[:pad_max_number]
        iscrowd = iscrowd[:pad_max_number]
        segm = segm[:pad_max_number]

    bboxes = bboxes.reshape(-1, 4)
    label = label.reshape(-1, 1)
    iscrowd = iscrowd.reshape(-1, 1)

    if training:
        label = label + 1

    image_bgr = image[..., ::-1].copy()
    image_shape = image_bgr.shape[:2]

    bboxes = process_bboxes(bboxes, image_shape)
    seg_masks = convert_seg2mask(
        segm, bboxes=bboxes, shape=image_shape
    )

    input_data = {
        'image': image_bgr,
        'img_shape': image_shape,
        'labels': label,
        'bboxes': bboxes,
        'segm_masks': seg_masks,
        'iscrowd': iscrowd,
        'img_id': img_id
    }

    input_data = pad_data(input_data, pad_max_number=pad_max_number)

    if training:
        flip = (np.random.rand() < config.flip_ratio)
        if flip:
            input_data = flip_column(input_data)

    if config.keep_ratio:
        input_data = rescale_column_test(input_data, config=config)
    else:
        input_data = resize_column_test(input_data, config=config)

    input_data = resize_annotation(input_data, config)
    input_data = pad_img(input_data, config)
    input_data = pad_annotation(input_data, config)
    input_data = create_meta(input_data)
    input_data = imnormalize_column(input_data, config=config)
    output_data = transpose_column(input_data)
    columns = [
        'tensor', 'img_meta', 'bboxes', 'segm_masks', 'labels', 'valid'
    ]

    return tuple([output_data[k] for k in columns])


def create_meta(data):
    new_data = data.copy()
    img_meta = np.concatenate(
        [
            data['img_shape'], data['resized_shape'], data['pad_shape'],
            np.array([data['img_id']])
        ]
    )
    new_data['img_meta'] = img_meta
    return new_data


def process_bboxes(bboxes, shape):
    bboxes = bboxes.copy()
    bboxes[::, [2, 3]] = bboxes[::, [0, 1]] + bboxes[::, [2, 3]]
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, shape[1] - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, shape[0] - 1)
    return bboxes


def convert_seg2mask(segms, bboxes, shape):
    h, w = shape
    result = []
    for segm_rle, _ in zip(segms, bboxes):
        segm_rle = {
            'counts': segm_rle.tobytes(),
            'size': [h, w]
        }
        mask = mask_utils.decode(segm_rle)
        result.append(mask)

    return np.array(result).reshape(-1, *shape)


def pad_data(data, pad_max_number):
    new_data = data.copy()
    new_data['bboxes'] = np.pad(
        data['bboxes'], ((0, pad_max_number - data['bboxes'].shape[0]), (0, 0)),
        mode='constant', constant_values=0
    )
    new_data['labels'] = np.pad(
        data['labels'], ((0, pad_max_number - data['bboxes'].shape[0]), (0, 0)),
        mode='constant', constant_values=-1
    )
    new_data['iscrowd'] = np.pad(
        data['iscrowd'], ((0, pad_max_number - data['bboxes'].shape[0]), (0, 0)),
        mode='constant', constant_values=1
    )
    new_data['valid'] = (~(new_data['iscrowd'].astype(np.bool_))).astype(
        np.int32
    )
    new_data['segm_masks'] = np.pad(
        data['segm_masks'], ((0, pad_max_number - data['bboxes'].shape[0]), (0, 0), (0, 0)),
        mode='constant', constant_values=0
    )

    return new_data


def flip_column(data):
    """Flip operation for image."""
    new_data = data.copy()
    img_data = data['image'].copy()
    img_data = np.flip(img_data, axis=1)
    bboxes = data['bboxes'].copy()
    flipped_bboxes = data['bboxes'].copy()
    _, w, _ = img_data.shape

    flipped_bboxes[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped_bboxes[..., 2::4] = w - bboxes[..., 0::4] - 1

    new_data['image'] = img_data
    segm_masks = data['segm_masks'].copy()
    flipped_segm_masks = segm_masks[..., ::-1]

    new_data['bboxes'] = flipped_bboxes
    new_data['segm_masks'] = flipped_segm_masks

    return new_data


def rescale_column_test(data, config):
    """rescale operation for image of eval"""
    new_data = data.copy()
    img_data, scale_factor = resize_img(data['image'], config)

    new_data['image'] = img_data
    new_data['scale_factor'] = np.array(scale_factor)
    new_data['resized_shape'] = img_data.shape[:2]
    return new_data


def resize_img(img, config):
    """Resize image according configuration."""
    if config.keep_ratio:
        img_data, scale_factor = rescale_according_size(
            img, (config.img_width, config.img_height)
        )
    else:
        img_data = img
        h, w = img_data.shape[:2]
        img_data = cv2.resize(
            img_data, (config.img_width, config.img_height),
            interpolation=cv2.INTER_LINEAR)
        h_scale = config.img_height / h
        w_scale = config.img_width / w
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
        )

    return img_data, scale_factor


def resize_column_test(input_data, config):
    """resize operation for image of eval"""
    new_data = input_data.copy()
    img_data = input_data['image'].copy()

    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.img_width, config.img_height),
        interpolation=cv2.INTER_LINEAR
    )
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
    )

    new_data['image'] = img_data
    new_data['scale_factor'] = np.array(scale_factor)
    new_data['resized_shape'] = img_data.shape[:2]

    return new_data


def resize_annotation(data, config):
    bboxes = data['bboxes'].copy()
    scale_factor = data['scale_factor'].copy()

    bboxes = bboxes * scale_factor

    segm_masks = data['segm_masks'].copy()
    h, w = data['resized_shape']
    h //= config.mask_divider
    w //= config.mask_divider
    segm_masks = np.array([
        cv2.resize(mask, dsize=(w, h)) for mask in segm_masks
    ])

    new_data = data.copy()
    new_data['bboxes'] = bboxes
    new_data['segm_masks'] = segm_masks

    return new_data


def pad_annotation(data, config):
    segm_masks = data['segm_masks'].copy()
    h, w = data['resized_shape']
    h //= config.mask_divider
    w //= config.mask_divider
    pad_h, pad_w = data['pad_shape']
    pad_h //= config.mask_divider
    pad_w //= config.mask_divider
    new_segm_masks = np.pad(
        segm_masks, [(0, 0), (0, pad_h - h), (0, pad_w - w)]
    )

    new_data = data.copy()
    new_data['segm_masks'] = new_segm_masks

    return new_data


def pad_img(data, config):
    """Pad image to needed size"""
    img_data = data['image']
    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]

    assert ((pad_h >= 0) and (pad_w >= 0)), \
        f'{config.img_width}, {img_data.shape[1]}, ' \
        f'{config.img_height}, {img_data.shape[0]}'
    pad_img_data = np.zeros(
        (config.img_height, config.img_width, 3)
    ).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    new_data = data.copy()
    new_data['image'] = pad_img_data
    new_data['pad_shape'] = pad_img_data.shape[:2]
    return new_data


def rescale_according_size(img, size):
    """Rescale image according size with ratio saving.

    Args:
        img: image
        size (tuple): scale tuple

    Returns:
        rescaled_img: image after rescaling
        scale_factor: proportion between original and new image size.
    """
    h, w = img.shape[:2]

    scale_factor = min(size[1] / h, size[0] / w)
    new_size = (
        int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    )
    assert new_size[0] <= size[0] and new_size[1] <= size[1]
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor


def imnormalize_column(data, config):
    """Imnormalize operation for image."""
    # Computed from random subset of ImageNet training images
    mean = np.asarray(config.img_mean)
    std = np.asarray(config.img_std)
    img_data = data['image'].copy().astype(np.float32)
    if config.to_rgb:
        cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    new_data = data.copy()
    new_data['tensor'] = img_data
    return new_data


def transpose_column(data):
    """Transpose operation for image."""
    new_data = data.copy()
    new_data['tensor'] = data['tensor'].transpose(2, 0, 1).copy()
    new_data['tensor'] = new_data['tensor'].astype(np.float32)
    new_data['img_meta'] = data['img_meta'].astype(np.float32)
    new_data['bboxes'] = data['bboxes'].astype(np.float32)
    new_data['segm_masks'] = data['segm_masks'].astype(np.int32)
    new_data['labels'] = data['labels'].astype(np.int32)
    new_data['valid'] = data['valid'].astype(np.bool)
    return new_data


def convert_ann2rle(segm, h, w):
    """Transform the rle coco annotation (a single one) into coco style.
    In this case, one mask can contain several polygons, later leading to
    several `Annotation` objects. In case of not having a valid polygon (the
    mask is a single pixel) it will be an empty list.

    Args:
        segm: "segmetation" field form COCO annotation.
        h: Image height.
        w: Image width.

    Returns:
        dict: RLE
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = mask_utils.frPyObjects(segm, h, w)  # uncompressed RLE
    else:
        rle = segm

    return rle


class CustomCocoDataset:

    def __init__(self, img_folder, annotation_path, training, config):
        self.img_folder = img_folder
        self.files_list = os.listdir(self.img_folder)
        self.name2path = {
            fn: os.path.join(self.img_folder, fn) for fn in self.files_list
        }
        self.training = training

        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.id2name = {
            img['id']: img['file_name'] for img in self.annotations['images']
        }
        self.id2info = {
            img['id']: img for img in self.annotations['images']
        }
        self.img_ids = sorted(self.id2name.keys())
        self.length = len(self.img_ids)

        self.cat_id2label = {
            cat['id']: i
            for i, cat in enumerate(self.annotations['categories'])
        }
        self.cat_id2name = {
            cat['id']: cat['name'] for cat in self.annotations['categories']
        }

        self.id2annot = {}

        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            self.id2annot[img_id] = self.id2annot.get(
                img_id, defaultdict(list)
            )
            self.id2annot[img_id]['label'].append(
                [self.cat_id2label[ann['category_id']]]
            )
            self.id2annot[img_id]['category_id'].append(
                ann['category_id']
            )
            self.id2annot[img_id]['bbox'].append(np.array(ann['bbox']))
            self.id2annot[img_id]['segmentation'].append(ann['segmentation'])

            self.id2annot[img_id]['iscrowd'].append([ann['iscrowd']])

        use_instaboost_fast = getattr(config, 'use_instaboost_fast', None)
        if use_instaboost_fast:
            from instaboostfast import InstaBoostConfig, get_new_data
        else:
            try:
                from instaboost import InstaBoostConfig, get_new_data
            except ImportError:
                if use_instaboost_fast is None:
                    logging.warning("Can't load `instaboost` package, "
                                    'will try to load `instaboostfast`.')
                    from instaboostfast import InstaBoostConfig, get_new_data
                else:
                    raise

        self.get_new_data = get_new_data
        self.instaboost_config = InstaBoostConfig(
            action_candidate=tuple(config.action_candidate),
            action_prob=tuple(config.action_prob),
            scale=tuple(config.scale),
            dx=config.dx,
            dy=config.dy,
            theta=tuple(config.theta),
            color_prob=config.color_prob,
            heatmap_flag=config.heatmap_flag
        )
        self.aug_prob = config.aug_prob

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img_id = self.img_ids[item]
        img_name = self.id2name[img_id]
        img_path = self.name2path[img_name]
        img = np.array(Image.open(img_path).convert('RGB'))
        annot = copy.deepcopy(self.id2annot.get(img_id, defaultdict(list)))

        # Instaboost augmentation
        aug_flag = np.random.choice(
            [0, 1],
            p=[1 - self.aug_prob, self.aug_prob]
        )
        if aug_flag and self.training:
            keys = annot.keys()

            # change from our annot format to instaboost format
            instaboost_annot = [
                {
                    key: annot[key][i] for key in keys
                }
                for i in range(len(annot["label"]))
            ]
            if instaboost_annot:
                instaboost_annot, img = self.get_new_data(instaboost_annot,
                                                          img,
                                                          self.instaboost_config)
                # change from instaboost annot format to our format
                annot = {
                    key: [
                        instaboost_annot[i][key]
                        for i in range(len(instaboost_annot))
                    ]
                    for key in instaboost_annot[0].keys()
                }
        annot['segmentation'] = [
            convert_ann2rle(
                ann,
                h=self.id2info[img_id]['height'],
                w=self.id2info[img_id]['width']
            )['counts']
            for ann in annot['segmentation']
        ]

        bboxes = annot['bbox']
        segm = annot['segmentation']
        label = annot['label']
        iscrowd = annot['iscrowd']

        return img, label, bboxes, segm, iscrowd, img_id


def create_custom_coco_det_dataset(
        path, config, training=False, python_multiprocessing=False,
        sampler=None
):
    """Create COCO dataset based on original data (raw images and labels)."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(1)
    dataset_generator = CustomCocoDataset(
        img_folder=os.path.join(path, 'data'),
        annotation_path=os.path.join(path, 'labels.json'),
        training=training, config=config
    )
    if training:
        shuffle = True if (sampler is None) else None
        num_shards = None if config.device_num == 1 else config.device_num
        shard_id = None if config.device_num == 1 else config.rank_id
        dataset = ms.dataset.GeneratorDataset(
            source=dataset_generator,
            column_names=[
                'image', 'label', 'bboxes', 'segm', 'iscrowd', 'img_id'
            ],
            num_shards=num_shards, shard_id=shard_id,
            sampler=sampler, shuffle=shuffle,
        )
    else:
        dataset = ms.dataset.GeneratorDataset(
            source=dataset_generator,
            column_names=[
                'image', 'label', 'bboxes', 'segm', 'iscrowd', 'img_id'
            ],
            shuffle=training
        )

    compose_map_func = partial(
        preprocess_coco, training=training, config=config
    )
    dataset = dataset.map(
        input_columns=[
            'image', 'label', 'bboxes', 'segm', 'iscrowd', 'img_id'
        ],
        output_columns=[
            'image', 'image_shape', 'box', 'segm', 'label', 'valid_num'
        ],
        operations=compose_map_func,
        python_multiprocessing=bool(python_multiprocessing),
        num_parallel_workers=config.num_parallel_workers
    ).project(['image', 'image_shape', 'box', 'segm', 'label', 'valid_num'])

    batch_size = config.batch_size if training else config.test_batch_size
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def prepare_data(config):
    """Prepare datasets for trianing."""
    sampler = None
    if config.train_dataset_divider and config.train_dataset_num:
        random.seed(0)
        logging.info(
            'Create sampler %d/%d', config.train_dataset_num,
            config.train_dataset_divider
        )
        indices = random.sample(
            list(range(config.train_dataset_num)),
            config.train_dataset_num // config.train_dataset_divider
        )
        sampler = ms.dataset.SubsetRandomSampler(indices)

    train_dataset = create_custom_coco_det_dataset(
        path=config.train_dataset, config=config, training=True,
        python_multiprocessing=bool(config.python_multiprocessing),
        sampler=sampler
    )

    if config.rank_id == 0 and config.run_eval:
        val_dataset = create_custom_coco_det_dataset(
            path=config.val_dataset, config=config, training=False,
            python_multiprocessing=bool(config.python_multiprocessing)
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset
