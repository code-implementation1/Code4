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
"""LibraRcnn dataset"""
from __future__ import division

import os
import shutil
import random
import logging

from functools import partial

import numpy as np
import cv2
import mindspore as ms
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
from pycocotools.coco import COCO


class Expand:
    """expand image"""

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if np.random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


def rescale_with_tuple(img, scale):
    """Rescale image according scale tuple.

    Args:
        img: image
        scale (tuple): scale tuple

    Returns:
        rescaled_img: image after rescaling
        scale_factor: proportion between original and new image size.
    """
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(
        h * float(scale_factor) + 0.5)
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor


def rescale_with_factor(img, scale_factor):
    """Rescale image according scale tuple.

    Args:
        img: image
        scale_factor (float): show how rescale image

    Returns:
        rescaled_img: image after rescaling
    """
    h, w = img.shape[:2]
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)


def pad_img(img, config):
    """Pad image to needed size"""
    img_data = img
    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]

    assert ((pad_h >= 0) and (pad_w >= 0)), \
        f'{config.img_width}, {img_data.shape[1]}, ' \
        f'{config.img_height}, {img_data.shape[0]}'
    pad_img_data = np.zeros(
        (config.img_height, config.img_width, 3)
    ).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data
    return pad_img_data


def rescale_column(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """rescale operation for image"""
    img_data, scale_factor = resize_img(img, config)

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    pad_img_data = pad_img(img_data, config)

    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return pad_img_data, img_shape, gt_bboxes, gt_label, gt_num


def rescale_column_test(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """rescale operation for image of eval"""
    img_data, scale_factor = resize_img(img, config)

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    pad_img_data = pad_img(img_data, config)

    img_shape = np.append(img_shape, img_data.shape[:2])
    img_shape = np.append(img_shape, pad_img_data.shape[:2])
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return pad_img_data, img_shape, gt_bboxes, gt_label, gt_num


def resize_column(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """resize operation for image"""
    img_data = img
    h, w = img_data.shape[:2]
    img_data = cv2.resize(
        img_data, (config.img_width, config.img_height),
        interpolation=cv2.INTER_LINEAR
    )
    h_scale = config.img_height / h
    w_scale = config.img_width / w

    scale_factor = np.array(
        [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return img_data, img_shape, gt_bboxes, gt_label, gt_num


def resize_column_test(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """resize operation for image of eval"""
    img_data = img
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
    img_shape = np.append(img_shape, (h_scale, w_scale))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    gt_bboxes = gt_bboxes * scale_factor

    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

    return img_data, img_shape, gt_bboxes, gt_label, gt_num


def impad_to_multiple_column(
        img, img_shape, gt_bboxes, gt_label, gt_num, config
):
    """impad operation for image"""
    img_data = cv2.copyMakeBorder(
        img, 0, config.img_height - img.shape[0], 0,
        config.img_width - img.shape[1], cv2.BORDER_CONSTANT, value=0
    )
    img_data = img_data.astype(np.float32)
    return img_data, img_shape, gt_bboxes, gt_label, gt_num


def imnormalize_column(img, img_shape, gt_bboxes, gt_label, gt_num, config):
    """imnormalize operation for image"""
    # Computed from random subset of ImageNet training images
    mean = np.asarray(config.img_mean)
    std = np.asarray(config.img_std)
    img_data = img.copy().astype(np.float32)
    if config.to_rgb:
        cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB, img_data)  # inplace
    cv2.subtract(img_data, np.float64(mean.reshape(1, -1)), img_data)  # inplace
    cv2.multiply(img_data, 1 / np.float64(std.reshape(1, -1)), img_data)  # inplace

    img_data = img_data.astype(np.float32)
    return img_data, img_shape, gt_bboxes, gt_label, gt_num


def flip_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """flip operation for image"""
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1

    return img_data, img_shape, flipped, gt_label, gt_num


def transpose_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool)
    return img_data, img_shape, gt_bboxes, gt_label, gt_num


def expand_column(img, img_shape, gt_bboxes, gt_label, gt_num):
    """expand operation for image"""
    expand = Expand()
    img, gt_bboxes, gt_label = expand(img, gt_bboxes, gt_label)

    return img, img_shape, gt_bboxes, gt_label, gt_num


def data_to_mindrecord_byte_image(
        config, img_path, mindrecord_path, file_num=8
):
    """Create MindRecord file."""
    os.makedirs(mindrecord_path, exist_ok=False)
    path = os.path.join(mindrecord_path, 'file.mindrecord')
    writer = FileWriter(path, file_num)
    train_cls = config.coco_classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(img_path, 'labels.json')
    shutil.copy(anno_json, mindrecord_path)
    image_dir = os.path.join(img_path, 'data')

    coco = COCO(anno_json)
    classes_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classes_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}
    img_id_dict = {}
    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(image_dir, file_name)
        annos = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classes_dict[label["category_id"]]
            if class_name in train_cls:
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append(
                    [x1, y1, x2, y2] + [train_cls_dict[class_name]] +
                    [int(label["iscrowd"])]
                )

        image_files.append(image_path)
        if annos:
            image_anno_dict[image_path] = np.array(annos)
        else:
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])

        img_id_dict[image_path] = img_id

    rcnn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
        "img_id": {"type": "int32"}
    }
    writer.add_schema(rcnn_json, "rcnn_json")

    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        img_id = np.int32(img_id_dict[image_name])
        row = {"image": img, "annotation": annos, "img_id": img_id}
        writer.write_raw_data([row])
    writer.commit()


def create_mindrecord_dataset(
        path, config, training=False, python_multiprocessing=False,
        sampler=None
):
    """Create LibraRcnn dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(1)
    path = os.path.join(path, 'file.mindrecord')
    if not os.path.exists(path):
        path = path + '0'
    if training:
        shuffle = True if (sampler is None) else None
        num_shards = None if config.device_num == 1 else config.device_num
        shard_id = None if config.device_num == 1 else config.rank_id
        ds = de.MindDataset(
            path, columns_list=["image", "annotation", "img_id"],
            num_shards=num_shards, shard_id=shard_id,
            num_parallel_workers=config.num_parallel_workers,
            shuffle=shuffle
        )
    else:
        if config.rank_id == 0:
            ds = de.MindDataset(
                path, columns_list=["image", "annotation", "img_id"],
                num_parallel_workers=4, shuffle=training
            )
        else:
            return None
    decode = ms.dataset.vision.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = partial(
        preprocess_mindrecord, training=training, config=config
    )
    ds = ds.map(
        input_columns=["image", "annotation", "img_id"],
        output_columns=["image", "image_shape", "box", "label", "valid_num"],
        operations=compose_map_func,
        python_multiprocessing=bool(python_multiprocessing),
        num_parallel_workers=config.num_parallel_workers
    ).project(['image', 'image_shape', 'box', 'label', 'valid_num'])
    batch_size = config.batch_size if training else config.test_batch_size
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    return ds


def create_coco_det_dataset(
        path, config, training=False, python_multiprocessing=False,
        sampler=None
):
    """Create COCO dataset based on original data (raw images and labels)."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(1)
    if training:
        shuffle = True if (sampler is None) else None
        num_shards = None if config.device_num == 1 else config.device_num
        shard_id = None if config.device_num == 1 else config.rank_id
        dataset = ms.dataset.CocoDataset(
            dataset_dir=os.path.join(path, 'data'),
            annotation_file=os.path.join(path, 'labels.json'),
            task='Detection',
            num_shards=num_shards, shard_id=shard_id,
            sampler=sampler, shuffle=shuffle,
        )
    else:
        dataset = ms.dataset.CocoDataset(
            dataset_dir=os.path.join(path, 'data'),
            annotation_file=os.path.join(path, 'labels.json'),
            task='Detection', shuffle=training
        )
    name_map = dataset.get_class_indexing()
    classes_map = {
        name_map[k][0]: i + 1 for i, k in enumerate(name_map.keys())
    }
    config.classes_map = classes_map

    decode = ms.dataset.vision.Decode()
    dataset = dataset.map(input_columns=["image"], operations=decode)
    compose_map_func = partial(
        preprocess_coco, training=training, config=config
    )
    dataset = dataset.map(
        input_columns=['image', 'bbox', 'category_id', 'iscrowd'],
        output_columns=[
            'image', 'image_shape', 'box', 'label', 'valid_num'
        ],
        operations=compose_map_func,
        python_multiprocessing=python_multiprocessing,
        num_parallel_workers=config.num_parallel_workers
    ).project(['image', 'image_shape', 'box', 'label', 'valid_num'])

    batch_size = config.batch_size if training else config.test_batch_size
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def preprocess_mindrecord(image, box, img_id, training, config):
    """Preprocess function for dataset."""

    def _infer_data(
            image_bgr, image_shape, gt_box_new, gt_label_new,
            gt_iscrowd_new_revert
    ):
        image_shape = image_shape[:2]
        input_data = (
            image_bgr, image_shape, gt_box_new, gt_label_new,
            gt_iscrowd_new_revert
        )

        if config.keep_ratio:
            input_data = rescale_column_test(*input_data, config=config)
        else:
            input_data = resize_column_test(*input_data, config=config)
        input_data = imnormalize_column(*input_data, config=config)

        output_data = transpose_column(*input_data)
        output_data = (
            output_data[0],
            np.append(output_data[1], img_id).astype('float32'),
            *output_data[2:]
        )
        return output_data

    def _data_aug(image, box, training):
        """Data augmentation function."""
        pad_max_number = config.num_gts
        if pad_max_number < box.shape[0]:
            box = box[:pad_max_number, :]
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gt_box = box[:, :4]
        gt_label = box[:, 4]
        if not training:
            gt_label = gt_label - 1
        gt_iscrowd = box[:, 5]

        gt_box_new = np.pad(
            gt_box, ((0, pad_max_number - box.shape[0]), (0, 0)),
            mode="constant", constant_values=0
        )
        gt_label_new = np.pad(
            gt_label, ((0, pad_max_number - box.shape[0])), mode="constant",
            constant_values=-1
        )
        gt_iscrowd_new = np.pad(
            gt_iscrowd, ((0, pad_max_number - box.shape[0])), mode="constant",
            constant_values=1
        )
        gt_iscrowd_new_revert = (~(gt_iscrowd_new.astype(np.bool))).astype(
            np.int32
        )

        if not training:
            return _infer_data(
                image_bgr, image_shape, gt_box_new, gt_label_new,
                gt_iscrowd_new_revert
            )

        flip = (np.random.rand() < config.flip_ratio)
        expand = (np.random.rand() < config.expand_ratio)
        input_data = (
            image_bgr, image_shape, gt_box_new, gt_label_new,
            gt_iscrowd_new_revert
        )

        if expand:
            input_data = expand_column(*input_data)
        if config.keep_ratio:
            input_data = rescale_column(*input_data, config=config)
        else:
            input_data = resize_column(*input_data, config=config)
        input_data = imnormalize_column(*input_data, config=config)
        if flip:
            input_data = flip_column(*input_data)

        output_data = transpose_column(*input_data)
        return output_data

    return _data_aug(image, box, training)


def preprocess_coco(image, bbox, category_id, iscrowd, training, config):
    """Preprocess function for COCO dataset."""
    pad_max_number = config.num_gts
    if pad_max_number < bbox.shape[0]:
        bbox = bbox[:pad_max_number, :]
        category_id = category_id[:pad_max_number, :]
        iscrowd = iscrowd[:pad_max_number, :]
    image_bgr = image[..., ::-1].copy()
    image_shape = image_bgr.shape[:2]

    gt_bbox = bbox
    gt_bbox[::, [2, 3]] = gt_bbox[::, [0, 1]] + gt_bbox[::, [2, 3]]

    gt_label = category_id
    gt_label = np.array([[config.classes_map[i]] for i in gt_label[::, 0]])
    if not training:
        gt_label = gt_label - 1
    gt_iscrowd = iscrowd

    gt_box_new = np.pad(
        gt_bbox, ((0, pad_max_number - bbox.shape[0]), (0, 0)),
        mode="constant", constant_values=0
    )
    gt_label_new = np.pad(
        gt_label, ((0, pad_max_number - bbox.shape[0]), (0, 0)),
        mode="constant",
        constant_values=-1
    )
    gt_iscrowd_new = np.pad(
        gt_iscrowd, ((0, pad_max_number - bbox.shape[0]), (0, 0)),
        mode="constant",
        constant_values=1
    )
    gt_iscrowd_new_revert = (~(gt_iscrowd_new.astype(np.bool_))).astype(
        np.int32
    )

    image_shape = image_shape[:2]
    input_data = (
        image_bgr, image_shape, gt_box_new, gt_label_new,
        gt_iscrowd_new_revert
    )
    if training:
        expand = (np.random.rand() < config.expand_ratio)
        if expand:
            input_data = expand_column(*input_data)
        if config.keep_ratio:
            input_data = rescale_column(*input_data, config=config)
        else:
            input_data = resize_column(*input_data, config=config)
        flip = (np.random.rand() < config.flip_ratio)
        if flip:
            input_data = flip_column(*input_data)
    else:
        if config.keep_ratio:
            input_data = rescale_column_test(*input_data, config=config)
        else:
            input_data = resize_column_test(*input_data, config=config)

    input_data = imnormalize_column(*input_data, config=config)

    output_data = transpose_column(*input_data)
    return output_data


def resize_img(img, config):
    """Resize image according configuration."""
    if config.keep_ratio:
        img_data, scale_factor = rescale_with_tuple(
            img, (config.img_width, config.img_height)
        )
        if img_data.shape[0] > config.img_height:
            img_data, scale_factor2 = rescale_with_tuple(
                img_data, (config.img_height, config.img_height)
            )
            scale_factor = scale_factor * scale_factor2
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


def create_coco_label(is_training, config):
    """Get image path and annotation from COCO."""
    coco_root = config.coco_root
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type

    # Classes need to train or test.
    train_cls = config.coco_classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instance_set.format(data_type))
    if hasattr(config, 'train_set') and is_training:
        anno_json = os.path.join(coco_root, config.train_set)
    if hasattr(config, 'val_set') and not is_training:
        anno_json = os.path.join(coco_root, config.val_set)
    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}

    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            if class_name in train_cls:
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append(
                    [x1, y1, x2, y2] + [train_cls_dict[class_name]] +
                    [int(label["iscrowd"])]
                )

        image_files.append(image_path)
        if annos:
            image_anno_dict[image_path] = np.array(annos)
        else:
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])

    return image_files, image_anno_dict


def parse_json_annos_from_txt(anno_file, config):
    """for user defined annotations text file, parse it to json format data"""
    if not os.path.isfile(anno_file):
        raise RuntimeError(
            f'Evaluation annotation file {anno_file} is not valid.'
        )

    annos = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # set categories field
    for i, cls_name in enumerate(config.coco_classes):
        annos["categories"].append({"id": i, "name": cls_name})

    with open(anno_file, "rb") as f:
        lines = f.readlines()

    img_id = 1
    anno_id = 1
    for line in lines:
        line_str = line.decode("utf-8").strip()
        line_split = str(line_str).split(' ')
        # set image field
        file_name = line_split[0]
        annos["images"].append({"file_name": file_name, "id": img_id})
        # set annotations field
        for anno_info in line_split[1:]:
            anno = anno_info.split(",")
            x = float(anno[0])
            y = float(anno[1])
            w = float(anno[2]) - float(anno[0])
            h = float(anno[3]) - float(anno[1])
            category_id = int(anno[4])
            iscrowd = int(anno[5])
            annos["annotations"].append({"bbox": [x, y, w, h],
                                         "area": w * h,
                                         "category_id": category_id,
                                         "iscrowd": iscrowd,
                                         "image_id": img_id,
                                         "id": anno_id})
            anno_id += 1
        img_id += 1

    return annos


def create_train_data_from_txt(image_dir, anno_path):
    """Filter valid image file, which both in image_dir and anno_path."""

    def anno_parser(annos_str):
        """Parse annotation from string to list."""
        annos = []
        for anno_str in annos_str:
            anno = anno_str.strip().split(",")
            xmin, ymin, xmax, ymax = list(map(float, anno[:4]))
            cls_id = int(anno[4])
            iscrowd = int(anno[5])
            annos.append([xmin, ymin, xmax, ymax, cls_id, iscrowd])
        return annos

    image_files = []
    image_anno_dict = {}
    if not os.path.isdir(image_dir):
        raise RuntimeError("Path given is not valid.")
    if not os.path.isfile(anno_path):
        raise RuntimeError("Annotation file is not valid.")

    with open(anno_path, "rb") as f:
        lines = f.readlines()
    for line in lines:
        line_str = line.decode("utf-8").strip()
        line_split = str(line_str).split(' ')
        file_name = line_split[0]
        image_path = os.path.join(image_dir, file_name)
        if os.path.isfile(image_path):
            image_anno_dict[image_path] = anno_parser(line_split[1:])
            image_files.append(image_path)
    return image_files, image_anno_dict


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

    if config.train_data_type == 'coco':
        train_dataset = create_coco_det_dataset(
            path=config.train_dataset, config=config, training=True,
            python_multiprocessing=bool(config.python_multiprocessing),
            sampler=sampler
        )
    elif config.train_data_type == 'mindrecord':
        train_dataset = create_mindrecord_dataset(
            path=config.train_dataset, config=config, training=True,
            python_multiprocessing=bool(config.python_multiprocessing),
            sampler=sampler
        )
    else:
        raise ValueError(
            f'Unsupported train dataset type: {config.train_data_type}'
        )

    if config.rank_id == 0 and config.run_eval:
        # if config.val_data_type == 'coco':
        #     val_dataset = create_coco_det_dataset(
        #         path=config.val_dataset, config=config, training=False,
        #         python_multiprocessing=bool(config.python_multiprocessing)
        #     )
        if config.val_data_type == 'mindrecord':
            val_dataset = create_mindrecord_dataset(
                path=config.val_dataset, config=config, training=False,
                python_multiprocessing=bool(config.python_multiprocessing)
            )
        else:
            raise ValueError(
                f'Unsupported validation dataset type: {config.val_data_type}'
            )
    else:
        val_dataset = None

    return train_dataset, val_dataset
