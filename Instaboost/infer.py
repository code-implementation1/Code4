# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Infer model."""
import argparse
import os
import json

from pprint import pprint
from pathlib import Path

import mindspore as ms

import cv2
import numpy as np
from tqdm import tqdm
from src.blocks.detectors.mask_rcnn import MaskRCNNInfer
from src.common import set_context
from src.model_utils.config import (
    parse_yaml, parse_cli_to_yaml, merge, Config, compute_features_info
)
from src.dataset import (
    imnormalize_column, rescale_column_test, resize_column_test, create_meta,
    pad_img
)


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description='default name',
                                     add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(current_dir, 'configs',
                                             'config.yaml'),
                        help='Config file path')
    parser.add_argument('--checkpoint_path', help='Path to save checkpoint.')
    parser.add_argument('--pred_output', help='Path to model predictions.')
    parser.add_argument('--pred_input',
                        help='Path to image or folder with images.')
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper,
                             choices=choices, cfg_path=path_args.config_path)
    final_config = Config(merge(args, default))
    final_config = compute_features_info(final_config)
    pprint(final_config)
    print('Please check the above information for the configurations',
          flush=True)
    return final_config


def imread(path):
    """Read image."""
    img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def data_loader(path: Path, config):
    """Load image or images from folder in generator."""
    def process_img(img):
        data = {
            'image': img, 'img_shape': img.shape[:2],
            'bboxes': np.zeros((0, 4)), 'labels': np.zeros((0, 1)),
            'iscrowd': np.zeros((0, 1)), 'img_id': np.array(0)
        }
        if config.keep_ratio:
            data = rescale_column_test(data, config=config)
        else:
            data = resize_column_test(data, config=config)
        data = pad_img(data, config=config)
        data = create_meta(data)
        data = imnormalize_column(data, config=config)
        img = data['tensor']
        img = img.transpose(2, 0, 1).copy()
        img = img.astype(np.float32)
        meta = data['img_meta']
        return img, meta

    extensions = ('.png', '.jpg', '.jpeg')
    path = path.resolve()
    if path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                continue
            if item.suffix.lower() not in extensions:
                continue
            image = imread(str(item))
            image = image[..., ::-1].copy()
            tensor, img_meta = process_img(image)

            yield (
                str(item), ms.Tensor(img_meta[np.newaxis, :]),
                ms.Tensor(tensor[np.newaxis, :])
            )
    else:
        image = imread(str(path))
        image = image[..., ::-1].copy()
        tensor, img_meta = process_img(image)
        yield (
            str(path), ms.Tensor(img_meta[np.newaxis, :]),
            ms.Tensor(tensor[np.newaxis, :])
        )


def main():
    config = get_config()
    set_context(config)

    net = MaskRCNNInfer(config)

    param_dict = ms.load_checkpoint(config.checkpoint_path)
    param_dict_new = {}
    for key, value in param_dict.items():
        if not key.startswith('net.'):
            param_dict_new['net.' + key] = value
        else:
            param_dict_new[key] = value
    ms.load_param_into_net(net, param_dict_new)

    data_generator = data_loader(Path(config.pred_input), config)

    predictions = {}
    for name, meta, tensor in tqdm(data_generator):
        outputs = net(tensor, meta)
        bboxes, labels, segms, valids = outputs

        bboxes_squee = np.squeeze(bboxes.asnumpy()[0, ...])
        labels_squee = np.squeeze(labels.asnumpy()[0, ...])
        segms_squee = np.squeeze(segms.asnumpy()[0, ...])
        valids_squee = np.squeeze(valids.asnumpy()[0, ...])

        meta = meta.asnumpy()[0]

        bboxes_mask = bboxes_squee[valids_squee, :4]
        bboxes_mask[::, [2, 3]] -= bboxes_mask[::, [0, 1]] - 1
        bboxes_mask[::, [0, 2]] = np.clip(
            bboxes_mask[::, [0, 2]], a_min=0, a_max=meta[1]
        )
        bboxes_mask[::, [1, 3]] = np.clip(
            bboxes_mask[::, [1, 3]], a_min=0, a_max=meta[0]
        )
        segms_mask = segms_squee[valids_squee]
        segms_mask = net.net.mask_head.get_masks(
            segms_mask, bboxes_mask, meta[:2]
        )
        confs_mask = bboxes_squee[valids_squee, 4]
        labels_mask = labels_squee[valids_squee]

        predictions[name] = {
            'height': int(meta[0]),
            'width': int(meta[1]),
            'predictions': []
        }
        for box, conf, label, segm in zip(bboxes_mask, confs_mask, labels_mask, segms_mask):
            segm['counts'] = segm['counts'].decode()
            predictions[name]['predictions'].append({
                'bbox': {
                    'x_min': float(box[0]),
                    'y_min': float(box[1]),
                    'width': float(box[2]),
                    'height': float(box[3])
                },
                'class': {
                    'label': int(label),
                    'category_id': 'unknown',
                    'name': config.coco_classes[int(label) + 1]
                },
                'mask': segm,
                'score': float(conf)
            })

    with open(config.pred_output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=1)


if __name__ == '__main__':
    main()
