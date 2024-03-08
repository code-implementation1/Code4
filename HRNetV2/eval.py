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
"""Evaluate the model, measure object detection accuracy metrics."""
import os

import mindspore as ms
from tqdm import tqdm

from src.config import get_config
from src.models.hrnetv2.hrnetv2 import Model
from src.data.dataset import create_custom_coco_det_dataset
from src.eval_utils import get_metrics


def run_eval(config, net, ckpt_path, ds):
    param_dict = ms.load_checkpoint(ckpt_path)
    param_dict_new = {}
    for key, value in param_dict.items():
        if not key.startswith('net.'):
            param_dict_new['net.' + key] = value
        else:
            param_dict_new[key] = value
    ms.load_param_into_net(net, param_dict_new)

    metrics = get_metrics(config, net)
    metric_bbox = metrics['bbox_mAP']
    metric_segm = metrics['seg_mAP']

    if (
            config.prediction_path is not None
            and os.path.exists(config.prediction_path + '.bbox.json')
            and os.path.exists(config.prediction_path + '.segm.json')
    ):
        print(f'Load predictions from file: {config.prediction_path}')
        metric_bbox.load_preds(config.prediction_path + '.bbox.json')
        metric_segm.load_preds(config.prediction_path + '.segm.json')
    else:
        for _, data in enumerate(tqdm(
                ds.create_dict_iterator(num_epochs=1),
                total=ds.get_dataset_size()
        )):
            img_data = data['image']
            img_metas = data['image_shape']
            gt_bboxes = data['box']
            gt_labels = data['label']
            gt_segm = data['segm']
            gt_num = data['valid_num']

            # run net
            output = net(img_data, img_metas)
            all_bbox = output[0]
            all_label = output[1]
            all_mask = output[2]
            all_valid = output[3]

            metric_bbox.update(
                (all_bbox, all_label, all_mask, all_valid),
                (gt_bboxes, gt_labels, gt_segm, gt_num, img_metas)
            )
            metric_segm.update(
                (all_bbox, all_label, all_mask, all_valid),
                (gt_bboxes, gt_labels, gt_segm, gt_num, img_metas)
            )

        if config.prediction_path is not None:
            metric_bbox.dump_preds(config.prediction_path + '.bbox.json')
            metric_segm.dump_preds(config.prediction_path + '.segm.json')

    print(f'Eval result (bboxes): {metric_bbox.eval()}')
    print(f'Eval result (segmentation): {metric_segm.eval()}')


def main():
    config = get_config()
    ms.set_context(mode=ms.GRAPH_MODE,
                   device_target=config.device_target)

    val_dataset = create_custom_coco_det_dataset(
        path=config.val_dataset, config=config, training=False,
        python_multiprocessing=config.python_multiprocessing > 0
    )

    net = Model(config)

    run_eval(config, net, config.checkpoint_path, val_dataset)


if __name__ == '__main__':
    main()
