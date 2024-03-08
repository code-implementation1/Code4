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
"""Evaluation for LibraRcnn"""
import argparse
import os
import time
from pprint import pprint

import mindspore as ms
from mindspore.common import set_seed
from tqdm import tqdm

from src.model_utils.config import (
    parse_yaml, parse_cli_to_yaml, merge, Config, compute_features_info
)
from src.LibraRcnn.libra_rcnn import LibraRcnnInfer
from src.dataset import create_mindrecord_dataset
from src.eval_utils import COCOMeanAveragePrecision
ms.context.set_context(max_call_depth=2000)


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description='default name',
                                     add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(current_dir, 'configs',
                                             'resnet50_config.yaml'),
                        help='Config file path')
    parser.add_argument('--checkpoint_path', help='Path to save checkpoint.')
    parser.add_argument('--prediction_path', help='Path to model predictions.')
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


def rcnn_eval(ds, ckpt_path, anno_path):
    """RCNN model evaluation."""
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f'CheckPoint file {ckpt_path} is not valid.')
    net = LibraRcnnInfer(config)
    net.set_train(False)
    param_dict = ms.load_checkpoint(ckpt_path)
    param_dict_new = {}
    for key, value in param_dict.items():
        if not key.startswith('net.'):
            param_dict_new['net.' + key] = value
        else:
            param_dict_new[key] = value
    ms.load_param_into_net(net, param_dict_new)

    eval_iter = 0
    total = ds.get_dataset_size()

    metric = COCOMeanAveragePrecision(
        os.path.join(config.val_dataset, anno_path), bbox_normalization=False
    )

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    if config.prediction_path is not None and os.path.exists(config.prediction_path):
        print(f'Load predictions from file: {config.prediction_path}')
        metric.load_preds(config.prediction_path)
    else:
        for data in tqdm(
                ds.create_dict_iterator(num_epochs=1),
                total=ds.get_dataset_size()
        ):
            eval_iter = eval_iter + 1
            img_data = data['image']
            img_metas = data['image_shape']
            gt_bboxes = data['box']
            gt_labels = data['label']
            gt_num = data['valid_num']
            # run net
            output = net(
                img_data, img_metas,
            )

            # output
            all_bbox = output[0]
            all_label = output[1]
            all_mask = output[2]
            metric.update(
                (all_bbox, all_label, all_mask),
                (gt_bboxes, gt_labels, gt_num, img_metas)
            )

        if config.prediction_path is not None:
            metric.dump_preds(config.prediction_path)

    print(f'Eval result: {metric.eval()}')
    print("\nEvaluation done!")


def modelarts_pre_process():
    pass


def main():
    """Run Libra RCNN evaluation."""
    val_dataset = create_mindrecord_dataset(
        path=config.val_dataset, config=config, training=False,
        python_multiprocessing=config.python_multiprocessing
    )
    print("CHECKING MINDRECORD FILES DONE!")
    print("Start Eval!")
    start_time = time.time()
    rcnn_eval(
        val_dataset, config.checkpoint_path,
        os.path.join(config.val_dataset, 'labels.json')
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f'\nDone!\nTime taken: {int(total_time)} seconds')


if __name__ == '__main__':
    set_seed(1)
    config = get_config()
    ms.set_context(
        mode=ms.GRAPH_MODE, device_target=config.device_target
    )
    config.rank = 0
    main()
