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
"""Train blocks and get checkpoint files."""
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from pprint import pprint
import logging
from datetime import datetime
from functools import reduce
import mindspore as ms
from mindspore import Tensor
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.nn import SGD, Adam, metrics
from mindspore.communication.management import init, get_rank, get_group_size

from src.blocks.detectors.mask_rcnn import MaskRCNN

from src.network_define import (
    WithLossCell, TrainOneStepCell, LossNet, TrainOneStepCellCPU
)
from src.common import config_logging, set_context
from src.dataset import prepare_data
from src.lr_schedule import dynamic_lr, multistep_lr
from src.model_utils.config import get_config, Config
from src.model_utils.device_adapter import get_device_id
from src.common import get_callbacks
from src.mlflow_funcs import (
    mlflow_log_state, mlflow_log_args
)
from src.eval_utils import (
    EvalCell, COCOMeanAveragePrecision, PostProcessorMaskRCNN
)


def modelarts_pre_process():
    config = get_config()
    config.save_checkpoint_path = config.output_path


def load_ckpt_to_network(net, checkpoint_path, finetune=False):
    """Load pre-trained checkpoint."""
    new_param = {}
    if finetune:
        param_not_load = [
            'learning_rate',
            'stat.bbox_head.fc_cls.weight',
            'stat.bbox_head.fc_cls.bias',
            'stat.bbox_head.fc_reg.weight',
            'stat.bbox_head.fc_reg.bias',
            'bbox_head.fc_cls.weight',
            'bbox_head.fc_cls.bias',
            'bbox_head.fc_reg.weight',
            'bbox_head.fc_reg.bias',
            'accum.bbox_head.fc_cls.weight',
            'accum.bbox_head.fc_cls.bias',
            'accum.bbox_head.fc_reg.weight',
            'accum.bbox_head.fc_reg.bias',

            'stat.mask_head.conv_logits.bias',
            'stat.mask_head.conv_logits.weight',
            'mask_head.conv_logits.weight',
            'mask_head.conv_logits.bias',
            'accum.mask_head.conv_logits.weight',
            'accum.mask_head.conv_logits.bias'
        ]
    else:
        param_not_load = ['learning_rate']

    logging.info('Loading from checkpoint: %s', checkpoint_path)
    param_dict = ms.load_checkpoint(checkpoint_path)
    for key, value in param_dict.items():
        if key in param_not_load:
            continue
        if not key.startswith('net.'):
            new_param['net.' + key] = value
        else:
            new_param[key] = value

    ms.load_param_into_net(net, new_param)

    logging.info('\tDone!\n')
    return net


def flatten_dict(cfg_dict, parent_key='', sep='.'):
    """process values before parameters saving"""
    res_list = []
    for k, v in cfg_dict.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, Config):
            res_list.extend(
                flatten_dict(v.__dict__, parent_key=new_key, sep=sep).items()
            )
        else:
            res_list.append(
                (
                    new_key,
                    str(v) if len(str(v)) < 500 else str(v)[:477] + '...'
                )
            )

    return dict(res_list)


def main():
    """Training entry point."""
    config = get_config()
    mlflow_log_state()
    config_dict = config.__dict__
    config_dict = flatten_dict(config_dict)
    mlflow_log_args(config_dict)
    experiment_name = '_'.join(
        [datetime.now().strftime('%y%m%d'), 'MaskRCNN']
    )
    if config.brief is not None:
        experiment_name = f'{experiment_name}_{config.brief}'
    ckpt_save_dir = Path(os.path.join(config.train_outputs, experiment_name))

    if config.rank_id == 0:
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        dump_env_and_params(ckpt_save_dir, config)

    config_logging(filename_prefix=str(ckpt_save_dir / 'logs' / 'train'))

    train_dataset, val_dataset = prepare_data(config)

    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size = (
        None if val_dataset is None else val_dataset.get_dataset_size()
    )
    logging.info('Train dataset size: %d', train_dataset_size)

    logging.info('Creating network...')
    if config.detector == 'mask_rcnn':
        net = MaskRCNN(config)
    else:
        raise NotImplementedError(f'{config.detector} is not implemented.')
    logging.info(
        'Number of parameters: %s',
        str(sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in
            (net.trainable_params() + net.untrainable_params())
        )))
    criterion = LossNet()
    net_with_loss = WithLossCell(net, criterion)

    logging.info('Device type: %s', config.device_target)
    logging.info('Creating criterion, lr and opt objects...')

    opt = get_optimizer(config, net, train_dataset_size)

    logging.info('\tDone!\n')
    if config.device_target == 'CPU':
        net_with_loss = TrainOneStepCellCPU(
            net_with_loss, opt, sens=config.loss_scale)
    else:
        net_with_loss = TrainOneStepCell(
            net_with_loss, opt, scale_sense=config.loss_scale,
            grad_clip=config.grad_clip)

    # load pretrained checkpoint
    if config.pre_trained:
        net_with_loss = load_ckpt_to_network(
            net_with_loss, config.pre_trained, config.finetune
        )
    bbox_post_processor = PostProcessorMaskRCNN(
        bbox_normalization=True, segmentation=False
    )
    segm_post_processor = PostProcessorMaskRCNN(
        bbox_normalization=True,
        mask_postprocessor=net.mask_head.get_seg_masks,
        segmentation=True
    )
    eval_metrics = {
        'loss': metrics.Loss(),
        'bbox_mAP': COCOMeanAveragePrecision(
            annotations=os.path.join(config.val_dataset, 'labels.json'),
            post_processing=bbox_post_processor, mask_rcnn=True, segmentation=False
        ),
        'seg_mAP': COCOMeanAveragePrecision(
            annotations=os.path.join(config.val_dataset, 'labels.json'),
            post_processing=segm_post_processor, mask_rcnn=True, segmentation=True
        )}

    model = Model(
        net_with_loss, metrics=eval_metrics, eval_network=EvalCell(net),
        eval_indexes=[0, 1, 2])

    cb = get_callbacks(
        arch='MaskRCNN', logs_dir=ckpt_save_dir / 'logs',
        rank=config.rank_id, ckpt_dir=ckpt_save_dir,
        train_data_size=train_dataset_size, val_data_size=val_dataset_size,
        best_ckpt_dir=ckpt_save_dir / 'best_ckpt',
        summary_dir=ckpt_save_dir / 'summary',
        ckpt_save_every_step=config.save_every,
        print_loss_every=config.save_every,
        ckpt_keep_num=config.keep_checkpoint_max,
        best_ckpt_num=config.keep_best_checkpoints_max)
    model.fit(
        config.epoch_size, train_dataset,
        valid_dataset=val_dataset, callbacks=cb,
        dataset_sink_mode=bool(config.datasink),
        valid_frequency=config.eval_every)


def dump_env_and_params(ckpt_save_dir, args):
    """Dump information about environment ang hyperparameters."""
    shutil.copy(str(args.config_path), str(ckpt_save_dir))
    with open(str(ckpt_save_dir / 'cmd.txt'), 'w', encoding='utf-8'
              ) as file:
        file.write(' '.join(sys.argv))
    with open(str(ckpt_save_dir / 'args.txt'), 'w', encoding='utf-8'
              ) as file:
        file.write(str(args))
    try:
        with open(str(ckpt_save_dir / 'git.txt'), 'w', encoding='utf-8'
                  ) as file:
            commit_info = subprocess.check_output(
                ['git', 'show', '-s'],
                cwd=Path(__file__).absolute().parents[0],
            )
            decoded_commit_info = commit_info.decode('utf-8')
            decoded_commit_info = decoded_commit_info.replace('\n', ', ')
            file.write(decoded_commit_info)
    except subprocess.CalledProcessError as git_exception:
        logging.error('Git dumping error: %s', str(git_exception))
        logging.error(traceback.format_exc())


def get_optimizer(cfg, net, train_dataset_size):
    """Define optimizer according config."""
    if cfg.lr_type.lower() not in ('dynamic', 'multistep'):
        raise ValueError('Optimize type should be "dynamic" or "dynamic"')
    if cfg.lr_type.lower() == 'dynamic':
        lr = Tensor(dynamic_lr(cfg, train_dataset_size), ms.float32)
    else:
        lr = Tensor(multistep_lr(cfg, train_dataset_size), ms.float32)

    if cfg.opt_type.lower() not in ('sgd', 'adam'):
        raise ValueError('Optimize type should be "sgd" or "adam"')
    if cfg.opt_type.lower() == 'sgd':
        opt = SGD(
            params=net.trainable_params(), learning_rate=lr,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    else:
        opt = Adam(
            params=net.trainable_params(), learning_rate=lr,
            weight_decay=cfg.weight_decay
        )
    return opt


if __name__ == '__main__':
    set_seed(1)
    config_ = get_config()
    set_context(config_)
    ms.set_context(device_id=get_device_id(), max_call_depth=2000)

    if config_.device_target == 'GPU':
        ms.set_context(enable_graph_kernel=bool(config_.enable_graph_kernel))
    if config_.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        ms.set_auto_parallel_context(
            device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True
        )
    else:
        rank = 0
        device_num = 1

    config_.rank_id = rank
    config_.device_num = device_num

    print()  # just for readability
    pprint(config_)
    print(
        f'Please check the above information for the '
        f'configurations\n\n'
    )

    main()
