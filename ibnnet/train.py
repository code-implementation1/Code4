# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
python train.py
"""
import argparse
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init, get_rank

from src.loss import SoftmaxCrossEntropyExpand
from src.resnet_ibn import resnet50_ibn_a
from src.dataset import create_dataset_ImageNet as create_dataset, create_evalset
from src.lr_generator import lr_generator
from src.config import cfg

parser = argparse.ArgumentParser(description='Mindspore ImageNet Training')

parser.add_argument('--use_modelarts', action="store_true",
                    help="using this argument for modelarts")
# Datasets
parser.add_argument('--train_url', default='.', type=str)
parser.add_argument('--data_url', required=True, type=str, help='data path')
parser.add_argument('--ckpt_url', required=True, type=str, help="ckpt path")
parser.add_argument('--eval_url', required=True, type=str, help="eval path")
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                    help='Decrease learning rate at these epochs.')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('--pretrained', action="store_true",
                    help='use pre-trained model')
# Device options
parser.add_argument('--device_target', type=str,
                    default='Ascend', choices=['GPU', 'Ascend'])
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()


class EvalCallBack(Callback):
    """
    Precision verification using callback function.
    """
    # define the operator required
    def __init__(self, models, eval_ds, epochs_per_eval, file_name):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_ds
        self.epochs_per_eval = epochs_per_eval
        self.file_name = file_name

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch > 90:
            acc = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["acc"].append(acc["Accuracy"])
            print(acc)


if __name__ == "__main__":
    train_epoch = args.epochs
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target, save_graphs=False)
    device_id = args.device_id
    if args.device_num > 1:
        if target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
        elif target == 'GPU':
            init()
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
    else:
        context.set_context(device_id=device_id)


    train_dataset = create_dataset(dataset_path=args.data_url, do_train=True, repeat_num=1,
                                   batch_size=cfg.train_batch, target=target)
    eval_dataset = create_evalset(dataset_path=args.eval_url, do_train=False, repeat_num=1,
                                  batch_size=cfg.test_batch, target=target)

    net = resnet50_ibn_a(num_classes=cfg.class_num)
    if args.pretrained:
        param_dict = load_checkpoint(args.ckpt_url)
        load_param_into_net(net, param_dict)
    criterion = SoftmaxCrossEntropyExpand(sparse=True)
    step = train_dataset.get_dataset_size()
    lr = lr_generator(cfg.lr, train_epoch, steps_per_epoch=step)
    optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr,
                       momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    model = Model(net, loss_fn=criterion, optimizer=optimizer,
                  metrics={"Accuracy": Accuracy()})

    config_ck = CheckpointConfig(
        save_checkpoint_steps=step, keep_checkpoint_max=cfg.keep_checkpoint_max)

    ckpoint_cb = ModelCheckpoint(prefix="IBNNET_" + str(device_id), config=config_ck,
                                 directory='./ckpt')
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, eval_dataset, epoch_per_eval, "ibn")
    cb = [ckpoint_cb, time_cb, loss_cb, eval_cb]
    if args.device_num == 1:
        model.train(train_epoch, train_dataset, callbacks=cb, dataset_sink_mode=True)
    elif args.device_num > 1 and get_rank() % 8 == 0:
        model.train(train_epoch, train_dataset, callbacks=cb, dataset_sink_mode=True)
    else:
        model.train(train_epoch, train_dataset, dataset_sink_mode=True)
