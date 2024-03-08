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
"""lr generator for Mask-RCNN"""
import math


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps, min_lr=0.):
    base = float(current_step - warmup_steps) / float(decay_steps)
    base_lr -= min_lr
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr + min_lr
    return learning_rate


def dynamic_lr(config, steps_per_epoch):
    """dynamic learning rate generator"""
    base_lr = config.lr
    min_lr = config.min_lr
    total_steps = steps_per_epoch * (config.epoch_size + 1)
    warmup_steps = int(config.warmup_step)
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(
                linear_warmup_learning_rate(
                    i, warmup_steps, base_lr, base_lr * config.warmup_ratio
                )
            )
        else:
            lr.append(
                a_cosine_learning_rate(
                    i, base_lr, warmup_steps, total_steps, min_lr=min_lr
                )
            )

    return lr


def multistep_lr(config, dataset_size):
    base_lr = float(config.lr)
    learning_rate = base_lr
    lr_steps_index = 0
    lr = []
    step = 0
    warmup_steps = int(config.warmup_step)
    for epoch in range(config.epoch_size):
        if lr_steps_index < len(config.lr_steps):
            if epoch == config.lr_steps[lr_steps_index]:
                learning_rate = learning_rate * 0.1
                lr_steps_index += 1

        for _ in range(dataset_size):
            if step < warmup_steps:
                lr.append(
                    linear_warmup_learning_rate(
                        step, warmup_steps, base_lr,
                        base_lr * config.warmup_ratio
                    )
                )
            else:
                lr.append(learning_rate)
            step += 1

    return lr
