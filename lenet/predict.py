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
"""
######################## predict lenet example ########################

Note:
    To run this scripts, 'mindspore' and 'mindspore_lite' must be installed.

Usage:
    python predict.py --config_path [Your config path]
"""
import time
import os

import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from src.model_utils.config import config
from src.lenet import LeNet5
from src.dataset import create_dataset


def create_model():
    """
    create model.
    """
    if config.dataset_name == "mnist":
        num_channel = 1
    elif config.dataset_name == "imagenet2012":
        num_channel = 3
    else:
        raise RuntimeError(f'For lenet predict, input dataset only support mnist and imagenet2012, '
                           f'but got {config.dataset_name}')
    network = LeNet5(config.num_classes, num_channel=num_channel)
    if config.ckpt_file:
        param_dict = load_checkpoint(config.ckpt_file)
        load_param_into_net(network, param_dict)
    ms_model = Model(network)
    return ms_model


def read_image(img_path):
    img = Image.open(img_path).convert("RGB")
    mean = [0.475 * 225, 0.451 * 225, 0.392 * 225]
    std = [0.275 * 225, 0.267 * 225, 0.278 * 225]
    transform_list = [ds.vision.Resize((config.image_width, config.image_height)),
                      ds.vision.Normalize(mean, std),
                      ds.vision.HWC2CHW()]
    for transform in transform_list:
        img = transform(img)
    img = ms.Tensor(np.expand_dims(img, axis=0), ms.float32)
    return img


def predict_backend_lite(data_input):
    """
    model.predict using backend lite.
    """
    # model predict using backend lite
    ms_model = create_model()
    output = ms_model.predict(data_input, backend="lite")
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input, backend="lite")
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def predict_mindir(data_input):
    """
    predict by MindIR.
    """
    def _predict_core(lite_mode_input):
        """single input."""
        inputs = lite_mode_input.get_inputs()
        if len(inputs) > 1:
            raise RuntimeError("Only support single input in this net.")
        inputs[0].set_data_from_numpy(data_input.asnumpy())
        outputs = lite_mode_input.predict(inputs)
        return ms.Tensor(outputs[0].get_data_to_numpy())

    def _get_lite_context(l_context):
        lite_context_properties = {
            "cpu": ["inter_op_parallel_num", "precision_mode", "thread_num",
                    "thread_affinity_mode", "thread_affinity_core_list"],
            "gpu": ["device_id", "precision_mode"],
            "ascend": ["device_id", "precision_mode", "provider", "rank_id"]
        }
        lite_device_target = ms.get_context('device_target').lower()
        if lite_device_target not in ['cpu', 'gpu', 'ascend']:
            raise RuntimeError(f"Device target should be in ['cpu', 'gpu', 'ascend'], but got {lite_device_target}")
        l_context.target = [lite_device_target]
        l_context_device_dict = {'cpu': l_context.cpu, 'gpu': l_context.gpu, 'ascend': l_context.ascend}
        for single_property in lite_context_properties.get(lite_device_target):
            try:
                context_value = ms.get_context(single_property)
                if context_value:
                    setattr(l_context_device_dict.get(lite_device_target), single_property, context_value)
            except ValueError:
                print(f'For set lite context, fail to get parameter {single_property} from ms.context.'
                      f' Will use default value')
        return l_context

    try:
        import mindspore_lite as mslite
    except ImportError:
        raise ImportError(f"For predict by MindIR, mindspore_lite should be installed.")

    lite_context = mslite.Context()
    lite_context = _get_lite_context(lite_context)

    lite_model = mslite.Model()
    lite_model.build_from_file(config.mindir_path, mslite.ModelType.MINDIR, lite_context)

    output = _predict_core(lite_model)
    t_start = time.time()
    for _ in range(100):
        _predict_core(lite_model)
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def predict_lenet(data_input):
    """
    ms.Model.predict
    """
    print('predict with config: ', config)
    ms_model = create_model()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    # model predict
    output = ms_model.predict(data_input)
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input)
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def _get_max_index_from_res(data_input):
    data_input = data_input.asnumpy()
    data_input = data_input.flatten()
    res_index = np.where(data_input == np.max(data_input))  # (array([6]), )
    return res_index[0][0]


def get_mnist_val_input():
    """
    Get random single input from mnist validation dataset.
    """
    ds_val = create_dataset(os.path.join(config.data_path, "test"), 1, 2)
    if ds_val.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    image, _ = next(ds_val.create_tuple_iterator())
    return image


if __name__ == "__main__":
    if config.dataset_name == "imagenet2012":
        test_data_input = read_image(config.img_path)
    elif config.dataset_name == "mnist":
        test_data_input = get_mnist_val_input()
    else:
        raise RuntimeError(f'For lenet predict, input dataset only support mnist and imagenet2012, '
                           f'but got {config.dataset_name}')


    if config.enable_predict:
        res, avg_t = predict_lenet(test_data_input)
        print("Prediction res: ", _get_max_index_from_res(res))
        print(f"Prediction avg time: {avg_t * 1000} ms")

    if config.enable_predict_lite_backend:
        res_lite, avg_t_lite = predict_backend_lite(test_data_input)
        print("Predict using backend lite, res: ", _get_max_index_from_res(res_lite))
        print(f"Predict using backend lite, avg time: {avg_t_lite * 1000} ms")

    if config.enable_predict_lite_mindir:
        res_mindir, avg_t_mindir = predict_mindir(test_data_input)
        print("Predict by mindir, res: ", _get_max_index_from_res(res_mindir))
        print(f"Predict by mindir, avg time: {avg_t_mindir * 1000} ms")
