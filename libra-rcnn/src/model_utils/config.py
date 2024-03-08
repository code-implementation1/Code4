# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
"""Parse arguments"""

import os
import ast
import argparse
from pprint import pformat
import yaml

config = None


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="grid_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError(
                    'At most 3 docs (config, description for help, choices) are supported in config yaml'
                )
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def compute_features_info(cfg):
    """Compute features information."""
    cfg.img_height = (
        (cfg.img_height + cfg.divider - 1) // cfg.divider * cfg.divider
    )
    cfg.img_width = (
        (cfg.img_width + cfg.divider - 1) // cfg.divider * cfg.divider
    )

    if not hasattr(cfg, "feature_shapes"):
        cfg.feature_shapes = [
            [cfg.img_height // 4, cfg.img_width // 4],
            [cfg.img_height // 8, cfg.img_width // 8],
            [cfg.img_height // 16, cfg.img_width // 16],
            [cfg.img_height // 32, cfg.img_width // 32],
            [cfg.img_height // 64, cfg.img_width // 64],
        ]
    cfg.anchor_generator.num_anchors = (
        len(cfg.anchor_generator.ratios) * len(cfg.anchor_generator.scales)
    )
    cfg.num_bboxes = cfg.anchor_generator.num_anchors * sum(
        [lst[0] * lst[1] for lst in cfg.feature_shapes])

    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    global config
    if config is not None:
        return config

    parser = argparse.ArgumentParser(description='default name', add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        '--config_path', type=str,
        default=os.path.join(
            current_dir, '..', '..', 'configs', 'resnet50_config.yaml'
        ),
        help='Config file path'
    )
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(
        parser=parser, cfg=default, helper=helper, choices=choices,
        cfg_path=path_args.config_path
    )
    default = Config(merge(args, default))
    default = compute_features_info(default)
    config = default
    return default
