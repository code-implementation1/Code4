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
"""Convert dataset from coco format to mindrecord."""
import argparse
import os
from pprint import pprint

from src.dataset import data_to_mindrecord_byte_image
from src.model_utils.config import parse_yaml, parse_cli_to_yaml, merge, Config


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
    parser.add_argument('--converted_coco_path',
                        help='Path to original coco subset.')
    parser.add_argument('--converted_mindrecord_path',
                        help='Path to result mindrecord folder')
    parser.add_argument('--converted_mindrecord_files_num', type=int,
                        default=1, help='Number of output mindrecord files.')
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper,
                             choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    pprint(final_config)
    print('Please check the above information for the configurations',
          flush=True)
    return Config(final_config)


def main():
    """Entry point."""
    config = get_config()
    data_to_mindrecord_byte_image(
        config, img_path=config.converted_coco_path,
        mindrecord_path=config.converted_mindrecord_path,
        file_num=config.converted_mindrecord_files_num
    )


if __name__ == '__main__':
    main()
