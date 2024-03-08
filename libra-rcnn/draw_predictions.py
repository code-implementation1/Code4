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
"""Draw inference results on single image."""
import argparse
import json
import os

import cv2

from src.detecteval import draw_one_box


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', help='Path to image.')
    parser.add_argument('--preds', help='Path to inference result.')
    parser.add_argument('--output', help='Path to output image')

    return parser.parse_args()


def main():
    args = parse_args()

    img_path = os.path.abspath(args.image)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    with open(args.preds, 'r') as f:
        preds = json.load(f)

    img_preds = preds[img_path]

    for pred in sorted(img_preds['predictions'], key=lambda x: x['score']):
        bbox = pred['bbox']
        x_min, y_min, x_max, y_max = (
            bbox['x_min'], bbox['y_min'],
            bbox['x_min'] + bbox['width'],
            bbox['y_min'] + bbox['height']
        )
        label = pred['class']['label']
        cls_name = pred['class']['name']
        score = pred['score']

        text = f'{cls_name}: {round(score, 3)}'
        draw_one_box(img, text, (x_min, y_min, x_max, y_max), label)

    cv2.imwrite(args.output, img)


if __name__ == '__main__':
    main()
