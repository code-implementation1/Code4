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
from .max_iou_assigner import MaxIOUAssigner
from .random_sampler import RandomSampler
from .assigner_sampler import AssignerSampler

from ..bbox_coder import DeltaXYWHBBoxCoder


def build_max_iou_random(
        num_bboxes: int,
        num_gts: int,
        assigner_cfg: dict,
        sampler_cfg: dict,
        bbox_coder: DeltaXYWHBBoxCoder,
        rcnn_mode: bool = False
) -> AssignerSampler:
    """Build MaxIOUAssigner and RandomSampler."""
    assigner = MaxIOUAssigner(
        num_bboxes=num_bboxes, num_gts=num_gts, **assigner_cfg
    )
    sampler = RandomSampler(
        num_bboxes=num_bboxes, num_gts=num_gts, **sampler_cfg
    )
    assigner_sampler_obj = AssignerSampler(
        assigner=assigner, sampler=sampler, rcnn_mode=rcnn_mode,
        bbox_coder=bbox_coder
    )
    return assigner_sampler_obj


__all__ = [
    'MaxIOUAssigner', 'RandomSampler', 'build_max_iou_random'
]
