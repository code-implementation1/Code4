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
"""Coco metrics utils"""
import json

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms
from mindspore import nn
from mindspore.nn.metrics import Metric


class EvalCell(nn.Cell):
    """Wrapper for model evaluation in model.fit"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(
            self, img_data, img_metas, gt_bboxes, gt_seg_masks, gt_labels,
            gt_valids
    ):
        pred_bboxes, pred_labels, pred_mask, pred_valids, _, _, _ = self.net(
            img_data, img_metas, None, None, None, None
        )

        return (
            ms.Tensor(np.array(-1.)),
            (pred_bboxes, pred_labels, pred_mask, pred_valids),
            (gt_bboxes, gt_labels, gt_seg_masks, gt_valids, img_metas)
        )


def norm_bboxes(boxes, meta):
    """Normalize bboxes according original images size."""
    boxes = boxes.copy()
    boxes[..., [0, 2]] = np.clip(
        boxes[..., [0, 2]], a_min=0, a_max=meta[3]
    )
    boxes[..., [1, 3]] = np.clip(
        boxes[..., [1, 3]], a_min=0, a_max=meta[2]
    )
    boxes[..., [0, 2]] /= meta[3]
    boxes[..., [1, 3]] /= meta[2]
    boxes[..., [0, 2]] *= meta[1]
    boxes[..., [1, 3]] *= meta[0]
    return boxes


def convert_to_numpy(obj):
    """Convert model outputs to NumPy format."""
    if isinstance(obj, ms.Tensor):
        return obj.asnumpy()
    if isinstance(obj, (list, tuple)):
        return [convert_to_numpy(a) for a in obj]
    if isinstance(obj, np.ndarray):
        return obj
    raise ValueError(f'Unsupported object type {type(obj)}')


class COCOMeanAveragePrecision(Metric):
    """Compute mean average precision using pycocotools."""

    def __init__(
            self, annotations,
            mask_rcnn=False, post_processing=lambda *x, **y: {},
            segmentation=False
    ):
        super().__init__()
        self.coco = COCO(annotations)
        self.cat_ids_dict = self.coco.getCatIds()
        self.post_processing = post_processing
        self.mask_rcnn = mask_rcnn
        self.segm = segmentation
        self.clear()

    def clear(self):
        self.preds = []
        self.coco_eval = None

    def update(self, preds, gt):
        *_, metas = convert_to_numpy(gt)
        preds = convert_to_numpy(preds)

        self.prepare_preds_single(preds, metas)

    def prepare_preds_single(self, preds, metas):
        preds_post = self.post_processing(
            *preds, metas=metas, cat_ids_dict=self.cat_ids_dict
        )
        if self.segm:
            for field in ['image_id', 'category_id', 'segmentation']:
                assert all([field in p for p in preds_post]), f'{field} is not found'
            self.preds.extend(preds_post)
        else:
            for field in ['image_id', 'bbox', 'category_id']:
                assert all([field in p for p in preds_post]), f'{field} is not found'
            self.preds.extend(preds_post)

    def eval(self):
        """Evaluate."""
        if not self.preds:
            return 0.0
        coco_dets = self.coco.loadRes(self.preds)
        det_img_ids = self.coco.getImgIds()

        iou_type = 'segm' if self.mask_rcnn and self.segm else 'bbox'
        self.coco_eval = COCOeval(self.coco, coco_dets, iou_type)

        tgt_ids = det_img_ids

        self.coco_eval.params.imgIds = tgt_ids
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()
        return self.coco_eval.stats[0]

    def dump_preds(self, path):
        """Save predictions to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.preds, f, indent=1)

    def load_preds(self, path):
        """Load predictions from file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.preds = json.load(f)


class PostProcessorMaskRCNN:
    """Wrapper to bbox postprocessing for MaskRCNN models."""

    def __init__(
            self, bbox_normalization=True, segmentation=False,
            mask_postprocessor=lambda *x, **y: x
    ):
        """Init PostProcessorMaskRCNN."""
        self.bbox_normalization = bbox_normalization
        self.mask_postprocessor = mask_postprocessor
        self.segmentation = segmentation

    def __call__(
            self, pred_bboxes, pred_labels, pred_masks, pred_valid,
            metas, **kwargs
    ):
        """Run post-processing."""
        size = len(metas)
        preds = []
        for i in range(size):
            valid = pred_valid[i]
            bboxes = pred_bboxes[i]
            labels = pred_labels[i]
            masks = pred_masks[i]
            meta = metas[i]

            valid = valid.reshape(-1)

            x1y1x2y2 = bboxes[::, :4]
            x1y1x2y2 = x1y1x2y2[valid]
            if self.bbox_normalization:
                x1y1x2y2 = norm_bboxes(x1y1x2y2, meta)

            xywh = x1y1x2y2.copy()
            xywh[::, [2, 3]] = xywh[::, [2, 3]] - xywh[::, [0, 1]]

            labels = labels.reshape(-1)
            labels = labels[valid]
            labels = [
                np.array([kwargs['cat_ids_dict'][label]]) for label in labels
            ]

            conf = bboxes[::, [4]]
            conf = conf[valid]

            img_id = meta[-1]
            bboxes_size = len(x1y1x2y2)
            if self.segmentation:
                masks = masks[valid]
                segms = self.mask_postprocessor(
                    mask_pred=masks,
                    det_bboxes=x1y1x2y2,
                    ori_shape=meta[:2]
                )
                for j in range(bboxes_size):
                    segms[j]['counts'] = segms[j]['counts'].decode()
                    pred = {
                        'image_id': int(img_id),
                        'category_id': int(labels[j]),
                        'score': float(conf[j]),
                        'segmentation': segms[j]
                    }
                    preds.append(pred)
            else:
                for j in range(bboxes_size):
                    pred = {
                        'image_id': int(img_id),
                        'bbox': xywh[j].tolist(),
                        'score': float(conf[j]),
                        'category_id': int(labels[j])
                    }
                    preds.append(pred)
        return preds
