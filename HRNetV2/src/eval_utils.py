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
import os

import cv2
import numpy as np

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms
from mindspore import nn
from mindspore.nn.metrics import Metric, Loss

class EvalCell(nn.Cell):
    """Wrapper for model evaluation in model.fit"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(
            self, img_data, img_metas, gt_bboxes, gt_seg_masks, gt_labels,
            gt_valids
    ):
        (
            pred_bboxes, pred_labels, pred_mask, pred_valids, _
        ) = self.net(
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


def prepare_masks(
        mask_pred, det_bboxes, ori_shape,
        mask_thr_binary=0.5
):
    """Get segmentation masks from mask_pred and bboxes.
    Args:
        mask_pred (ndarray): shape (n, h, w).
            For single-scale testing, mask_pred is the direct output of
            model, whose type is Tensor, while for multiscale testing,
            it will be converted to numpy array outside of this method.
        det_bboxes (ndarray): shape (n, 4/5)
        ori_shape: original image size
        mask_thr_binary (float): threshold that defines mask activations.
    Returns:
        list[list]: encoded masks
    """
    assert isinstance(mask_pred, np.ndarray)
    # when enabling mixed precision training, mask_pred may be float16
    # numpy array
    mask_pred = mask_pred.astype(np.float32)
    segms = []
    bboxes = det_bboxes[:, :4]

    img_h, img_w = ori_shape.astype(np.int32)
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :].astype(np.int32)
        w = max(bbox[2], 1)
        h = max(bbox[3], 1)
        w = min(w, img_w - bbox[0])
        h = min(h, img_h - bbox[1])
        if w <= 0 or h <= 0:
            print(
                f'there is invalid proposal bbox, index={i} bbox={bbox} w={w} '
                f'h={h}'
            )
            w = max(w, 1)
            h = max(h, 1)

        mask_pred_ = mask_pred[i, :, :]

        bbox_mask = cv2.resize(
            mask_pred_, (w, h), interpolation=cv2.INTER_LINEAR
        )
        bbox_mask = (bbox_mask > mask_thr_binary).astype(np.uint8)

        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F')
        )[0]
        segms.append(rle)

    return segms


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
            for field in ['image_id', 'category_id', 'score', 'segmentation']:
                assert all([field in p for p in preds_post]), f'{field} is not found'
            self.preds.extend(preds_post)
        else:
            for field in ['image_id', 'bbox', 'score', 'category_id']:
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
    """Wrapper to bbox postprocessing for RetinaNet models."""

    def __init__(
            self, bbox_normalization=True, segmentation=False,
            mask_postprocessor=lambda *x, **y: x
    ):
        """Init PostProcessorRetinaNet."""
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

def get_metrics(config, model):
    """Create evaluation metrics."""
    bbox_post_processor = PostProcessorMaskRCNN(
        bbox_normalization=True, segmentation=False
    )
    segm_post_processor = PostProcessorMaskRCNN(
        bbox_normalization=True,
        mask_postprocessor=model.mask_head.get_seg_masks,
        segmentation=True
    )

    eval_metrics = {
        'loss': Loss(),
        'bbox_mAP': COCOMeanAveragePrecision(
            os.path.join(config.val_dataset, 'labels.json'),
            post_processing=bbox_post_processor,
            mask_rcnn=True, segmentation=False
        ),
        'seg_mAP': COCOMeanAveragePrecision(
            os.path.join(config.val_dataset, 'labels.json'),
            post_processing=segm_post_processor,
            mask_rcnn=True, segmentation=True
        )
    }

    return eval_metrics
