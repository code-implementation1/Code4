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
import mindspore as ms

from mindspore import nn
from mindspore.nn.metrics import Metric
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment

from src.detecteval import DetectEval


class EvalCell(nn.Cell):
    """Wrapper for model evaluation in model.fit"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids):
        pred_bboxes, pred_labels, pred_valids, _, _, _ = self.net(
            img_data, img_metas, None, None, None
        )

        return (
            ms.Tensor(np.array(-1.)),
            (pred_bboxes, pred_labels, pred_valids),
            (gt_bboxes, gt_labels, gt_valids, img_metas)
        )


class MeanAveragePrecision(Metric):
    """Custom mAP metric."""

    def __init__(
            self, num_classes, iou_thr=0.5, norm_pred=False, norm_gt=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thr = iou_thr
        self.norm_pred = norm_pred
        self.norm_gt = norm_gt
        self.clear()

    def clear(self):
        self.pred_bboxes, self.pred_cls, self.pred_mask = [], [], []
        self.gt_bboxes, self.gt_cls, self.gt_mask = [], [], []

    def update(self, preds, gt):
        pred_bboxes, pred_cls, pred_mask = [x.asnumpy() for x in preds]
        gt_bboxes, gt_cls, gt_mask, metas = [x.asnumpy() for x in gt]
        batch_size = pred_bboxes.shape[0]
        for i in range(batch_size):
            self.pred_bboxes.append(
                norm_bboxes(pred_bboxes[i], metas[i])
                if self.norm_pred else pred_bboxes[i]
            )
            self.pred_cls.append(pred_cls[i])
            self.pred_mask.append(pred_mask[i])
            self.gt_bboxes.append(
                norm_bboxes(gt_bboxes[i], metas[i])
                if self.norm_gt else gt_bboxes[i]
            )
            self.gt_cls.append(gt_cls[i])
            self.gt_mask.append(gt_mask[i])

    def _make_pairs(self, gt_list, pr_list):
        """
        Give correspondence for segments from two lists.

        Args:
            gt_list:
                List bboxes: list[bbox_1, ..., bbox_N],
                where box_* is [y_min, x_min, y_max, x_max].
            pr_list:
                List bboxes: list[bbox_1, ..., bbox_N],
                where box_* is [y_min, x_min, y_max, x_max].

        Returns:
            (gt_indexs: List[int], pr_indexs: List[int])
                Correspondence of indices of given lists.

        """
        # Cost matrix construction.
        cost_mat = []
        for i, gt_list_i in enumerate(gt_list):
            if gt_list_i is None:
                cost_mat.append([1] * len(pr_list))
            else:
                cost_mat.append([])
                for pr_list_j in pr_list:
                    cost_mat[i].append(
                        1 - self._calc_iou(gt_list_i, pr_list_j))
        # Use the Hungarian algorithm.
        row_ind, col_ind = linear_sum_assignment(np.array(cost_mat, ndmin=2))
        return row_ind, col_ind

    def _calc_iou(self, gt_bbox: np.ndarray, pr_bbox: np.ndarray) -> float:
        """
        Calculate IoU two bboxes.

        Parameters
        ----------
        gt_bbox: np.ndarray
            Ground true bbox (2, 2).
        pr_bbox: np.ndarray
            Predict bbox (2, 2).

        Return
        ------
        float
            IoU value.
        """
        gt_bbox = np.array(gt_bbox).reshape(-1, 2)
        pr_bbox = np.array(pr_bbox).reshape(-1, 2)
        sp = np.array([gt_bbox[0], pr_bbox[0]])
        ep = np.array([gt_bbox[1], pr_bbox[1]])

        wh = np.min(ep.T, axis=-1) - np.max(sp.T, axis=-1)
        if not np.all(wh > 0):
            return 0

        inter = np.prod(wh)
        gt_s = np.prod(gt_bbox[1] - gt_bbox[0])
        pr_s = np.prod(pr_bbox[1] - pr_bbox[0])
        iou = inter / (gt_s + pr_s - inter + 1e-6)
        return iou

    def eval(self):
        confidence = [np.array([]) for i in range(self.num_classes)]
        iou = [np.array([]) for i in range(self.num_classes)]
        gt_number = [0 for i in range(self.num_classes)]
        batch_size = len(self.gt_mask)
        for i in range(batch_size):
            for cls_index in range(self.num_classes):
                gt_pos_mask = np.logical_and(
                    self.gt_mask[i], self.gt_cls[i] == cls_index
                ).reshape(-1)
                gt_bboxs = self.gt_bboxes[i][gt_pos_mask]

                pred_pos_mask = np.logical_and(
                    self.pred_mask[i], self.pred_cls[i] == cls_index
                ).reshape(-1)
                pred_bboxs_ = self.pred_bboxes[i][pred_pos_mask]
                pred_bboxs = pred_bboxs_[::, :4]
                pred_confs = pred_bboxs_[::, 4:]

                gt_p, pr_p = self._make_pairs(gt_bboxs, pred_bboxs)
                iou_list = []
                confs_list = []
                for j, gt_p_j in enumerate(gt_p):
                    iou_ = self._calc_iou(
                        gt_bboxs[gt_p_j], pred_bboxs[pr_p[j]]
                    )
                    iou_list.append(iou_)
                    confs_list.append(pred_confs[pr_p[j]])

                confidence[cls_index] = np.append(
                    confidence[cls_index], confs_list
                )
                iou[cls_index] = np.append(
                    iou[cls_index], iou_list
                )
                gt_number[cls_index] += gt_pos_mask.sum()

        res = []
        iou_thrs = np.arange(0.5, 0.95 + 0.05, 0.05)
        for iou_thr in iou_thrs:
            classes_metrics = {}
            for i in range(self.num_classes):
                is_tp_list = iou[i] >= iou_thr
                is_fp_list = iou[i] < iou_thr

                tp_list = is_tp_list.astype(np.int32)
                fp_list = is_fp_list.astype(np.int32)

                classes_metrics[i] = {
                    'confidence': confidence[i],
                    'tp': tp_list,
                    'fp': fp_list,
                    'count_gt': gt_number[i]
                }

            # mAP
            ap = []
            count_gt = []
            for metrics in classes_metrics.values():
                ap.append(
                    self._calculate_ap11(
                        metrics['confidence'],
                        metrics['tp'],
                        metrics['fp'],
                        metrics['count_gt'])
                )
                count_gt.append(metrics['count_gt'])
            ap = np.array(ap)
            macro_ap = np.mean(ap)

            res.append(macro_ap)
        return np.mean(res)

    @staticmethod
    def _calculate_ap11(confs_list: np.ndarray, tp_list: np.ndarray,
                        fp_list: np.ndarray, count_gt: int) -> float:
        """
        Calculate AP(average precision) with 11 points interpolation.

        Parameters
        ----------
        confs_list: np.ndarray
            List confidences.
        tp_list: np.ndarray
            List true positive values.
        fp_list: np.ndarray
            List false positive values.
        count_gt: int
            Count ground true.

        Return
        ------
        float
            Calculated AP.
        """
        sort_idxs = np.argsort(confs_list)[::-1]
        cum_tp = np.cumsum(tp_list[sort_idxs])
        cum_fp = np.cumsum(fp_list[sort_idxs])
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / count_gt

        cur = 0
        while cur != precision.size:
            idx = cur + precision[cur:].argmax()
            precision[cur:idx] = precision[idx]
            cur = idx + 1

        recall11 = np.arange(11) * 0.1
        precision11 = np.zeros_like(recall11)
        for i in range(recall.size)[::-1]:
            precision11[recall11 < recall[i]] = precision[i]

        return np.mean(precision11)


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


class COCOMeanAveragePrecision(Metric):
    """Compute mean average precision using pytcocotools."""

    def __init__(self, annotations, bbox_normalization=True):
        super().__init__()
        self.coco = COCO(annotations)
        self.empty = np.array(
            [[self.coco.getImgIds()[0], -1, -1, -1, -1, -1, -1.]]
        )
        self.cat_ids_dict = self.coco.getCatIds()
        self.bbox_normalization = bbox_normalization
        self.clear()
        # self.cat_ids_dict = {i: idx for i, idx in enumerate(cat_ids)}

    def clear(self):
        self.preds = []
        self.coco_eval = None

    def update(self, preds, gt):
        pred_bboxes, pred_cls, pred_mask = [
            x.asnumpy() if not isinstance(x, np.ndarray) else x for x in preds
        ]
        _, _, _, metas = [
            x.asnumpy() if not isinstance(x, np.ndarray) else x for x in gt
        ]
        batch_size = pred_bboxes.shape[0]

        for i in range(batch_size):
            mask = pred_mask[i].reshape(-1)

            bboxes = pred_bboxes[i][::, :4]
            bboxes = bboxes[mask]
            if self.bbox_normalization:
                bboxes = norm_bboxes(bboxes, metas[i])
            bboxes[::, [2, 3]] = bboxes[::, [2, 3]] - bboxes[::, [0, 1]] # + 1

            classes = pred_cls[i].reshape(-1)
            classes = classes[mask]
            classes = [
                np.array([self.cat_ids_dict[label]]) for label in classes
            ]

            conf = pred_bboxes[i][::, [4]]
            conf = conf[mask]

            img_id = np.array([metas[i][-1]])

            for j in range(bboxes.shape[0]):
                pred = np.concatenate([img_id, bboxes[j], conf[j], classes[j]])
                self.preds.append(pred)

    def eval(self):
        if self.preds:
            preds = np.array(self.preds).reshape(-1, 7)
        else:
            preds = self.empty
        coco_dets = self.coco.loadRes(preds)
        det_img_ids = self.coco.getImgIds()

        iou_type = 'bbox'
        self.coco_eval = DetectEval(self.coco, coco_dets, iou_type)

        tgt_ids = det_img_ids

        self.coco_eval.params.imgIds = tgt_ids
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()
        return self.coco_eval.stats[0]

    def dump_preds(self, path):
        """Save prediciont to file."""
        preds = []
        for pred in self.preds:
            obj = {
                'image_id': int(pred[0]),
                'bbox': pred[1:5].tolist(),
                'score': float(pred[5]),
                'category_id': int(pred[6])
            }
            preds.append(obj)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(preds, f, indent=1)

    def load_preds(self, path):
        """Load predictions from file."""
        with open(path, 'r', encoding='utf-8') as f:
            preds = json.load(f)

        for pred in preds:
            pred = [
                pred['image_id'], *pred['bbox'], pred['score'],
                pred['category_id']
            ]
            self.preds.append(np.array(pred, dtype=np.float32))
