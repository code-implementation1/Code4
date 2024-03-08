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
#
# This file or its part has been derived from the following repository
# and modified: https://github.com/open-mmlab/mmdetection/tree/v2.28.2
# ============================================================================
"""FCNMaskHead for Mask-RCNN detection models."""
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
import pycocotools.mask as mask_util

from ..layers import ConvModule, build_upsample_layer
from ..roi_extractors import ROIAlign

from .. import Config


def _pair(x: Union[int, Tuple[int, int]]):
    return (x, x) if isinstance(x, int) else x


def prepare_config(cfg=None):
    if isinstance(cfg, tuple):
        cfg = dict(cfg)
    elif isinstance(cfg, Config):
        cfg = cfg.as_dict().copy()

    return cfg


class FCNMaskHead(nn.Cell):
    """FCNMaskHead for Mask-RCNN detection models.

        Args:
            loss_mask (nn.Cell): Segmentation mask loss.
            train_batch_size (int): Training batch size.
            test_batch_size (int): Test batch size.
            loss_mask_weight (float): Weight of segmentation mask loss.
            num_convs (int): Number of convolution layers.
            roi_feat_size (int): Height and width of input ROI features maps.
            in_channels (int): Number of input channels
            conv_kernel_size (int): Convolution kernel size.
            conv_out_channels (int): Number of output channels.
            num_classes (int): Number of classes.
            class_agnostic (bool): If True, single mask will be generated
                else mask will be generated for each class.
            upsample_cfg (Union[Dict, Tuple]): Configuration for upsampling
                layer.
            conv_cfg (Union[Dict, Tuple, None]): Convolution configuration.
            norm_cfg (Union[Dict, Tuple, None]): Normalization configuration.
            train_cfg (Config): Training config.
            test_cfg (Config): Inference config.
    """

    def __init__(
            self,
            loss_mask: nn.Cell,
            train_batch_size: int,
            test_batch_size: int,
            loss_mask_weight: float = 1.0,
            num_convs: int = 4,
            roi_feat_size: int = 14,
            in_channels: int = 256,
            conv_kernel_size: int = 3,
            conv_out_channels: int = 256,
            num_classes: int = 80,
            class_agnostic: bool = False,
            upsample_cfg: Union[Dict, Tuple] = (
                ('type', 'deconv'), ('scale_factor', 2)),
            conv_cfg: Union[Dict, Tuple] = None,
            norm_cfg: Union[Dict, Tuple] = None,
            train_cfg: Config = Config(
                dict(
                    rpn_proposal=dict(max_per_img=1000)
                )
            ),
            test_cfg: Config = Config(
                dict(
                    rcnn=dict(
                        score_thr=0.05,
                        iou_threshold=0.5,
                        max_per_img=100
                    )
                )
            )
    ):
        """Init FCNMaskHead."""
        super(FCNMaskHead, self).__init__()
        self.upsample_cfg = prepare_config(upsample_cfg)
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.mask_size = _pair(train_cfg.rcnn.mask_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor')
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = prepare_config(conv_cfg)
        self.norm_cfg = prepare_config(norm_cfg)
        self.max_per_img = test_cfg.rcnn.max_per_img
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.loss_mask_weight = loss_mask_weight

        self.convs = nn.CellList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels
        )
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample_cfg.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
                has_bias=True
            )
        self.upsample = build_upsample_layer(self.upsample_cfg)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(
            logits_in_channel, out_channels, 1, has_bias=True
        )
        self.relu = nn.ReLU()

        self.loss_mask = loss_mask

        self.bboxes_range = ms.Tensor(
            np.arange(self.max_per_img * test_batch_size).reshape(-1, 1),
            ms.int32
        )

        self.mask_thr_binary = test_cfg.rcnn.mask_thr_binary
        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)
        self.eps = ms.Tensor(1e-7, ms.float32)

        self.gt_roi_align = ROIAlign(
            pooled_height=train_cfg.rcnn.mask_size,
            pooled_width=train_cfg.rcnn.mask_size
        )

    def init_weights(self):
        pass

    def construct(self, x):
        """Forward MaskHead."""
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def loss(self, seg_logits, labels, weights, seg_targets):
        """Loss method."""
        labels = ops.cast(labels, ms.int32) - 1
        fake_inds = ops.arange(0, seg_logits.shape[0])
        seg_logits = seg_logits[fake_inds, labels]

        weights = ops.cast(weights, ms.float32)

        # seg_mask_loss
        seg_targets = ops.cast(seg_targets, ms.float32)
        seg_mask_loss = self.loss_mask(seg_logits, seg_targets)
        seg_mask_loss = ops.reduce_mean(seg_mask_loss, (1, 2))
        seg_mask_loss = seg_mask_loss * weights
        seg_mask_loss = seg_mask_loss / (ops.reduce_sum(weights, (0,)) + self.eps)
        seg_mask_loss = ops.reduce_sum(seg_mask_loss, (0,))
        seg_mask_loss = seg_mask_loss * self.loss_mask_weight
        return seg_mask_loss

    def get_targets(self, pos_proposals, gt_masks, mask_divider):
        """Compute mask target for each positive proposal in the image.
        """
        n, maxh, maxw = gt_masks.shape
        pos_proposals = pos_proposals / mask_divider
        x1 = ops.clip(pos_proposals[:, [0]], 0, maxw)
        x2 = ops.clip(pos_proposals[:, [2]], 0, maxw)
        y1 = ops.clip(pos_proposals[:, [1]], 0, maxh)
        y2 = ops.clip(pos_proposals[:, [3]], 0, maxh)
        rois = ops.concat(
            (ops.arange(0, n, 1, dtype=ms.float32)[:, None], x1, y1, x2, y2),
            axis=1
        )
        targets = self.gt_roi_align(
            ops.cast(gt_masks[:, None, :, :], ms.float32), rois
        )
        targets = ops.round(targets)
        targets = ops.squeeze(targets)
        return targets

    def choose_masks(self, mask_logits, det_labels):
        """Choose computed masks by labels."""
        det_labels = ops.reshape(det_labels, (-1, 1))
        indices = ops.concat((self.bboxes_range, det_labels), axis=1)
        pred_masks = ops.gather_nd(mask_logits, indices)
        pred_masks = ops.sigmoid(pred_masks)
        pred_masks = ops.reshape(
            pred_masks,
            (
                self.test_batch_size, self.max_per_img,
                self.mask_size[0], self.mask_size[1]
            )
        )
        return pred_masks

    def get_masks(
            self, mask_pred, det_bboxes, ori_shape,
    ):
        """Get segmentation masks from mask_pred and bboxes.
        Args:
            mask_pred (ndarray): shape (n, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multiscale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (ndarray): shape (n, 4/5)
            ori_shape: original image size
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
                    f'there is invalid proposal bbox, index={i} bbox={bbox} '
                    f'w={w} h={h}'
                )
                w = max(w, 1)
                h = max(h, 1)

            mask_pred_ = mask_pred[i, :, :]

            bbox_mask = cv2.resize(
                mask_pred_, (w, h), interpolation=cv2.INTER_LINEAR
            )
            bbox_mask = (bbox_mask > self.mask_thr_binary).astype(np.uint8)

            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

        return segms

    def get_seg_masks(self, mask_pred, det_bboxes, ori_shape, rle=True):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            ori_shape (Tuple): original image height and width, shape (2,).
            rle (bool): If True, encode mask to RLE format.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.
        """
        bboxes = det_bboxes[:, :4]
        img_h = int(ori_shape[0])
        img_w = int(ori_shape[1])

        n = len(mask_pred)
        num_chunks = n

        if num_chunks > 0:
            chunks = np.array_split(np.arange(n), num_chunks)
        else:
            chunks = []
        threshold = self.mask_thr_binary
        im_masks = np.zeros(shape=(n, img_h, img_w), dtype=np.uint8)
        segms = []

        for inds in chunks:
            masks_chunk, spatial_inds = self._do_paste_mask(
                mask_pred[inds], bboxes[inds], img_h, img_w
            )

            if threshold >= 0:
                mask = (masks_chunk >= threshold).astype(dtype=np.uint8)
            else:
                # for visualization and debugging
                mask = (masks_chunk * 255).astype(dtype=np.uint8)

            im_masks[(inds,) + spatial_inds] = mask

        if rle:
            for i in range(n):
                rle = mask_util.encode(
                    np.array(im_masks[i][:, :, np.newaxis], order='F')
                )[0]
                segms.append(rle)
            return segms

        return im_masks

    def _do_paste_mask(self, masks, boxes, img_h, img_w):
        """Paste instance masks according to boxes.

        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        Args:
            masks (Tensor): N, 1, H, W
            boxes (Tensor): N, 4
            img_h (int): Height of the image to be pasted.
            img_w (int): Width of the image to be pasted.

        Returns:
            tuple: (Tensor, tuple). The first item is mask tensor, the second one
                is the slice object.
            If skip_empty == False, the whole image will be pasted. It will
                return a mask of shape (N, img_h, img_w) and an empty tuple.
            If skip_empty == True, only area around the mask will be pasted.
                A mask of shape (N, h', w') and its start and end coordinates
                in the original image will be returned.
        """
        # On GPU, paste all masks together (up to chunk size)
        # by using the entire image to sample the masks
        # Compared to pasting them one by one,
        # this has more operations but is faster on COCO-scale dataset.
        x0_int = np.clip(
            np.floor(boxes[:, 0].min()) - 1, a_min=0, a_max=img_w
        ).astype(np.int32)
        y0_int = np.clip(
            np.floor(boxes[:, 1].min()) - 1, a_min=0, a_max=img_h
        ).astype(np.int32)
        x1_int = np.clip(
            np.ceil(boxes[:, 2].max()) + 1, a_min=0, a_max=img_w
        ).astype(np.int32)
        y1_int = np.clip(
            np.ceil(boxes[:, 3].max()) + 1, a_min=0, a_max=img_h
        ).astype(np.int32)

        x0, y0, x1, y1 = np.split(boxes, 4, axis=1)

        img_y = np.arange(y0_int, y1_int).astype(np.float32) + 0.5
        img_x = np.arange(x0_int, x1_int).astype(np.float32) + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        # img_x, img_y have shapes (N, w), (N, h)

        if np.isinf(img_x).any():
            inds = np.where(np.isinf(img_x))
            img_x[inds] = 0
        if np.isinf(img_y).any():
            inds = np.where(np.isinf(img_y))
            img_y[inds] = 0

        gx = np.tile(img_x[:, np.newaxis, :], (1, img_y.shape[1], 1))
        gy = np.tile(img_y[:, :, np.newaxis], (1, 1, img_x.shape[1]))
        grid = np.stack([gx, gy], axis=3)

        img_mask = ops.grid_sample(
            ms.Tensor(masks[np.newaxis, ...]), ms.Tensor(grid),
            align_corners=False
        ).asnumpy()

        return img_mask, (slice(y0_int, y1_int), slice(x0_int, x1_int))
