# Contents

* [HRNetV2 Description](#hrnetv2-description)
    * [Model Architecture](#model-architecture)
    * [Dataset](#dataset)
* [Environment Requirements](#environment-requirements)
* [Quick Start](#quick-start)
    * [Prepare the model](#prepare-the-model)
    * [Run the scripts](#run-the-scripts)
* [Script Description](#script-description)
    * [Script and Sample Code](#script-and-sample-code)
    * [Script Parameters](#script-parameters)
* [Training](#training)
    * [Training process](#training-process)
* [Evaluation](#evaluation)
    * [Evaluation process](#evaluation-process)
        * [Evaluation on GPU](#evaluation-on-gpu)
    * [Evaluation result](#evaluation-result)
* [Model Description](#model-description)
    * [Performance](#performance)
* [Description of Random Situation](#description-of-random-situation)
* [ModelZoo Homepage](#modelzoo-homepage)

## [HRNetV2 Description](#contents)

High-resolution representation learning plays an essential role in many vision problems,
e.g., pose estimation and semantic segmentation. The high-resolution network (HRNet),
recently developed for human pose estimation, maintains high-resolution representations
through the whole process by connecting high-to-low resolution convolutions in parallel
and produces strong high-resolution representations by repeatedly conducting fusions
across parallel convolutions.

In this paper, we conduct a further study on high-resolution representations by
introducing a simple yet effective modification and apply it to a wide range of vision
tasks. We augment the high-resolution representation by aggregating the (upsampled)
representations from all the parallel convolutions rather than only the representation
from the high-resolution convolution as done in "K. Sun, B. Xiao, D. Liu, and J. Wang.
Deep high-resolution representation learning for human pose estimation. In CVPR, 2019".
This simple modification leads to stronger representations, evidenced by superior results.
We show top results in semantic segmentation on Cityscapes, LIP, and PASCAL Context, and
facial landmark detection on AFLW, COFW, 300W, and WFLW. In addition, we build a
multi-level representation from the high-resolution representation and apply it to the
Faster R-CNN object detection framework and the extended frameworks. The proposed
approach achieves superior results to existing single-model networks on COCO object
detection.

[Paper](https://arxiv.org/abs/1904.04514): Ke Sun, Yang Zhao, Borui Jiang, Tianheng Cheng,
Bin Xiao, Dong Liu, Yadong Mu, Xinggang Wang, Wenyu Liu, Jingdong Wang.
High-Resolution Representations for Labeling Pixels and Regions (2019).

### [Model Architecture](#contents)

The model is based on the Mask R-CNN configuration but the backbone is HRNet
and the neck module is HRFPN.
Other blocks are the same as in the Mask R-CNN model: proposal generator, target assigner,
bounding box processing functions.

### [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [COCO-2017](https://cocodataset.org/#download)

* Dataset size: 25.4G
    * Train: 18.0G，118287 images
    * Val: 777.1M，5000 images
    * Test: 6.2G，40670 images
    * Annotations: 474.7M, represented in 3 JSON files for each subset.
* Data format: image and json files.
    * Note: Data will be processed in dataset.py

## [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset COCO-2017.
* Install third-parties requirements:

```text
numpy~=1.21.2
opencv-python~=4.5.4.58
pycocotools>=2.0.5
matplotlib
seaborn
tqdm==4.64.1
decorator~=5.1.1
```

* We use COCO-2017 as training dataset in this example by default, and you
 can also use your own datasets. Dataset structure:

```shell
.
└── coco-2017
    ├── train
    │   ├── data
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   └── labels.json
    ├── validation
    │   ├── data
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   └── labels.json
    └── test
        ├── data
        │    ├── 000000000001.jpg
        │    ├── 000000000002.jpg
        │    └── ...
        └── labels.json
```

The training/evaluation scripts use raw images and annotation files. There is no need to prepare any MindRecord dataset.

## [Quick Start](#contents)

### [Prepare the model](#contents)

1. Prepare yaml config file. Create file and copy content from
 `default_config.yaml` to created file.
2. Change data settings: experiment folder (`train_outputs`), image size
 settings (`img_width`, `img_height`, etc.), subsets folders (`train_dataset`,
 `val_dataset`), information about categories etc.
3. Change other training hyperparameters (learning rate, regularization etc.).
4. Prepare pre-trained checkpoints.

### [Run the scripts](#contents)

After installing MindSpore via the official website, you can start training and
evaluation as follows:

* running on GPU

```shell
# distributed training on GPU
bash scripts/run_distributed_training_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] (OPTIONAL)[PRETRAINED_PATH]

# run eval on GPU
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
HRNetV2
├── README.md            # description of HRNetV2
├── convert_from_pt.py   # convert model weights from PyTorch model
├── eval.py              # evaluation script (MindSpore)
├── infer.py             # run inference on single file/directory (MindSpore)
├── train.py             # training script
├── configs              # model configurations
│   ├── default_config.yaml  # default model configuration (Mask R-CNN HRNetV2p-W32)
│   ├── mask_rcnn_hrnetv2p_w18_1x_coco.yaml  # configuration for Mask R-CNN HRNetV2p-W18
│   └── mask_rcnn_hrnetv2p_w32_1x_coco.yaml  # configuration for Mask R-CNN HRNetV2p-W32
├── scripts                              # shell scripts:
│   ├── build_custom.sh                  # build ROIAlign layer (CPU+GPU)
│   ├── run_eval_gpu.sh                  # run evaluation on GPU
│   ├── run_infer_gpu.sh                 # run inference on single image/folder on GPU
│   ├── run_distributed_training_gpu.sh  # run training on multiple GPUs
│   └── run_standalone_training_gpu.sh   # run training on single GPU
└── src
    ├── callback.py        # callback functions for training
    ├── common.py          # base configuration of training pipeline
    ├── config.py          # configuration parser and default arguments
    ├── detecteval.py      # wrapper functions for evaluation process
    ├── device_adapter.py  # functions for distributed training (MPI 'world' identifiers)
    ├── eval_utils.py      # COCO evaluation class (collect and calculate detection metrics)
    ├── lr_schedule.py     # learning rate scheduling function
    ├── mlflow_funcs.py    # MLflow metric logging adapters
    ├── network_define.py  # training wrappers for network
    ├── data
    │   ├── coco.py        # wrapper for reading ImageNet dataset
    │   └── dataset.py     # wrapper for reading ImageNet dataset
    └── models
        └── hrnetv2
            ├── hrnetv2.py             # base HRNetV2 model (architecture + inference wrapper)
            ├── bin                    # ROIAlign compiled layer (created by script)
            │   ├── roi_align_cpu.so   # ROIAlign CPU implementation
            │   └── roi_align_gpu.so   # ROIAlign GPU implementation
            └── layers                 # HRNetV2 layers
                ├── cpu
                │   └── roi_align.cpp  # ROIAlign for CPU
                ├── cuda
                │   └── roi_align_cuda_kernel.cu  # ROIAlign for GPU (Nvidia CUDA)
                ├── targets
                │   ├── assigner_sampler.py  # target assigner class
                │   ├── max_iou_assigner.py  # maximal IoU assigner class
                │   └── random_sampler.py    # random target sampler
                ├── anchor_generator.py      # anchor generator class
                ├── backbone.py              # HRNet model backbone
                ├── bbox_coder.py            # DeltaXYWH bounding box coder class
                ├── conv_module.py           # convolution module (for compatibility)
                ├── fcn_mask_head.py         # FCN mask head layer
                ├── initialization.py        # initializers for layers/blocks
                ├── iou2d_calculator.py      # IoU metric calculator for training/inference
                ├── misc.py                  # Identity and FixedResize pseudo-layers
                ├── neck.py                  # FPN neck layer
                ├── proposal_generator.py    # proposal generator for detection model
                ├── resnet.py                # ResNet backbone (basic blocks)
                ├── roi_head.py              # standard ROI head layer
                ├── rpn.py                   # RPN head layer
                ├── shared_2fc_bbox_head.py  # shared 2FC bounding box head layer
                ├── single_roi_extractor.py  # single ROI extractor module
                └── transforms.py            # helper functions for bounding box calculations
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```yaml
# Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
enable_modelarts: 0
data_url: ""
train_url: "/cache/data/mask_rcnn_models"
data_path: "/cache/data"
output_path: "/cache/train"
prediction_path: 'preds.json'
checkpoint_path: ''
load_path: "/cache/checkpoint_path"
enable_profiling: 0

train_outputs: '/data/mask_rcnn_models'
pre_trained: ''
brief: 'gpu-1_1024x1024'
device_target: GPU
mode: 'graph'
# ==============================================================================
detector: 'mask_rcnn'

# backbone
backbone:
  extra:
    stage1:
      num_modules: 1
      num_branches: 1
      block: 'BOTTLENECK'
      num_blocks: [4]
      num_channels: [64]
    stage2:
      num_modules: 1
      num_branches: 2
      block: 'BASIC'
      num_blocks: [4, 4]
      num_channels: [32, 64]
    stage3:
      num_modules: 4
      num_branches: 3
      block: 'BASIC'
      num_blocks: [4, 4, 4]
      num_channels: [32, 64, 128]
    stage4:
      num_modules: 3
      num_branches: 4
      block: 'BASIC'
      num_blocks: [4, 4, 4, 4]
      num_channels: [32, 64, 128, 256]
  pretrained: '/data/backbones/hrnetv2_w32.ckpt'

# neck
neck:
  fpn:
    in_channels: [32, 64, 128, 256]
    out_channels: 256
    num_outs: 5

# rpn
rpn:
  in_channels: 256
  feat_channels: 256
  num_classes: 1
  bbox_coder:
    target_means: [0., 0., 0., 0.]
    target_stds: [1.0, 1.0, 1.0, 1.0]
  loss_cls:
    loss_weight: 1.0
  loss_bbox:
    loss_weight: 1.0
  anchor_generator:
    scales: [8]
    strides: [4, 8, 16, 32, 64]
    ratios: [0.5, 1.0, 2.0]

bbox_head:
  in_channels: 256
  fc_out_channels: 1024
  roi_feat_size: 7
  reg_class_agnostic: False
  bbox_coder:
    target_means: [0.0, 0.0, 0.0, 0.0]
    target_stds: [0.1, 0.1, 0.2, 0.2]
  loss_cls:
    loss_weight: 1.0
  loss_bbox:
    loss_weight: 1.0

mask_head:
  num_convs: 4
  in_channels: 256
  conv_out_channels: 256
  loss_mask:
    loss_weight: 1.0

# roi_align
roi:
  roi_layer: {type: 'RoIAlign', out_size: 7, sampling_ratio: 0}
  out_channels: 256
  featmap_strides: [4, 8, 16, 32]
  finest_scale: 56
  sample_num: 640

mask_roi:
  roi_layer: {type: 'RoIAlign', out_size: 14, sampling_ratio: 0}
  out_channels: 256
  featmap_strides: [4, 8, 16, 32]
  finest_scale: 56
  sample_num: 128

train_cfg:
  rpn:
    assigner:
      pos_iou_thr: 0.7
      neg_iou_thr: 0.3
      min_pos_iou: 0.3
      match_low_quality: True
    sampler:
      num: 256
      pos_fraction: 0.5
      neg_pos_ub: -1
      add_gt_as_proposals: False
  rpn_proposal:
      nms_pre: 2000
      max_per_img: 1000
      iou_threshold: 0.7
      min_bbox_size: 0
  rcnn:
      assigner:
        pos_iou_thr: 0.5
        neg_iou_thr: 0.5
        min_pos_iou: 0.5
        match_low_quality: True
      sampler:
        num: 512
        pos_fraction: 0.25
        neg_pos_ub: -1
        add_gt_as_proposals: True
      mask_size: 28

test_cfg:
  rpn:
    nms_pre: 1000
    max_per_img: 1000
    iou_threshold: 0.7
    min_bbox_size: 0
  rcnn:
    score_thr: 0.05
    iou_threshold: 0.5
    max_per_img: 100
    mask_thr_binary: 0.5

# optimizer
opt_type: "sgd"
lr: 0.02
accumulate_step: 1
min_lr: 0.0000001
momentum: 0.9
weight_decay: 0.0001
warmup_step: 500
warmup_ratio: 0.001
lr_steps: [8, 11]
lr_type: "multistep"
grad_clip: 0


# train
num_gts: 100
batch_size: 2
test_batch_size: 1
loss_scale: 256
epoch_size: 12
run_eval: 1
eval_every: 1
enable_graph_kernel: 0
finetune: 0
datasink: 0

#distribution training
run_distribute: 0
device_id: 0
device_num: 1
rank_id: 0

# Number of threads used to process the dataset in parallel
num_parallel_workers: 1
# Parallelize Python operations with multiple worker processes
python_multiprocessing: 0
# dataset setting
train_dataset: '/data/coco-2017/train'
val_dataset: '/data/coco-2017/validation'
coco_classes: ['background', 'person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
num_classes: 81
train_dataset_num: 0
train_dataset_divider: 0

# images
img_width: 1024
img_height: 1024
mask_divider: 1.0
divider: 32
img_mean: [123.675, 116.28, 103.53]
img_std: [58.395, 57.12, 57.375]
to_rgb: 1
keep_ratio: 1

# augmentation
flip_ratio: 0.5
expand_ratio: 0.0

# callbacks
save_every: 100
keep_checkpoint_max: 5
keep_best_checkpoints_max: 5

---
# Config description for each option
enable_modelarts: 'Whether training on modelarts, default: False'

device_target: "device where the code will be implemented, default is GPU."
train_outputs: "Path for folder with experiments."
brief: "Short experiment name, experiment folder will arrive in `train_outputs` folder. `brief` will suffix of experiment folder name."
img_width: "Input images weight."
img_height: "Input images height."

lr: "Base learning rate value."
batch_size: "Training batch size."
pre_trained: "Path to pretraining model (resume training or train new fine tuned model)."

---
train_data_type: ["coco", "mindrecord"]
val_data_type: ["coco", "mindrecord"]
device_target: ["GPU"]
backbone_type: ["resnet", "resnext"]
```

## [Training](#contents)

To train the model, run `train.py`.

### [Training process](#contents)

Standalone training mode:

```bash
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH]
```

* `CONFIG_PATH`: path to config file.

Training result will be stored in the path passed in config option `train_outputs`

## [Evaluation](#contents)

### [Evaluation process](#contents)

#### [Evaluation on GPU](#contents)

```shell
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH]
```

* `CONFIG_PATH`: path to config file.
* `CHECKPOINT_PATH`: path to checkpoint.

### [Evaluation result](#contents)

Result for GPU:

```log
Start Eval!
loading annotations into memory...
Done (t=0.48s)
creating index...
index created!

========================================

total images num:  5000
Processing, please wait a moment.
100%|██████████| 5000/5000 [0:59:32<00:00,  1.29it/s]
Loading and preparing results...
Converting ndarray to lists...
(138134, 7)
0/138134
DONE (t=0.92s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=27.15s).
Accumulating evaluation results...
DONE (t=4.65s).
Eval result (bboxes):
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.604
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.690
mAP: 0.405

Eval result (segmentation):
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.574
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649
mAP: 0.363
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | GPU                                    |
|---------------------|----------------------------------------|
| Model Version       | Mask R-CNN HRNetV2p-W32                |
| Resource            | NVIDIA GeForce RTX 3090 (x8)           |
| Uploaded Date       | 2023-11-10 (YYYY-MM-DD)                |
| MindSpore Version   | 2.1.0                                  |
| Dataset             | COCO2017                               |
| Pretrained          | noisy weights (start mAP=39.0%)        |
| Training Parameters | epoch = 6, batch_size = 3 (per device) |
| Optimizer           | SGD (momentum)                         |
| Loss Function       | box head loss, mask head loss          |
| Speed               | 2855 ms/step                           |
| Total time          | 23h 47m 34s                            |
| outputs             | mAP                                    |
| mAP                 | 39.6%                                  |
| Model for inference | ???                                    |
| configuration       | mask_rcnn_hrnetv2p_w32_1x_coco.yaml    |
| Scripts             |                                        |

## [Description of Random Situation](#contents)

We use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).


