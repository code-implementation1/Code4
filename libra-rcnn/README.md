# Contents

* [Contents](#contents)
    * [Libra R-CNN Description](#librarcnn-description)
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
        * [Training Process](#training-process)
        * [Transfer Training](#transfer-training)
        * [Distribute training](#distribute-training)
    * [Evaluation](#evaluation)
        * [Evaluation Process](#evaluation-process)
            * [Evaluation on GPU](#evaluation-on-gpu)
        * [Evaluation result](#evaluation-result)
    * [Inference](#inference)
        * [Inference Process](#inference-process)
            * [Inference on GPU](#inference-on-gpu)
        * [Inference result](#inference-result)
   * [Model Description](#model-description)
        * [Performance](#performance)
   * [Description of Random Situation](#random-situation)
   * [ModelZoo Homepage](#modelzoo-homepage)

## [Libra R-CNN Description](#contents)

Compared with model architectures, the training process, which is also crucial
to the success of detectors, has received relatively less attention in object
detection. Revisiting the standard training practice of detectors shows that
the detection performance is often limited by the imbalance during the training
process, which generally consists of three levels - sample level, feature
level, and objective level. To mitigate the adverse effects caused thereby,
Libra R-CNN was proposed. It is a simple but effective framework towards
balanced learning for object detection. It integrates three novel components:
IoU-balanced sampling, balanced feature pyramid, and balanced L1 loss,
respectively for reducing the imbalance at sample, feature, and objective
level. Benefited from the overall balanced design, Libra R-CNN significantly
improves the detection performance.

[Paper](https://arxiv.org/abs/1904.02701): Jiangmiao Pang, Kai Chen,
Jianping Shi, Huajun Feng, Wanli Ouyang, Dahua Lin. Computer Vision and Pattern
Recognition (CVPR), 2019 (In press).

### [Model Architecture](#contents)

**Overview of the pipeline of Libra R-CNN:**
Region proposals are obtained from RPN and used for RoI feature extraction
from the output feature maps of a CNN backbone. The RoI features are used to
perform classification and localization. Model has:

1. additional Balance Features Pyramid (BFP) block that applied after FPN as
 part of Neck. Block was designed to eliminate imbalance on the features level.
 the extracted visual features become fully utilized.
2. Balanced Sampler that allows the model to train on more samples that are
 situated near GT bounding boxes (hard samples). The selected region samples
 become more representative.
3. BalancedL1Loss that allows the RCNN block localization branch to fit with
 more balanced gradients. The designed objective function becomes more optimal.

Libra R-CNN result prediction pipeline:

1. Backbone (ResNet, ResNeXt).
2. Neck (FPN + BFP).
3. RPN.
4. Proposal generator.
5. ROI extractor (based on ROIAlign operation).
6. Shared2FCBBoxHead (standard RCNN).
7. multiclass NMS (reduce number of proposed boxes and omit objects with low
 confidence).

Libra R-CNN result training pipeline:

1. Backbone (ResNet, ResNeXt).
2. Neck (FPN + BFP).
3. RPN.
4. RPN Assigner+Sampler (Random sampler).
5. RPN Classification + Localization losses.
6. Proposal generator.
7. RCNN Assigner+Sampler (Balanced sampler).
8. ROI extractor (based on ROIAlign operation).
9. Shared2FCBBoxHead (standard RCNN).
10. RCNN Classification + Localization losses.

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
* Build CUDA and CPP dependencies:

```shell
bash scripts/build_custom.sh
```

* Install third-parties requirements:

```text
Cython~=0.28.5
mindinsight
numpy~=1.21.2
opencv-python~=4.5.4.58
pycocotools>=2.0.5
matplotlib
seaborn
pandas
tqdm==4.64.1
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

* Also we make evaluation only on mindrecord converted dataset. Use the
 `convert_dataset.py` script to convert original COCO subset to mindrecord.

```shell
python convert_dataset.py --config_path default_config.yaml --converted_coco_path data/coco-2017/validation --converted_mindrecord_path data/mindrecord/validation
```

Result mindrecord dataset will have next format:

```shell
.
└── mindrecord
    ├── validation
    │   ├── file.mindrecord
    │   ├── file.mindrecord.db
    │   └── labels.json
    └── test
        ├── file.mindrecord
        ├── file.mindrecord.db
        └── labels.json
```

It is possible to convert train dataset to mindrecord format and use it to
train model.

## [Quick Start](#contents)

### [Prepare the model](#contents)

1. Prepare yaml config file. Create file and copy content from
 `default_config.yaml` to created file.
1. Change data settings: experiment folder (`train_outputs`), image size
 settings (`img_width`, `img_height`, etc.), subsets folders (`train_dataset`,
 `val_dataset`, `train_data_type`, `val_data_type`), information about
 categories etc.
1. Change the backbone settings: type (`backbone.type`), path to pretrained
 ImageNet weights (`backbone.pretrained`), layer freezing settings
 (`backbone.frozen_stages`).
1. Change other training hyper parameters (learning rate, regularization,
 augmentations etc.).
1. Prepare pre_trained checkpoints.

### [Run the scripts](#contents)

After installing MindSpore via the official website, you can start training and
evaluation as follows:

* running on GPU

```shell
# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]

# run eval on GPU
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (Optional)[PREDICTION_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└─ cv
  └─ libra-rcnn
    ├─ README.md                       ## descriptions about Libra R-CNN
    ├─ configs
    |  ├── default_config.yaml            ## resnet50 config
    |  ├── resnet101_config.sh            ## resnet101 config
    |  └── resnext101_config.sh           ## resnext101 config
    ├─ scripts
    |  ├── build_custom.sh              ## build dependencies
    |  ├── run_distribute_train_gpu.sh  ## bash script for distributed on gpu
    |  ├── run_eval_gpu.sh              ## bash script for eval on gpu
    |  ├── run_infer_gpu.sh             ## bash script for gpu model inference
    |  └── run_standalone_train_gpu.sh  ## bash script for distributed on gpu
    ├─ src
    |  ├── LibraRcnn
    |  |    ├── cuda  
    |  |    |   └── roi_align_cuda_kernel.cu      ## RoIAlign CUDA code
    │  │    ├── neck
    │  │    │   ├── bfp.py                        ## Balanced Feature Pyramid
    │  │    │   ├── conv_module.py                ## Convolution module
    │  │    │   ├── fpn.py                        ## Feature Pyramid Network
    │  │    │   ├── __init__.py                   ## init file
    │  │    │   └── non_local_2d.py               ## Non-local block
    │  │    ├── sampling
    │  │    │    ├── assigner_sampler.py          ## Wrapper for assigner and sampler
    │  │    │    ├── balanced_sampler.py          ## Balanced sampler
    │  │    │    ├── __init__.py                  ## init file
    │  │    │    ├── max_iou_assigner.py          ## MaxIOUAssigner
    │  │    │    └── random_sampler.py            ## Random sampler
    |  |    ├── __init__.py               ## init file
    │  │    ├── anchor_generator.py       ## Anchor generator
    │  │    ├── balanced_l1_loss.py       ## implemented BalancedL1Loss
    │  │    ├── bbox_coder.py             ## implemented BoundingBbox coder
    │  │    ├── bbox_head_libra.py        ## RCNN block
    │  │    ├── libra_rcnn.py             ## Full model with implemented pipeline
    │  │    ├── proposal_generator.py     ## Generate proposal bounding bboxes.
    │  │    ├── resnet.py                 ## ResNet
    │  │    ├── resnext.py                ## ResNext
    │  │    ├── roi_align.py              ## Single ROI Extractor
    │  │    ├── rpn.py                    ## Region Proposals Network
    │  │    └── sampling_builder.py       ## Functional for assigner-sampler object generation.
    |  ├── model_utils
    |  |    ├── __init__.py                  ## init file
    |  |    ├── config.py                    ## configuration file parsing utils
    |  |    ├── device_adapter.py            ## file to adapt current used devices
    |  |    ├── local_adapter.py             ## file to work with local devices
    |  |    └── moxing_adapter.py            ## file to work with model arts devices
    |  ├── __init__.py                   ## init file
    |  ├── callback.py                   ## callbacks
    |  ├── common.py                     ## common functional with common setting
    |  ├── dataset.py                    ## images loading and preprocessing
    |  ├── detecteval.py                 ## DetectEval class to analyze predictions
    |  ├── eval_utils.py                 ## evaluation metrics
    |  ├── lr_schedule.py                ## optimizer settings
    |  ├── mlflow_funcs.py               ## mlflow utilities
    |  └── network_define.py             ## model wrappers for training
    ├── __init__.py                        ## init file
    ├── draw_predictions.py                ## draw inference results on single image
    ├── convert_dataset.py                 ## script to convert dataset to mindrecord format
    ├── default_config.yaml                ## default configuration file
    ├── eval.py                            ## eval script (ckpt)
    ├── infer.py                           ## inference script (ckpt)
    ├── requirements.txt                   ## list of requirements
    └── train.py                           ## train script
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```shell
train_outputs: '/data/libra_rcnn_models'                        ## Path to folder with experiments  
brief: 'gpu-1_1024x1024'                                        ## Experiment short name
device_target: GPU                                              ## Device
# ===================================================
backbone:
  type: "resnet"                                                ## Backbone type (resnet or resnext)
  depth: 50                                                     ## Backbone depth
  pretrained: '/data/backbones/resnet50.ckpt'                   ## Path to pretrained backbone
  frozen_stages: 1                                              ## Number of frozen stages (use if backbone is pretrained, set -1 to not freeze)
  norm_eval: True                                               ## Whether make batch normalization work in eval mode all time (use if backbone is pretrained)
  num_stages: 4                                                 ## Number of resnet/resnext stages
  out_indices: [0, 1, 2, 3]                                     ## Define indices of resnet/resnext outputs
  groups: 1                                                     ## Group number in ResNext layers (only for resnet)
  base_width: 4                                                 ## Base width in ResNext layers (only for resnext)

neck:
  fpn:
    in_channels: [256, 512, 1024, 2048]                         ## Channels number for each input feature map (FPN)
    out_channels: 256                                           ## Channels number for each output feature map (FPN)
    num_outs: 5                                                 ## Number of output feature map (FPN)
  bfp:
    in_channels: 256                                            ## Channels number for each input feature map (BFP)
    num_levels: 5                                               ## Number of output feature map (BFP)
    refine_level: 2                                             ## Index of integration and refine level of BSF in multi-level features
    refine_type: 'non_local'                                    ## Type of the refine op

rpn:
  in_channels: 256                                              ## Input channels
  feat_channels: 256                                            ## Number of channels in intermediate feature map in RPN
  cls_out_channels: 1                                           ## Output classes number
  target_means: [0., 0., 0., 0.]                                ## Parameter for bbox encoding (RPN targets generation)
  target_stds: [1.0, 1.0, 1.0, 1.0]                             ## Parameter for bbox encoding (RPN targets generation)
  cls_loss:
    weight: 1.0                                                 ## RPN classification loss weight
  reg_loss:
    weight: 1.0                                                 ## RPN localization loss weight
  bbox_assign_sampler:
    neg_iou_thr: 0.3                                            ## IoU threshold for negative bboxes
    pos_iou_thr: 0.7                                            ## IoU threshold for positive bboxes
    min_pos_iou: 0.3                                            ## Minimum iou for a bbox to be considered as a positive bbox
    num_expected_neg: 256                                       ## Max number of negative samples in RPN
    num_expected_pos: 128                                       ## Max number of positive samples in RPN
    neg_pos_ub: 5                                               ## Max positive-negative samples ratio
    match_low_quality: True                                     ## Allow low quality matches


anchor_generator:
  scales: [8]                                                   ## Anchor scales
  ratios: [0.5, 1.0, 2.0]                                       ## Anchor ratios
  strides: [4, 8, 16, 32, 64]                                   ## Anchor strides for each feature map

proposal:
  activate_num_classes: 2                                       ##
  use_sigmoid_cls: True                                         ## Whether use sigmoid or softmax to obtained confidence
  train:
    nms_pre: 2000                                               ## max number of samples per level
    max_num: 1000                                               ## max number of output samples
    nms_thr: 0.7                                                ## NMS threshold for proposal generator
    min_bbox_size: 0                                            ## min bboxes size
  test:
    nms_pre: 1000                                               ## max number of samples per level
    max_num: 1000                                               ## max number of output samples
    nms_thr: 0.7                                                ## NMS threshold for proposal generator
    min_bbox_size: 0                                            ## min bboxes size

rcnn:
  in_channels: 256                                              ## Number of input channels
  fc_out_channels: 1024                                         ## Number of intermediate channels before classification
  roi_feat_size: 7                                              ## Input feature map side length
  bbox_coder:
    target_means: [0.0, 0.0, 0.0, 0.0]                          ## Bounding box coder parameter
    target_stds: [0.1, 0.1, 0.2, 0.2]                           ## Bounding box coder parameter
  cls_loss:
    weight: 1.0                                                 ## Classification loss weight
  reg_loss:
    alpha: 0.5                                                  ##
    gamma: 1.5                                                  ##
    beta: 1.0                                                   ##
    weight: 1.0                                                 ## Localization loss weight
  assign_sampler:
    neg_iou_thr: 0.5                                            ## IoU threshold for negative bboxes.
    pos_iou_thr: 0.5                                            ## IoU threshold for positive bboxes.
    min_pos_iou: 0.5                                            ## Minimum IOU for a bbox to be considered as a positive bbox
    num_bboxes: 1000                                            ## Number of input bboxes to assigner-sampler
    num_expected_pos: 128                                       ## Max number of sampled positive bboxes
    num_expected_neg: 512                                       ## Max number of sampled negative bboxes
    floor_thr: -1                                               ## Low border for balanced bins
    floor_fraction: 0                                           ## Fraction of negative samples that are not in balanced bins
    num_bins: 3                                                 ## Number of balanced bins
    neg_pos_ub: -1                                              ## max positive-negative samples ratio
    match_low_quality: False                                    ## Allow low quality match
  score_thr: 0.05                                               ## Confidence threshold
  iou_thr: 0.5                                                  ## IOU threshold

roi:
  roi_layer: {type: 'RoIAlign', out_size: 7, sampling_ratio: 0} ## RoI configuration
  out_channels: 256                                             ## out roi channels
  featmap_strides: [4, 8, 16, 32]                               ## strides for RoIAlign layer
  finest_scale: 56                                              ## parameter that define roi level
  sample_num: 640                                               ##


# optimizer
opt_type: "sgd"                                                 ## Optimizer type (sgd or adam)
lr: 0.002                                                       ## Base learning rate
min_lr: 0.000001                                                ## Minimal valuer for learning rate
momentum: 0.9                                                   ## Optimizer parameter
weight_decay: 0.0001                                            ## Regularization
warmup_step: 500                                                ## Number of warmup steps
warmup_ratio: 0.001                                             ## Initial learning rate = base_lr * warmup_ratio
lr_steps: [8, 11]                                               ## Epochs numbers when learning rate is divided by 10 (for multistep lr_type)
lr_type: "dynamic"                                              ## Learning rate scheduling type
grad_clip: 0                                                    ## Gradient clipping (set 0 to turn off)

# train
num_gts: 100
batch_size: 4                                                   ## Train batch size
test_batch_size: 1                                              ## Test batch size
loss_scale: 256                                                 ## Loss scale
epoch_size: 12                                                  ## Number of epochs
run_eval: 1                                                     ## Whether evaluation or not
eval_every: 1                                                   ## Evaluation interval
enable_graph_kernel: 0                                          ## Turn on kernel fusion
finetune: 0                                                     ## Turn on finetune (for transfer learning)
datasink: 0                                                     ## Turn on datasink mode
pre_trained: 'libra_faster_rcnn_r50_fpn_1x_coco_noised.ckpt'    ## Path to pretrained model weights

#distribution training
run_distribute: 0                                               ## Turn on distributed training
device_id: 0                                                    ##
device_num: 1                                                   ## Number of devices (only if distributed training turned on)
rank_id: 0                                                      ##

num_parallel_workers: 6                                         ## Number of threads used to process the dataset in parallel
python_multiprocessing: 0                                       ## Parallelize Python operations with multiple worker processes

# dataset setting
train_data_type: 'coco'                                                         ## Train dataset type
val_data_type: 'mindrecord'                                                     ## Validation dataset type
train_dataset: '/data/coco-2017/train'                                          ## Path to train dataset
val_dataset: '/data/mindrecord_eval'                                            ## Path to validation dataset
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
num_classes: 81                                                              ## Number of classes (including background)
train_dataset_num: 0                                                         ## Parameter to reduce training data (debugging)
train_dataset_divider: 0                                                     ## Parameter to reduce training data (debugging)

# images
img_width: 1024                                                         ## Input images width
img_height: 1024                                                        ## Input images height
divider: 64                                                             ## Automatically make width and height are dividable by divider
img_mean: [123.675, 116.28, 103.53]                                     ## Image normalization parameters
img_std: [58.395, 57.12, 57.375]                                        ## Image normalization parameters
to_rgb: 1                                                               ## RGB or BGR
keep_ratio: 1                                                           ## Keep ratio in original images

# augmentation
flip_ratio: 0.5                                                         ## Probability of image horizontal flip
expand_ratio: 0.0                                                       ## Probability of image expansion

# callbacks
save_every: 100                                                         ## Save model every <n> steps
keep_checkpoint_max: 5                                                  ## Max number of saved periodical checkpoints
keep_best_checkpoints_max: 5                                            ## Max number of saved best checkpoints
 ```

## [Training](#contents)

To train the model, run `train.py`.

### [Training process](#contents)

Standalone training mode:

```bash
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for these scripts.

* `CONFIG_PATH`: path to config file.
* `TRAIN_DATA`: path to train dataset.
* `VAL_DATA`: path to validation dataset.
* `TRAIN_OUT`: path to folder with training experiments.
* `BRIEF`: short experiment name.
* `PRETRAINED_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

Training result will be stored in the current path, whose folder name is "LOG".
Under this, you can find checkpoint files together with result like the
following in log.

```log
[2023-05-18T21:49:49.384] INFO: Creating network...
[2023-05-18T21:49:51.379] INFO: Load backbone weights...
[2023-05-18T21:49:52.018] INFO: Number of parameters: 42069344
[2023-05-18T21:49:52.019] INFO: Device type: GPU
[2023-05-18T21:49:52.019] INFO: Creating criterion, lr and opt objects...
[2023-05-18T21:49:52.246] INFO: Done!

[2023-05-18T21:49:52.766] WARNING: Directory already exists: /data/libra_rcnn/models/230518_LibraRCNN_bs-2_512x512_reduced/best_ckpt
[2023-05-18T21:52:13.548] INFO: epoch: 1, step: 100, loss: 3.225108,  lr: 0.002008
[2023-05-18T21:53:37.952] INFO: epoch: 1, step: 200, loss: 1.598624,  lr: 0.004006
[2023-05-18T21:55:01.839] INFO: epoch: 1, step: 300, loss: 1.696110,  lr: 0.006004
[2023-05-18T21:56:26.139] INFO: epoch: 1, step: 400, loss: 1.745857,  lr: 0.008002
[2023-05-18T21:57:49.916] INFO: epoch: 1, step: 500, loss: 1.739790,  lr: 0.010000
[2023-05-18T21:59:13.319] INFO: epoch: 1, step: 600, loss: 1.732734,  lr: 0.010000
[2023-05-18T22:00:36.708] INFO: epoch: 1, step: 700, loss: 1.683713,  lr: 0.010000
[2023-05-18T22:01:59.972] INFO: epoch: 1, step: 800, loss: 1.686329,  lr: 0.010000
[2023-05-18T22:03:23.391] INFO: epoch: 1, step: 900, loss: 1.625534,  lr: 0.010000
[2023-05-18T22:04:46.495] INFO: epoch: 1, step: 1000, loss: 1.620223,  lr: 0.010000
[2023-05-18T22:06:09.919] INFO: epoch: 1, step: 1100, loss: 1.617684,  lr: 0.010000
[2023-05-18T22:07:33.220] INFO: epoch: 1, step: 1200, loss: 1.552914,  lr: 0.010000
[2023-05-18T22:08:56.358] INFO: epoch: 1, step: 1300, loss: 1.587838,  lr: 0.010000
[2023-05-18T22:10:19.755] INFO: epoch: 1, step: 1400, loss: 1.461630,  lr: 0.010000
[2023-05-18T22:11:42.861] INFO: epoch: 1, step: 1500, loss: 1.487898,  lr: 0.010000
[2023-05-18T22:13:05.979] INFO: epoch: 1, step: 1600, loss: 1.505629,  lr: 0.010000
[2023-05-18T22:14:29.166] INFO: epoch: 1, step: 1700, loss: 1.555685,  lr: 0.010000
[2023-05-18T22:15:52.469] INFO: epoch: 1, step: 1800, loss: 1.526768,  lr: 0.010000
[2023-05-18T22:17:15.822] INFO: epoch: 1, step: 1900, loss: 1.489634,  lr: 0.010000
[2023-05-18T22:18:39.015] INFO: epoch: 1, step: 2000, loss: 1.461078,  lr: 0.010000
[2023-05-18T22:20:02.259] INFO: epoch: 1, step: 2100, loss: 1.467376,  lr: 0.010000
[2023-05-18T22:21:25.447] INFO: epoch: 1, step: 2200, loss: 1.419187,  lr: 0.010000
[2023-05-18T22:22:48.878] INFO: epoch: 1, step: 2300, loss: 1.486784,  lr: 0.010000
[2023-05-18T22:24:11.913] INFO: epoch: 1, step: 2400, loss: 1.456151,  lr: 0.010000
[2023-05-18T22:25:35.149] INFO: epoch: 1, step: 2500, loss: 1.419397,  lr: 0.010000
[2023-05-18T22:39:28.004] INFO: Eval epoch time: 832394.916 ms, per step time: 166.479 ms
[2023-05-18T22:39:51.987] INFO: Result metrics for epoch 1: {'loss': 1.6339870454788208, 'mAP': 0.009215981141746316}
[2023-05-18T22:39:51.993] INFO: Train epoch time: 2998957.220 ms, per step time: 1199.583 ms
...
[2023-05-19T06:47:39.084] INFO: epoch: 12, step: 100, loss: 0.695335,  lr: 0.000100
[2023-05-19T06:49:02.144] INFO: epoch: 12, step: 200, loss: 0.690558,  lr: 0.000100
[2023-05-19T06:50:25.223] INFO: epoch: 12, step: 300, loss: 0.690762,  lr: 0.000100
[2023-05-19T06:51:48.427] INFO: epoch: 12, step: 400, loss: 0.671814,  lr: 0.000100
[2023-05-19T06:53:11.612] INFO: epoch: 12, step: 500, loss: 0.672740,  lr: 0.000100
[2023-05-19T06:54:34.807] INFO: epoch: 12, step: 600, loss: 0.638905,  lr: 0.000100
[2023-05-19T06:55:57.938] INFO: epoch: 12, step: 700, loss: 0.646298,  lr: 0.000100
[2023-05-19T06:57:21.246] INFO: epoch: 12, step: 800, loss: 0.674287,  lr: 0.000100
[2023-05-19T06:58:44.266] INFO: epoch: 12, step: 900, loss: 0.706308,  lr: 0.000100
[2023-05-19T07:00:07.285] INFO: epoch: 12, step: 1000, loss: 0.697345,  lr: 0.000100
[2023-05-19T07:01:30.193] INFO: epoch: 12, step: 1100, loss: 0.706317,  lr: 0.000100
[2023-05-19T07:02:53.249] INFO: epoch: 12, step: 1200, loss: 0.683363,  lr: 0.000100
[2023-05-19T07:04:16.357] INFO: epoch: 12, step: 1300, loss: 0.683934,  lr: 0.000100
[2023-05-19T07:05:39.291] INFO: epoch: 12, step: 1400, loss: 0.703060,  lr: 0.000100
[2023-05-19T07:07:02.387] INFO: epoch: 12, step: 1500, loss: 0.663818,  lr: 0.000100
[2023-05-19T07:08:25.507] INFO: epoch: 12, step: 1600, loss: 0.672882,  lr: 0.000100
[2023-05-19T07:09:48.659] INFO: epoch: 12, step: 1700, loss: 0.655955,  lr: 0.000100
[2023-05-19T07:11:11.697] INFO: epoch: 12, step: 1800, loss: 0.661157,  lr: 0.000100
[2023-05-19T07:12:34.856] INFO: epoch: 12, step: 1900, loss: 0.680200,  lr: 0.000100
[2023-05-19T07:13:58.116] INFO: epoch: 12, step: 2000, loss: 0.612170,  lr: 0.000100
[2023-05-19T07:15:21.155] INFO: epoch: 12, step: 2100, loss: 0.651833,  lr: 0.000100
[2023-05-19T07:16:44.153] INFO: epoch: 12, step: 2200, loss: 0.687610,  lr: 0.000100
[2023-05-19T07:18:07.285] INFO: epoch: 12, step: 2300, loss: 0.660294,  lr: 0.000100
[2023-05-19T07:19:30.352] INFO: epoch: 12, step: 2400, loss: 0.670971,  lr: 0.000100
[2023-05-19T07:20:53.502] INFO: epoch: 12, step: 2500, loss: 0.687066,  lr: 0.000000
[2023-05-19T07:34:14.680] INFO: Eval epoch time: 800207.471 ms, per step time: 160.041 ms
[2023-05-19T07:34:49.118] INFO: Result metrics for epoch 12: {'loss': 0.674599184089899, 'mAP': 0.1796725043650461}
[2023-05-19T07:34:49.125] INFO: Train epoch time: 2912752.065 ms, per step time: 1165.101 ms
```

### [Transfer Training](#contents)

You can train your own model based on either pretrained classification model
or pretrained detection model. You can perform transfer training by following
steps.

1. Prepare your dataset.
1. Change configuraino YAML file according to your own dataset, especially the
 change `num_classes` value and `coco_classes` list.
1. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by
 `pretrained` argument. Transfer training means a new training job, so just set
 `finetune` 1.
1. Run training.

### [Distribute training](#contents)

Distribute training mode (OpenMPI must be installed):

```shell
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for this scripts:

* `CONFIG_PATH`: path to config file.
* `DEVICE_NUM`: number of devices.
* `TRAIN_DATA`: path to train dataset.
* `VAL_DATA`: path to validation dataset.
* `TRAIN_OUT`: path to folder with training experiments.
* `BRIEF`: short experiment name.
* `PRETRAINED_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

Sample:

```bash
bash scripts/run_distribute_train_gpu.sh configs/resnet50_train_config.yaml 4 /data/coco-2017/train/ /data/coco_mindrecord/validation/ /experiments/libra_rcnn_models/ gpu-4_1024x1024_bs-4_noised-ckpt_lr-2e-3_epoch-12 libra_faster_rcnn_r50_fpn_1x_coco_noised.ckpt
```

## [Evaluation](#contents)

### [Evaluation process](#contents)

#### [Evaluation on GPU](#contents)

```shell
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (Optional)[PREDICTION_PATH]
```

We need four parameters for this scripts.

* `CONFIG_PATH`: path to config file.
* `VAL_DATA`: the absolute path for dataset subset (validation).
* `CHECKPOINT_PATH`: path to checkpoint.
* `PREDICTION_PATH`: path to file with predictions JSON file (predictions may
 be saved to this file and loaded after).

> checkpoint can be produced in training process.

### [Evaluation result](#contents)

Result for GPU:

```log
CHECKING MINDRECORD FILES DONE!
Start Eval!
loading annotations into memory...
Done (t=0.48s)
creating index...
index created!

========================================

total images num:  5000
Processing, please wait a moment.
100%|██████████| 5000/5000 [1:02:34<00:00,  1.33it/s]
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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.632
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.705
Eval result: 0.42353085741140223

Evaluation done!

Done!
Time taken: 3797 seconds
```

## [Inference](#contents)

### [Inference process](#contents)

#### [Inference on GPU](#contents)

Run model inference from libra_rcnn directory:

```bash
bash scripts/run_infer_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH] [PRED_INPUT] [PRED_OUTPUT]
```

We need 4 parameters for these scripts:

* `CONFIG_FILE`： path to config file.
* `CHECKPOINT_PATH`: path to saved checkpoint.
* `PRED_INPUT`: path to input folder or image.
* `PRED_OUTPUT`: path to output JSON file.

### [Inference result](#contents)

Predictions will be saved in JSON file. File content is list of predictions
for each image. It's supported predictions for folder of images (png, jpeg
file in folder root) and single image.

Typical outputs of such script for single image:

```log
{
 "/data/coco-2017/validation/data/000000110042.jpg": {
  "height": 640,
  "width": 425,
  "predictions": [
   {
    "bbox": {
     "x_min": 136.0658416748047,
     "y_min": 440.034423828125,
     "width": 120.91867065429688,
     "height": 171.26031494140625
    },
    "class": {
     "label": 61,
     "category_id": "unknown",
     "name": "toilet"
    },
    "score": 0.9828336238861084
   },
   {
    "bbox": {
     "x_min": 1.0483801364898682,
     "y_min": 156.45419311523438,
     "width": 28.757143020629883,
     "height": 187.60565185546875
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "score": 0.9815477132797241
   },
   ...
   {
    "bbox": {
     "x_min": 2.246119976043701,
     "y_min": 45.19614791870117,
     "width": 47.0843391418457,
     "height": 137.6111602783203
    },
    "class": {
     "label": 25,
     "category_id": "unknown",
     "name": "umbrella"
    },
    "score": 0.05292666703462601
   }
  ]
 }
}
```

Typical outputs for folder with images:

```log
{
 "/data/coco-2017/validation/data/000000194832.jpg": {
  "height": 425,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 282.4643249511719,
     "y_min": 93.33438110351562,
     "width": 59.890960693359375,
     "height": 38.003662109375
    },
    "class": {
     "label": 62,
     "category_id": "unknown",
     "name": "tv"
    },
    "score": 0.9436116814613342
   },
   {
    "bbox": {
     "x_min": 8.066539764404297,
     "y_min": 0.6642341613769531,
     "width": 632.9083862304688,
     "height": 351.5968933105469
    },
    "class": {
     "label": 6,
     "category_id": "unknown",
     "name": "train"
    },
    "score": 0.7751350402832031
   },
  ...
   {
    "bbox": {
     "x_min": 255.4423828125,
     "y_min": 91.15775299072266,
     "width": 89.24874877929688,
     "height": 55.54553985595703
    },
    "class": {
     "label": 62,
     "category_id": "unknown",
     "name": "tv"
    },
    "score": 0.05151025578379631
   }
  ]
 },
 "/data/coco-2017/validation/data/000000104572.jpg": {
  "height": 419,
  "width": 640,
  "predictions": [
   ...
   }
  ]
 },
 ...
 "/data/coco-2017/validation/data/000000533855.jpg": {
  "height": 428,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 371.26348876953125,
     "y_min": 115.46067810058594,
     "width": 269.73651123046875,
     "height": 242.7736053466797
    },
    "class": {
     "label": 54,
     "category_id": "unknown",
     "name": "donut"
    },
    "score": 0.9951920509338379
   },
   ...
   {
    "bbox": {
     "x_min": 0.0,
     "y_min": 0.0,
     "width": 641.0,
     "height": 424.9591979980469
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "score": 0.06853365153074265
   }
  ]
 }
}
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          |  GPU                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| Model Version       | Libra RCNN ResNet50                                                                                         |
| Resource            | NVIDIA GeForce RTX 3090 (x4)                                                                                |
| Uploaded Date       | 07/03/2023 (month/day/year)                                                                                 |
| MindSpore Version   | 2.0.0.dev20230609                                                                                           |
| Dataset             | COCO2017                                                                                                    |
| Pretrained          | noised checkpoint (mAP=36.5%) |
| Training Parameters | epoch = 12,  batch_size = 4 (per device)                                                                    |
| Optimizer           | SGD (momentum)                                                                                              |
| Loss Function       | Sigmoid Cross Entropy, L1Loss, SoftMax Cross Entropy, BalancedL1Loss                                        |
| Speed               | 4pcs: 1108.9ms/step                                                                                         |
| Total time          | 4pcs: 31h 42m 34s                                                                                           |
| outputs             | mAP                                                                                                         |
| mAP                 | 38.8                                                                                                        |
| Model for inference | 470.4M(.ckpt file)                                                                                          |
| configuration       | resnet50_train_config.yaml                                                                                  |
| Scripts             |                                                                                                             |

## [Description of Random Situation](#contents)

We use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
