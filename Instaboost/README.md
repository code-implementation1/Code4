# Contents

* [Contents](#contents)
    * [Instaboost Description](#instaboost-description)
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
   * [Description of Random Situation](#description-of-random-situation)
   * [ModelZoo Homepage](#modelzoo-homepage)

## [Instaboost Description](#contents)

Instance segmentation requires a large number of training samples to achieve satisfactory performance and benefits from proper data augmentation. To enlarge the training set and increase the diversity, previous methods have investigated using data annotation from other domain (e.g. bbox, point) in a weakly supervised mechanism. In paper, they present a simple, efficient and effective method to augment the training set using the existing instance mask annotations. Exploiting the pixel redundancy of the background, they are able to improve the performance of Mask R-CNN for 1.7 mAP on COCO dataset and 3.3 mAP on Pascal VOC dataset by simply introducing random jittering to objects. Furthermore, they propose a location probability map based approach to explore the feasible locations that objects can be placed based on local appearance similarity. With the guidance of such map, they boost the performance of R101-Mask R-CNN on instance segmentation from 35.7 mAP to 37.9 mAP without modifying the backbone or network structure. Our method is simple to implement and does not increase the computational complexity. It can be integrated into the training pipeline of any instance segmentation model without affecting the training and inference efficiency.

[Paper](https://arxiv.org/abs/1908.07801): Hao-Shu Fang, Jianhua Sun, Runzhong Wang, Minghao Gou, Yong-Lu Li, Cewu Lu. Computer Vision and Pattern
Recognition (CVPR), 2019 (In press).

### [Model Architecture](#contents)

**Overview of the pipeline of ResNet MaskRCNN:**
MaskRCNN is a conceptually simple, flexible, and general framework for object
instance segmentation. The approach efficiently detects objects in an image
while simultaneously generating a high-quality segmentation mask for each
instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a
branch for predicting an object mask in parallel with the existing branch for
bounding box recognition. Mask R-CNN is simple to train and adds only a small
overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to
generalize to other tasks, e.g., allowing to estimate human poses in the same
framework.
It shows top results in all three tracks of the COCO suite of challenges,
including instance segmentation, bounding box object detection, and person
keypoint detection. Without bells and whistles, Mask R-CNN outperforms all
existing, single-model entries on every task, including the COCO 2016
challenge winners.

Region proposals are obtained from RPN and used for RoI feature extraction
from the output feature maps of a CNN backbone. The RoI features are used to
perform classification and localization and mask computation.

MaskRCNN result prediction pipeline:

1. ResNet backbone.
2. RPN.
3. Proposal generator.
4. ROI extractor (based on ROIAlign operation).
5. Bounding box head.
6. multiclass NMS (reduce number of proposed boxes and omit objects with low
 confidence).
7. ROI extractor (based on ROIAlign operation).
8. Mask head.

MaskRCNN result training pipeline:

1. ResNet backbone.
2. RPN.
3. RPN Assigner+Sampler.
4. RPN Classification + Localization losses.
5. Proposal generator.
6. RCNN Assigner+Sampler.
7. ROI extractor (based on ROIAlign operation).
8. Bounding box head.
9. RCNN Classification + Localization losses.
10. ROI extractor (based on ROIAlign operation).
11. Mask head.
12. ROI extractor (based on ROIAlign operation).
13. Mask loss.

**Overview of the pipeline of ResNet Cascade MaskRCNN:**
In object detection, an intersection over union (IoU) threshold is required to
define positives and negatives. An object detector, trained with low IoU threshold,
e.g. 0.5, usually produces noisy detections. However, detection performance tends
to degrade with increasing the IoU thresholds. Two main factors are responsible
for this: 1) overfitting during training, due to exponentially vanishing
positive samples, and 2) inference-time mismatch between the IoUs for which
the detector is optimal and those of the input hypotheses. A multi-stage
object detection architecture, the Cascade R-CNN, is proposed to address
these problems. It consists of a sequence of detectors trained with increasing
IoU thresholds, to be sequentially more selective against close false positives.
The detectors are trained stage by stage, leveraging the observation that the
output of a detector is a good distribution for training the next higher quality
detector. The resampling of progressively improved hypotheses guarantees that all
detectors have a positive set of examples of equivalent size, reducing the
overfitting problem. The same cascade procedure is applied at inference, enabling
a closer match between the hypotheses and the detector quality of each stage.
A simple implementation of the Cascade R-CNN is shown to surpass all single-model
object detectors on the challenging COCO dataset. Experiments also show that
the Cascade R-CNN is widely applicable across detector architectures, achieving consistent gains
independently of the baseline detector strength.

Cascade MaskRCNN result prediction pipeline (training is not implemented):

1. ResNet backbone.
2. RPN.
3. Proposal generator.
4. ROI extractor (based on ROIAlign operation).
5. Bounding box head 0.
6. ROI extractor (based on ROIAlign operation).
7. Bounding box head 1.
8. ROI extractor (based on ROIAlign operation).
9. Bounding box head 2.
10. Calculating the average classification logits for the three previous outputs from bbox heads.
11. Multiclass NMS (reduce number of proposed boxes and omit objects with low
 confidence).
12. ROI extractor (based on ROIAlign operation).
13. Mask head 0.
14. ROI extractor (based on ROIAlign operation).
15. Mask head 1.
16. ROI extractor (based on ROIAlign operation).
17. Mask head 2.
18. Calculating the average mask logits for the three previous outputs from mask heads.
19. Choose final masks from average mask logits.

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
numpy~=1.21.2
opencv-python~=4.5.4.58
pycocotools>=2.0.5
matplotlib
seaborn
pandas
instaboostfast
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
  └─ Instaboost
    ├─ configs
    │  ├── default_config.yaml                                ## default config
    │  ├── cascade_mask_rcnn_r50_fpn.yaml                     ## cascade mask rcnn config
    │  ├── mask_rcnn_r50_1024_dist_noised_experiment.yaml     ## mask rcnn resnet50 config which was used for training from noise checkpoint
    │  ├── mask_rcnn_r50_fpn.yaml                             ## mask rcnn resnet50 config
    │  ├── mask_rcnn_r101_fpn.yaml                            ## mask rcnn resnet101 config
    │  └── mask_rcnn_x_101_64x4d_fpn.yaml                     ## mask rcnn resnext101 config
    ├─ scripts
    │  ├── build_custom.sh              ## build dependencies
    │  ├── run_distribute_train_gpu.sh  ## bash script for distributed on gpu
    │  ├── run_eval_gpu.sh              ## bash script for eval on gpu
    │  ├── run_infer_gpu.sh             ## bash script for gpu model inference
    │  └── run_standalone_train_gpu.sh  ## bash script for distributed on gpu
    ├── src
    │   ├── blocks
    │   │   ├── anchor_generator
    │   │   │   ├── anchor_generator.py                         ## Anchor generator.
    │   │   │   └── __init__.py
    │   │   ├── assigners_samplers
    │   │   │   ├── assigner_sampler.py                         ## Wrapper for assigner and sampler.
    │   │   │   ├── __init__.py
    │   │   │   ├── mask_assigner_sampler.py                    ## Wrapper for assigner and sampler working with masks too.
    │   │   │   ├── max_iou_assigner.py                         ## MaxIOUAssigner.
    │   │   │   └── random_sampler.py                           ## Random Sampler.
    │   │   ├── backbones
    │   │   │   ├── __init__.py
    │   │   │   ├── resnet.py                                   ## Implemented ResNet backbone.
    │   │   │   └── resnext.py                                  ## Implemented ResNext backbone.
    │   │   ├── bbox_coders
    │   │   │   ├── delta_xywh_bbox_coder.py                    ## Bounding box coder.
    │   │   │   └── __init__.py
    │   │   ├── bbox_heads
    │   │   │   ├── shared_2fc_bbox_head.py                     ## Bounding box head.
    │   │   │   └── __init__.py
    │   │   ├── dense_heads
    │   │   │   ├── __init__.py
    │   │   │   ├── proposal_generator.py                       ## Proposal generator (part of RPN).
    │   │   │   └── rpn.py                                      ## Region Proposal Network.
    │   │   ├── detectors
    │   │   │   ├── __init__.py
    │   │   │   ├── cascade_mask_rcnn.py                        ## Cascade MaskRCNN model.
    │   │   │   └── mask_rcnn.py                                ## MaskRCNN model.
    │   │   ├── initialization
    │   │   │   ├── initialization.py                           ## Weight initialization functional.
    │   │   │   └── __init__.py
    │   │   ├── layers
    │   │   │   ├── conv_module.py                              ## Convolution module.
    │   │   │   └── __init__.py
    │   │   ├── mask_heads
    │   │   │   ├── fcn_mask_head.py                            ## Mask head.
    │   │   │   └── __init__.py
    │   │   ├── necks
    │   │   │   ├── fpn.py                                      ## Feature Pyramid Network.
    │   │   │   └── __init__.py
    │   │   ├── roi_extractors
    │   │   │   ├── __init__.py
    │   │   │   ├── single_layer_roi_extractor.py               ## Single ROI Extractor.
    │   │   │   └── cuda  
    │   │   │      └── roi_align_cuda_kernel.cu                 ## RoIAlign CUDA code
    │   │   └── __init__.py
    │   ├── model_utils
    │   │   ├── config.py                                     ## Configuration file parsing utils.
    │   │   ├── device_adapter.py                             ## File to adapt current used devices.
    │   │   ├── __init__.py
    │   │   ├── local_adapter.py                              ## File to work with local devices.
    │   │   └── moxing_adapter.py                             ## File to work with model arts devices.
    │   ├── callback.py                                             ## Callbacks.
    │   ├── common.py                                               ## Common functional with common setting.
    │   ├── dataset.py                                              ## Images loading and preprocessing.
    │   ├── eval_utils.py                                           ## Evaluation metrics utilities.
    │   ├── __init__.py
    │   ├── lr_schedule.py                                          ## Optimizer settings.
    │   ├── mlflow_funcs.py                                         ## mlflow utilities.
    │   └── network_define.py                                       ## Model wrappers for training.
    ├── eval.py                                                       ## Run models evaluation.
    ├── infer.py                                                      ## Make predictions for models.
    ├── __init__.py
    ├── README.md                                                     ## Instaboost definition.
    ├── requirements.txt                                              ## Dependencies.
    └── train.py                                                      ## Train script.
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```shell
train_outputs: '/data/mask_rcnn_models'                         ## Path to folder with experiments  
brief: 'gpu-1_1024x1024'                                        ## Experiment short name
device_target: GPU                                              ## Device
mode: 'graph'
# ==============================================================================
detector: 'mask_rcnn'

# backbone
backbone:
  type: "resnet"                                                ## Backbone type (resnet or resnext)
  depth: 50                                                     ## Backbone depth
  pretrained: '/data/backbones/resnet50.ckpt'                   ## Path to pretrained backbone
  frozen_stages: -1                                             ## Number of frozen stages (use if backbone is pretrained, set -1 to not freeze)
  norm_eval: True                                               ## Whether make batch normalization work in eval mode all time (use if backbone is pretrained)
  num_stages: 4                                                 ## Number of resnet/resnext stages
  out_indices: [0, 1, 2, 3]                                     ## Define indices of resnet/resnext outputs
  groups: 1                                                     ## Group number in ResNext layers (only for resnext)
  base_width: 4                                                 ## Base width in ResNext layers (only for resnext)

# neck
neck:
  fpn:
    in_channels: [256, 512, 1024, 2048]                         ## Channels number for each input feature map (FPN)
    out_channels: 256                                           ## Channels number for each output feature map (FPN)
    num_outs: 5                                                 ## Number of output feature map (FPN)

# rpn
rpn:
  in_channels: 256                                              ## Input channels
  feat_channels: 256                                            ## Number of channels in intermediate feature map in RPN
  num_classes: 1                                                ## Output classes number
  bbox_coder:
    target_means: [0., 0., 0., 0.]                              ## Parameter for bbox encoding (RPN targets generation)
    target_stds: [1.0, 1.0, 1.0, 1.0]                           ## Parameter for bbox encoding (RPN targets generation)
  loss_cls:
    loss_weight: 1.0                                            ## RPN classification loss weight
  loss_bbox:
    loss_weight: 1.0                                            ## RPN localization loss weight
  anchor_generator:
    scales: [8]                                                 ## Anchor scales
    strides: [4, 8, 16, 32, 64]                                 ## Anchor ratios
    ratios: [0.5, 1.0, 2.0]                                     ## Anchor strides for each feature map

bbox_head:
  in_channels: 256                                              ## Number of input channels
  fc_out_channels: 1024                                         ## Number of intermediate channels before
  roi_feat_size: 7                                              ## Input feature map side length
  reg_class_agnostic: False
  bbox_coder:
    target_means: [0.0, 0.0, 0.0, 0.0]                          ## Bounding box coder parameter
    target_stds: [0.1, 0.1, 0.2, 0.2]                           ## Bounding box coder parameter
  loss_cls:
    loss_weight: 1.0                                            ## Classification loss weight
  loss_bbox:
    loss_weight: 1.0                                            ## Localization loss weight

mask_head:
  num_convs: 4                                                 ## Number of convolution layers
  in_channels: 256                                             ## Number of input channels
  conv_out_channels: 256                                       ## Number of intermediate layers output channels
  loss_mask:
    loss_weight: 1.0                                           ## Mask loss weight

# roi_align
roi:                                                                    ## RoI extractor parameters
  roi_layer: {type: 'RoIAlign', out_size: 7, sampling_ratio: 0}         ## RoI configuration
  out_channels: 256                                                     ## out roi channels
  featmap_strides: [4, 8, 16, 32]                                       ## strides for RoIAlign layer
  finest_scale: 56                                                      ## parameter that define roi level
  sample_num: 640

mask_roi:                                                               ## RoI extractor parameters for masks head
  roi_layer: {type: 'RoIAlign', out_size: 14, sampling_ratio: 0}        ## RoI configuration
  out_channels: 256                                                     ## out roi channels
  featmap_strides: [4, 8, 16, 32]                                       ## strides for RoIAlign layer
  finest_scale: 56                                                      ## parameter that define roi level
  sample_num: 128

train_cfg:
  rpn:
    assigner:
      pos_iou_thr: 0.7                                            ## IoU threshold for negative bboxes
      neg_iou_thr: 0.3                                            ## IoU threshold for positive bboxes
      min_pos_iou: 0.3                                            ## Minimum iou for a bbox to be considered as a positive bbox
      match_low_quality: True                                     ## Allow low quality matches
    sampler:
      num: 256                                                    ## Number of chosen samples
      pos_fraction: 0.5                                           ## Fraction of positive samples
      neg_pos_ub: -1                                              ## Max positive-negative samples ratio
      add_gt_as_proposals: False                                  ## Allow low quality matches
  rpn_proposal:
      nms_pre: 2000                                              ## max number of samples per level
      max_per_img: 1000                                          ## max number of output samples
      iou_threshold: 0.7                                         ## NMS threshold for proposal generator
      min_bbox_size: 0                                           ## min bboxes size
  rcnn:
      assigner:
        pos_iou_thr: 0.5                                            ## IoU threshold for negative bboxes
        neg_iou_thr: 0.5                                            ## IoU threshold for positive bboxes
        min_pos_iou: 0.5                                            ## Minimum iou for a bbox to be considered as a positive bbox
        match_low_quality: True                                     ## Allow low quality matches
      sampler:
        num: 512                                                    ## Number of chosen samples
        pos_fraction: 0.25                                          ## Fraction of positive samples
        neg_pos_ub: -1                                              ## Max positive-negative samples ratio
        add_gt_as_proposals: True                                   ## Allow low quality matches
      mask_size: 28                                                 ## Output mask size

test_cfg:
  rpn:
    nms_pre: 1000                                                 ## max number of samples per level
    max_per_img: 1000                                             ## max number of output samples
    iou_threshold: 0.7                                            ## NMS threshold for proposal generator
    min_bbox_size: 0                                              ## min bboxes size
  rcnn:
    score_thr: 0.05                                               ## Confidence threshold
    iou_threshold: 0.5                                            ## IOU threshold
    max_per_img: 100                                              ## Max number of output bboxes
    mask_thr_binary: 0.5                                          ## mask threshold for masks

# optimizer
opt_type: "sgd"                                                ## Optimizer type (sgd or adam)
lr: 0.02                                                       ## Base learning rate
min_lr: 0.0000001                                              ## Minimum learning rate
momentum: 0.9                                                  ## Optimizer parameter
weight_decay: 0.0001                                           ## Regularization
warmup_step: 500                                               ## Number of warmup steps
warmup_ratio: 0.001                                            ## Initial learning rate = base_lr * warmup_ratio
lr_steps: [32, 44]                                             ## Epochs numbers when learning rate is divided by 10 (for multistep lr_type)
lr_type: "multistep"                                           ## Learning rate scheduling type
grad_clip: 0                                                   ## Gradient clipping (set 0 to turn off)


# train
num_gts: 100                                                   ## Limiting the number of objects in an image
batch_size: 16                                                 ## Train batch size
test_batch_size: 1                                             ## Test batch size
loss_scale: 256                                                ## Loss scale
epoch_size: 48                                                 ## Number of epochs
run_eval: 1                                                    ## Whether evaluation or not
eval_every: 1                                                  ## Evaluation interval
enable_graph_kernel: 0                                         ## Turn on kernel fusion
finetune: 0                                                    ## Turn on finetune (for transfer learning)
datasink: 1                                                    ## Turn on data sink mode
pre_trained: ''                                                ## Path to pretrained model weights

#distribution training
run_distribute: False                                          ## Turn on distributed training
device_id: 0
device_num: 1                                                  ## Number of devices (only if distributed training turned on)
rank_id: 0

# Number of threads used to process the dataset in parallel
num_parallel_workers: 6
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
img_width: 1024                                                         ## Input images width
img_height: 1024                                                        ## Input images height
ratio_range: [0.5, 2.0]
mask_divider: 1                                                         ## Mask divider coefficient
divider: 64                                                             ## Automatically make width and height are dividable by divider
img_mean: [123.675, 116.28, 103.53]                                     ## Image normalization parameters
img_std: [58.395, 57.12, 57.375]                                        ## Image normalization parameters
to_rgb: 1                                                               ## RGB or BGR
keep_ratio: 1                                                           ## Keep ratio in original images

# augmentation
flip_ratio: 0.5                                                         ## Probability of image horizontal flip
expand_ratio: 0.0                                                       ## Probability of image expansion

# instaboost augmentation
use_instaboost_fast: True                                               ## True if use instaboostfast, False if use instaboost

action_candidate: ['normal', 'horizontal', 'skip']                      ## Tuple of action candidates. 'normal', 'horizontal', 'vertical', 'skip' are supported
action_prob: [1, 0, 0]                                                  ## Tuple of corresponding action probabilities. Should be the same length as action_candidate
scale: [0.8, 1.2]                                                       ## Tuple of (min scale, max scale)
dx: 15                                                                  ## The maximum x-axis shift will be (instance width) / dx
dy: 15                                                                  ## The maximum y-axis shift will be (instance height) / dy
theta: [-1, 1]                                                          ## Tuple of (min rotation degree, max rotation degree)
color_prob: 0.5                                                         ## The probability of images for color augmentation
heatmap_flag: False                                                     ## Whether to use heatmap guided
aug_prob: 0.5                                                           ## The probability of using instaboost, set 0.0 if you don't want to use instaboost augmentation

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
100%|██████████| 5000/5000 [1:52:25<00:00,  1.35it/s]

Loading and preparing results...
DONE (t=0.57s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=26.16s).
Accumulating evaluation results...
DONE (t=5.47s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
Bbox eval result: 0.4052420347672526
Loading and preparing results...
DONE (t=1.41s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=30.37s).
Accumulating evaluation results...
DONE (t=5.49s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.573
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.538
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
Segmentation eval result: 0.36359401393146984
Evaluation done!

Done!

Time taken: 1047 seconds
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
 "/data/coco-2017/validation/data/000000437898.jpg": {
  "height": 427,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 534.9273681640625,
     "y_min": 312.78668212890625,
     "width": 344.0379638671875,
     "height": 368.2354736328125
    },
    "class": {
     "label": 69,
     "category_id": "unknown",
     "name": "oven"
    },
    "mask": {
     "size": [
      427,
      640
     ],
     "counts": "clo6>Y;DiDP2Q;8L4N1N2O1O1O1O001O001O003M000O10000000000000000O100000N20000002M100O100O1O1EXMRFi2l9<O1O1O10000O100001L3D<I7K5VO\\E_NR;V1g0N200O2M200O100O101N101N101O00000O100O10000O101O0O100O100O2O000010O0001O0O100O2M3H^c2"
    },
    "score": 0.9942078590393066
   },
   {
    "bbox": {
     "x_min": 4.772834777832031,
     "y_min": 401.60430908203125,
     "width": 191.59494018554688,
     "height": 91.5548095703125
    },
    "class": {
     "label": 71,
     "category_id": "unknown",
     "name": "sink"
    },
    "mask": {
     "size": [
      427,
      640
     ],
     "counts": "SR24W=0000000000000O1000000000000000000O10000O1O100000000000000001O001O000001O00000000000O10000000000000001O000000000000000000001O000O10001O1O002N1O1O1O1O1O1O001O00001O00001O000000000000000000000O100000000000000000000000O10000000000000000000000000000000000000000000000O100000000000000000000000001O000000001O00003M1O0O101O00001O1O001O1O001O001O001O00001O00001O000O2O00001O1O0ObWk5"
    },
    "score": 0.9717132449150085
   },
  ...
   {
    "bbox": {
     "x_min": 497.52142333984375,
     "y_min": 376.84844970703125,
     "width": 61.8050537109375,
     "height": 258.5343017578125
    },
    "class": {
     "label": 69,
     "category_id": "unknown",
     "name": "oven"
    },
    "mask": {
     "size": [
      427,
      640
     ],
     "counts": "gd_64U<MkD4T;5VD:h;e001O00000001O01O0000001O0000010O000O2O0O10001O000000001O00000000001O0000000000000000WO[DDNHh;a0k0O1E[CNg<0;O1O100O1O10UmS1"
    },
    "score": 0.055059224367141724
   }
  ]
 }
}
```

Typical outputs for folder with images:

```log
{
 "/data/coco-2017/validation/data/000000437898.jpg": {
  "height": 427,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 534.9273681640625,
     "y_min": 312.78668212890625,
     "width": 344.0379638671875,
     "height": 368.2354736328125
    },
    "class": {
     "label": 69,
     "category_id": "unknown",
     "name": "oven"
    },
    "mask": {
     "size": [
      427,
      640
     ],
     "counts": "clo6>Y;DiDP2Q;8L4N1N2O1O1O1O001O001O003M000O10000000000000000O100000N20000002M100O100O1O1EXMRFi2l9<O1O1O10000O100001L3D<I7K5VO\\E_NR;V1g0N200O2M200O100O101N101N101O00000O100O10000O101O0O100O100O2O000010O0001O0O100O2M3H^c2"
    },
    "score": 0.9942078590393066
   },
   {
    "bbox": {
     "x_min": 4.772834777832031,
     "y_min": 401.60430908203125,
     "width": 191.59494018554688,
     "height": 91.5548095703125
    },
    "class": {
     "label": 71,
     "category_id": "unknown",
     "name": "sink"
    },
    "mask": {
     "size": [
      427,
      640
     ],
     "counts": "SR24W=0000000000000O1000000000000000000O10000O1O100000000000000001O001O000001O00000000000O10000000000000001O000000000000000000001O000O10001O1O002N1O1O1O1O1O1O001O00001O00001O000000000000000000000O100000000000000000000000O10000000000000000000000000000000000000000000000O100000000000000000000000001O000000001O00003M1O0O101O00001O1O001O1O001O001O001O00001O00001O000O2O00001O1O0ObWk5"
    },
    "score": 0.9717132449150085
   },
  ...
   {
    "bbox": {
     "x_min": 497.52142333984375,
     "y_min": 376.84844970703125,
     "width": 61.8050537109375,
     "height": 258.5343017578125
    },
    "class": {
     "label": 69,
     "category_id": "unknown",
     "name": "oven"
    },
    "mask": {
     "size": [
      427,
      640
     ],
     "counts": "gd_64U<MkD4T;5VD:h;e001O00000001O01O0000001O0000010O000O2O0O10001O000000001O00000000001O0000000000000000WO[DDNHh;a0k0O1E[CNg<0;O1O100O1O10UmS1"
    },
    "score": 0.055059224367141724
   }
  ]
 },
 "/data/data/coco-2017/validation/data/000000416758.jpg": {
  "height": 480,
  "width": 640,
  "predictions": [
   {
   ...
   }
  ]
 }
}
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | GPU                                                  |
|---------------------|------------------------------------------------------|
| Model Version       | Mask RCNN ResNet 50 1024x1024                        |
| Resource            | NVIDIA GeForce RTX 3090 (x4)                         |
| Uploaded Date       | 23/10/2023 (day/month/year)                          |
| MindSpore Version   | 2.1.0                                                |
| Dataset             | COCO2017                                             |
| Pretrained          | noised checkpoint (bbox_mAP=39.49%, segm_mAP=35.57%) |
| Training Parameters | epoch = 48, batch_size = 4                           |
| Optimizer           | SGD (momentum)                                       |
| Loss Function       | Sigmoid Cross Entropy, SoftMax Cross Entropy, L1Loss |
| Speed               | 4pcs: 1911.457 ms/step                               |
| Total time          | 4pcs: 4d 18h 31m 33s                                 |
| outputs             | mAP(bbox), mAP(segm)                                 |
| mAP(bbox)           | 40.5                                                 |
| mAP(segm)           | 36.4                                                 |
| Model for inference | 534.4M(.ckpt file)                                   |
| configuration       | mask_rcnn_r50_1024_dist_noised_experiment.yaml       |
| Scripts             |                                                      |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also set
random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).