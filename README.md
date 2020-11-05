# Tracking without bells and whistles

## What's new in this fork?

I modified Tracktor to support arbitrary mmdetection models instead of only the default torchvision Faster-RCNN. I also made changes to fix issues encountered when using my custom dataset.

The fork is meant to make testing arbitrary datasets (not just MOT) on pretrained mmdetection models easier.

- Removed torchvision Faster-RCNN implementation
- Replaced with mmdetection framework wrapper. Should support any mmdetection object detection model and checkpoints (tested with Faster-, Mask- and Cascade-RCNN)
- Added argumentparser
- Remove pinned versions for packages, for better forward compatibility
- Remove sacred experiments, just run the code
- Lowered default threshold to 0.3, consistent with mmdetection's default
- Set public_detections to False by default, since they're not available for my dataset and shouldn't be enabled by default
- Set write_images to True by default
- Replace .cuda() with .to(device)
- Miscellaneous crash fixes and cleanup

## Original description

This repository provides the implementation of our paper **Tracking without bells and whistles** (Philipp Bergmann, [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/)) [https://arxiv.org/abs/1903.05625]. This branch includes an updated version of Tracktor for PyTorch 1.3 with an improved object detector. The original results of the paper were produced with the `iccv_19` branch.

In addition to our supplementary document, we provide an illustrative [web-video-collection](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-supp_video_collection.zip). The collection includes examplary Tracktor++ tracking results and multiple video examples to accompany our analysis of state-of-the-art tracking methods.

![Visualization of Tracktor](data/method_vis_standalone.png)

## Requirements

- Install [mmdetection](https://github.com/open-mmlab/mmdetection) 2.3.0+ with mmcv 1.1.4+. The rest of the guide will assume you installed it in your home folder.
- Train an RCNN-based object detection model per mmdetection's instructions, or download a pre-trained model
- Download re-identification Siamese network weights:
    1. Download zip file from [here](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v2.zip).
    2. Extract in `output` directory.

## Install

```bash
cd ~/mmdetection
git clone https://github.com/starkgate/tracking_wo_bnw
cd tracking_wo_bnw
pip install -r requirements.txt
pip install -e .
```

## Dataset

You can configure your dataset in `tracking_wo_bnw/experiments/cfgs/tracktor.yaml`: change `data_path: ...` to wherever your dataset is located. Tracktor will look for the following files and folders in `data_path`:

```bash
data_path_location
├── img
└── seqinfo.ini
```

`seqinfo.ini` contains metadata on the dataset (remove the comments in your file):

```ini
[Sequence]
name=MOT17-02-FRCNN # name of the dataset
imDir=img # name of the folder where the image frames are located
frameRate=5 # frame rate of your video
seqLength=82 # how many frames
imWidth=640 # width of the images (needs to be constant)
imHeight=480 # height
imExt=.png # extension of the images
```

## Test

Tracktor can be configured by changing the corresponding `experiments/cfgs/tracktor.yaml` config file. Settings include thresholds for detection and reidentification. Tracktor also takes the following arguments:

```
~/mmdetection/tracking_wo_bnw/experiments/scripts/test_tracktor.py \
	--config ~/mmdetection/work_dirs/your_model/your_model.py \
	--checkpoint ~/mmdetection/work_dirs/your_model/latest.pth
```

## Training the reidentifaction model

1. The training config file is located at `experiments/cfgs/reid.yaml`.

2. Start training by executing:
  ```
  python experiments/scripts/train_reid.py
  ```

## Publication
 If you use this software in your research, please cite our publication:

```
  @InProceedings{tracktor_2019_ICCV,
  author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}, Laura},
  title = {Tracking Without Bells and Whistles},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}}
```
