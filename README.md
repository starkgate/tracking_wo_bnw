# Tracking without bells and whistles

## What's new in this fork?

I modified Tracktor to support arbitrary mmdetection models instead of only the default torchvision Faster-RCNN. I also made changes to fix issues encountered when using my custom dataset.

My goal is to be able use Tracktor with an arbitrary pretrained object detection model on an arbitrary dataset (not just MOT).

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

- Install [mmdetection](https://github.com/open-mmlab/mmdetection) 2.3.0+ with mmcv 1.1.4+
- Train an RCNN-based object detection model per mmdetection's instructions

## Install

```bash
cd ~/mmdetection
git clone https://github.com/starkgate/tracking_wo_bnw
cd tracking_wo_bnw
pip install -r requirements.txt
pip install -e .
```

## Dataset

I was too lazy to make the names dynamic. It works, but you'll have to use hardcoded paths. I welcome pull requests!

Your dataset needs to have the following folder structure:

```bash
tracking_wo_bnw
├── data
│   └── MOT17Det
│       ├── test
│       │   └── rosbag
│       └── train
│           └── rosbag
│               └── img
│           ├── moc.txt
│           └── seqinfo.ini
```

`seqinfo.ini` contains metadata on the dataset (remove the comments in your file):

```ini
[Sequence]
name=MOT17-02-FRCNN # name of the dataset
imDir=img # folder where the image frames are located
frameRate=5 # frame rate of your video
seqLength=82 # how many frames
imWidth=640 # width of the images (needs to be constant)
imHeight=480 # height
imExt=.png # extension of the images
labels=~/mmdetection/tracking_wo_bnw/data/MOT17Det/train/rosbag/moc.txt # location of the groundtruth for evaluation
```

`moc.txt` contains the dataset's groundtruth detections, for evaluation:

```
# 1 line per object_id per frame
<frame> <object_id> <x> <y> <w> <h> 1 -1 -1 -1
# example
1 3 423.0 128.0 58.0 59.0 1 -1 -1 -1
1 4 486.0 175.0 49.0 31.0 1 -1 -1 -1
1 7 272.0 52.0 18.0 14.0 1 -1 -1 -1
2 19 144.0 44.0 26.0 34.0 1 -1 -1 -1
3 20 110.0 52.0 22.0 17.0 1 -1 -1 -1
3 30 333.0 39.0 12.0 11.0 1 -1 -1 -1
```

`1 -1 -1 -1` is only used for groundtruth

## Test

Tracktor can be configured by changing the corresponding `experiments/cfgs/tracktor.yaml` config file. Settings include thresholds for detection and reidentification.

```
~/mmdetection/tracking_wo_bnw/experiments/scripts/test_tracktor.py \
	--config ~/mmdetection/work_dirs/your_model/your_model.py \
	--checkpoint ~/mmdetection/work_dirs/your_model/latest.pth
```

## MOTChallenge data and ReID weights

1. MOTChallenge data:
    1. Download [MOT17Det](https://motchallenge.net/data/MOT17Det.zip), [MOT16Labels](https://motchallenge.net/data/MOT16Labels.zip), [2DMOT2015](https://motchallenge.net/data/2DMOT2015.zip), [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) and place them in the `data` folder. As the images are the same for MOT17Det, MOT17 and MOT16 we only need one set of images for all three benchmarks.
    2. Unzip all the data by executing:
    ```
    unzip -d MOT17Det MOT17Det.zip
    unzip -d MOT16Labels MOT16Labels.zip
    unzip -d 2DMOT2015 2DMOT2015.zip
    unzip -d MOT16-det-dpm-raw MOT16-det-dpm-raw.zip
    unzip -d MOT17Labels MOT17Labels.zip
    ```

2. Download object detector and re-identification Siamese network weights and MOTChallenge result files:
    1. Download zip file from [here](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v2.zip).
    2. Extract in `output` directory.

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
