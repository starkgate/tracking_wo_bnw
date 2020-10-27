import os

import motmetrics as mm
import numpy as np
from torch.utils.data import DataLoader
from tracktor.datasets.custom_sequence import CustomSequence

mm.lap.default_solver = 'lap'

from tqdm import tqdm
from sacred import Experiment
from tracktor.config import get_output_dir
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.model import Model
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

import argparse
import os.path as osp
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
print('Using device:', device)

ex = Experiment()

ex.add_config('tracking_wo_bnw/experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'tracking_wo_bnw/experiments/cfgs/oracle_tracktor.yaml')

tracktor = ex.configurations[0].__call__()['tracktor']
reid = ex.configurations[1].__call__()['reid']

# set all seeds
torch.manual_seed(tracktor['seed'])
torch.cuda.manual_seed(tracktor['seed'])
np.random.seed(tracktor['seed'])
torch.backends.cudnn.deterministic = True

output_dir = tracktor['output_dir']
sacred_config = osp.join(output_dir, 'sacred_config.yaml')

if not osp.exists(output_dir):
    os.makedirs(output_dir)

##########################
# Initialize the modules #
##########################

args = parse_args()

# object detection
print("Initializing object detector.")

obj_detect = Model(args.config, args.checkpoint, device)

# reid
reid_network = resnet50(pretrained=False, **reid['cnn'])
reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                        map_location=lambda storage, loc: storage))
reid_network.eval()
reid_network.cuda()

# tracktor
if 'oracle' in tracktor:
    tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
else:
    tracker = Tracker(obj_detect, reid_network, tracktor['tracker'], device)

time_total = 0
num_frames = 0
mot_accums = []

dataset = CustomSequence(cfg=tracktor)
for seq in dataset:
    tracker.reset()

    start = time.time()

    print(f"Tracking: {seq}")

    data_loader = DataLoader(seq, batch_size=1, shuffle=False)
    for i, frame in enumerate(tqdm(data_loader)):
        if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
            with torch.no_grad():
                tracker.step(frame)
            num_frames += 1
    results = tracker.get_results()

    time_total += time.time() - start

    print(f"Tracks found: {len(results)}")
    print(f"Runtime for {seq}: {time.time() - start :.2f} s.")

    if tracktor['interpolate']:
        results = interpolate(results)

    if seq.no_gt:
        print(f"No GT data for evaluation available.")
    else:
        mot_accums.append(get_mot_accum(results, seq))

    print(f"Writing predictions to: {output_dir}")
    seq.write_results(results, output_dir)

    if tracktor['write_images']:
        plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

print(f"Tracking runtime for all sequences (without evaluation or image writing): "
      f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
if mot_accums:
    evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)
