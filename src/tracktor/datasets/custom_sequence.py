import configparser
import csv
import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset

class CustomSequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, cfg):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self.folder = cfg['data_path']

        assert os.path.exists(self.folder), 'Image set does not exist: {}'.format(self.folder)
        self.data, self.no_gt = self._sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]

        sample = {}
        sample['img'] = data
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def _sequence(self):
        config_file = osp.join(self.folder, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']
        imExt = config['Sequence']['imExt']
        labels = config['Sequence']['labels']

        imDir = osp.join(self.folder, imDir)
        gt_file = labels

        total = []

        visibility = {}
        boxes = {}

        # frames start at 0
        for i in range(1, seqLength + 1):
            boxes[i] = {}
            visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=' ')
                for row in reader:
                    frame = int(row[0])
                    bbox = int(row[1])
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(float(row[2])) - 1
                    y1 = int(float(row[3])) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(float(row[4])) - 1
                    y2 = y1 + int(float(row[5])) - 1
                    bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                    boxes[frame][bbox] = bb
                    visibility[frame][bbox] = float(row[8])
        else:
            no_gt = True

        for i in range(1, seqLength + 1):
            im_path = osp.join(imDir, "{:06d}{:s}".format(i, imExt))

            # we need 'filename', 'ori_filename'
            sample = {'gt': boxes[i],
                      'filename': im_path,
                      'ori_filename': im_path,
                      'im_path': im_path,
                      'vis': visibility[i]}

            total.append(sample)

        return total, no_gt

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file = osp.join(output_dir, 'output.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])